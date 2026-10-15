"""Workshop 3: a replication of Krauss, Do & Huck (2017), "Deep neural networks, gradient-boosted trees,
random forests: Statistical arbitrage on the S&P 500", European Journal of Operational Research 259(2).

The paper's recipe, and where this module differs:

* Features: 31 lagged returns per stock, R(m) = P_t / P_{t-m} - 1 for m = 1..20 and 40, 60, ..., 240.
* Target: 1 if the stock's NEXT-day return beats the cross-sectional median of that day, else 0.
* Study periods: 750 trading days to train, the next 250 to trade; then slide forward 250 days, so
  every trading day is out of sample exactly once. The paper retrains on each block; so do we.
* Trading: each day rank stocks by predicted probability, long the top k, short the bottom k, equal
  weights, close everything the next day. The paper reports returns for $1 long and $1 short, i.e.
  mean(long legs) - mean(short legs); `long_short` does the same.
* Costs: the paper charges 5 bps per half-turn; its 0.45% a day before costs falling to about 0.25%
  after implies the whole book is charged as turning over every day (20 bps a day). `long_short`
  charges cost on the turnover that actually happened, which is much cheaper (about 11 bps a day
  instead of 20 in our run) because many names stay in the book from one day to the next.
* Ensemble: like the paper, the plain average of the three ML models (forest, boosting, neural
  net). The logistic regression is our extra baseline and is not in the ensemble.

Differences, stated: our universe is today's S&P 100 (survivorship-biased), not the point-in-time
S&P 500; our models are smaller than the paper's so a laptop or Colab trains them in minutes; the
paper's deep net is approximated by scikit-learn's MLPClassifier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

LAGS: List[int] = list(range(1, 21)) + list(range(40, 241, 20))
FEATURES: List[str] = [f"R{m}" for m in LAGS]
TRAIN_DAYS, TRADE_DAYS = 750, 250
DATA_FILE = Path(__file__).resolve().parent.parent / "workshop3" / "data" / "sp100_daily.csv.gz"
TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- data

def load_prices(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """The committed snapshot as a long table: date, symbol, close, adj_close, volume."""
    df = pd.read_csv(path or DATA_FILE, parse_dates=["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def wide(long: pd.DataFrame, column: str = "adj_close") -> pd.DataFrame:
    """Dates down, symbols across. A symbol is NaN before it listed."""
    return long.pivot(index="date", columns="symbol", values=column).sort_index()


# --------------------------------------------------------------------------- features and target

def lagged_returns(prices: pd.DataFrame, lags: Sequence[int] = LAGS) -> pd.DataFrame:
    """The paper's 31 features as a panel: index (date, symbol), one column R{m} per lag.

    R{m} on date t uses prices up to and including t only, so it is known at t's close.
    A row is kept only when every lag is available (a stock needs 241 days of history).
    """
    cols = {f"R{m}": (prices / prices.shift(m) - 1.0).stack(future_stack=True) for m in lags}
    out = pd.DataFrame(cols)
    out.index.names = ["date", "symbol"]
    return out.dropna()


def next_day_return(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from t's close to t+1's close, stored on date t (the day the decision is made)."""
    return prices.shift(-1) / prices - 1.0


def median_target(prices: pd.DataFrame) -> pd.DataFrame:
    """1.0 if the stock's next-day return beats that day's cross-sectional median, 0.0 if not, NaN if unknown."""
    r = next_day_return(prices)
    med = r.median(axis=1)
    y = r.gt(med, axis=0).astype(float)
    return y.where(r.notna())


def make_panel(prices: pd.DataFrame, lags: Sequence[int] = LAGS) -> pd.DataFrame:
    """Features, next-day return `fwd` and label `y` in one table, index (date, symbol).

    The last date has features but no `fwd`/`y` yet (tomorrow has not happened); it stays in the
    table so the live signal can be read from the same object. Training code drops rows with no `y`.
    """
    X = lagged_returns(prices, lags)
    fwd = next_day_return(prices).stack(future_stack=True).rename("fwd")
    y = median_target(prices).stack(future_stack=True).rename("y")
    fwd.index.names = y.index.names = ["date", "symbol"]
    return X.join(fwd, how="left").join(y, how="left")


# --------------------------------------------------------------------------- study periods

@dataclass
class StudyPeriod:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp

    def __str__(self) -> str:
        f = lambda d: d.strftime("%Y-%m-%d")
        return f"period {self.number:2d}: train {f(self.train_start)} to {f(self.train_end)}, trade {f(self.trade_start)} to {f(self.trade_end)}"


def study_periods(dates: Iterable, train_days: int = TRAIN_DAYS, trade_days: int = TRADE_DAYS,
                  last_partial: bool = True) -> List[StudyPeriod]:
    """Non-overlapping trading blocks of `trade_days`, each preceded by `train_days` of training dates.

    Dates are the panel's dates that have features. Training and trading never share a date. The
    label of the last training day is the return from that day's close to the first trading day's
    close, which is known at the first trading day's close, the moment its first decision is made:
    nothing from a trading day's future leaks into its own model. The final block may be shorter
    (the most recent months) when `last_partial`.
    """
    d = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(dates)))))
    out, start, n = [], train_days, 1
    while start < len(d):
        end = min(start + trade_days, len(d))
        if end - start < trade_days and not last_partial:
            break
        out.append(StudyPeriod(n, d[start - train_days], d[start - 1], d[start], d[end - 1]))
        start, n = end, n + 1
    return out


# --------------------------------------------------------------------------- models

def make_model(name: str, preset: str = "workshop", seed: int = 0):
    """A fresh scikit-learn classifier.

    preset "workshop": small enough to retrain on every study period in a few minutes on Colab.
    preset "paper": the paper's sizes (RF 1000 trees depth 20; GBT 100 trees depth 3, lr 0.1,
    half the features per split; DNN 31-31-10-5). Much slower; for after the workshop.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    paper = preset == "paper"
    if preset not in ("workshop", "paper"):
        raise ValueError("preset must be 'workshop' or 'paper'")
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    if name == "forest":
        return RandomForestClassifier(n_estimators=1000 if paper else 150, max_depth=20 if paper else 10,
                                      max_features=5, min_samples_leaf=1 if paper else 100,
                                      max_samples=None if paper else 0.3, n_jobs=-1, random_state=seed)
    if name == "boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, max_features=0.5,
                                              early_stopping=False, random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(31, 10, 5), alpha=1e-3,
                                                             max_iter=400 if paper else 60, early_stopping=True,
                                                             random_state=seed))
    raise ValueError(f"unknown model {name!r}; use logit, forest, boosting or mlp")


MODEL_NAMES = ("logit", "forest", "boosting", "mlp")
ENSEMBLE_MODELS = ("forest", "boosting", "mlp")      # the paper averages its three ML models


def fit_predict_period(panel: pd.DataFrame, period: StudyPeriod, models: Sequence[str], preset: str = "workshop",
                       seed: int = 0, features: Sequence[str] = FEATURES) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train each model on the period's training rows, predict P(beats median) on its trading rows."""
    dates = panel.index.get_level_values("date")
    train = panel[(dates >= period.train_start) & (dates <= period.train_end)].dropna(subset=["y"])
    trade = panel[(dates >= period.trade_start) & (dates <= period.trade_end)]
    X, y = train[list(features)].to_numpy(), train["y"].to_numpy().astype(int)
    preds, fitted = {}, {}
    for name in models:
        m = make_model(name, preset, seed)
        m.fit(X, y)
        preds[name] = m.predict_proba(trade[list(features)].to_numpy())[:, 1]
        fitted[name] = m
    out = pd.DataFrame(preds, index=trade.index)
    out["period"] = period.number
    return out, fitted


def walk_forward(panel: pd.DataFrame, models: Sequence[str] = ("logit", "forest", "boosting"),
                 preset: str = "workshop", seed: int = 0, periods: Optional[Sequence[StudyPeriod]] = None,
                 progress: Optional[Callable[[str], None]] = print) -> pd.DataFrame:
    """Out-of-sample predictions for every trading day, one model retrained per study period.

    Adds `ensemble`, the plain average of the paper's three ML models, when all three ran (the
    logit baseline is left out, as in the paper). Returns index (date, symbol), one column per model.
    """
    periods = periods or study_periods(panel.index.get_level_values("date").unique())
    parts = []
    for p in periods:
        t0 = time.time()
        part, _ = fit_predict_period(panel, p, models, preset, seed)
        parts.append(part)
        if progress:
            progress(f"{p}  ({time.time() - t0:.0f}s)")
    out = pd.concat(parts).sort_index()
    if all(m in models for m in ENSEMBLE_MODELS):
        out["ensemble"] = out[list(ENSEMBLE_MODELS)].mean(axis=1)
    return out


# --------------------------------------------------------------------------- trading

def daily_weights(scores: pd.Series, k: int = 10) -> pd.Series:
    """+1/k for the k highest scores of each date, -1/k for the k lowest, 0 otherwise. Index (date, symbol).

    Ties are broken by row order, so the result is deterministic. A date with fewer than 2k
    scored stocks gets no positions.
    """
    s = scores.dropna()
    g = s.groupby(level="date")
    top = g.rank(method="first", ascending=False) <= k
    bottom = g.rank(method="first", ascending=True) <= k
    enough = g.transform("size") >= 2 * k
    w = pd.Series(0.0, index=s.index)
    w[top & enough] = 1.0 / k
    w[bottom & enough] = -1.0 / k
    return w


def banded_weights(scores: pd.Series, k: int = 10, exit_k: int = 20) -> pd.Series:
    """Like daily_weights, but a stock already held stays until it drops out of the best (worst) `exit_k`.

    New names enter from the top (bottom) of the ranking, and each leg is topped back up to k names.
    Fewer changes per day means less turnover and lower costs; the price is holding some names whose
    signal has faded. exit_k = k gives the same book as daily_weights, up to how ties are broken.
    """
    W = scores.dropna().unstack("symbol")
    vals = W.to_numpy()
    out = np.zeros_like(vals)
    held_long = np.zeros(vals.shape[1], bool)
    held_short = np.zeros(vals.shape[1], bool)
    for i, row in enumerate(vals):
        ok = ~np.isnan(row)
        if ok.sum() < 2 * k:
            held_long[:] = held_short[:] = False
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(-row[idx], kind="mergesort")]           # best first
        rank_hi = np.full(len(row), 10 ** 9)
        rank_hi[order] = np.arange(1, len(order) + 1)
        rank_lo = np.full(len(row), 10 ** 9)
        rank_lo[order[::-1]] = np.arange(1, len(order) + 1)
        held_long &= rank_hi <= exit_k
        held_short &= rank_lo <= exit_k
        for j in order:
            if held_long.sum() >= k:
                break
            if not held_short[j]:
                held_long[j] = True
        for j in order[::-1]:
            if held_short.sum() >= k:
                break
            if not held_long[j]:
                held_short[j] = True
        out[i, held_long] = 1.0 / k
        out[i, held_short] = -1.0 / k
    res = pd.DataFrame(out, index=W.index, columns=W.columns).where(W.notna())
    w = res.stack(future_stack=True).dropna()
    w.index.names = ["date", "symbol"]
    return w


@dataclass
class LongShort:
    gross: pd.Series            # daily return before costs
    net: pd.Series              # after costs
    turnover: pd.Series         # sum |w_t - w_{t-1}| per day (2.0 per leg would be a full rotation of one leg)
    long_leg: pd.Series
    short_leg: pd.Series
    cost_bps: float

    def window(self, start=None, end=None) -> "LongShort":
        """The same backtest restricted to dates from `start` to `end` (inclusive)."""
        cut = lambda s: s.loc[start:end]
        return LongShort(cut(self.gross), cut(self.net), cut(self.turnover), cut(self.long_leg),
                         cut(self.short_leg), self.cost_bps)

    def stats(self) -> Dict[str, float]:
        return {**{f"{k} (before costs)": v for k, v in perf_stats(self.gross).items() if k in ("mean daily", "annual return", "Sharpe")},
                **perf_stats(self.net), "daily turnover": float(self.turnover.mean())}


def long_short(scores: pd.Series, fwd: pd.Series, k: int = 10, cost_bps: float = 5.0,
               exit_k: Optional[int] = None) -> LongShort:
    """Backtest the paper's daily rule. `scores` and `fwd` share the index (date, symbol).

    Daily return = sum(w * next-day return) with w = +-1/k: the mean of the long leg minus the mean
    of the short leg. Cost = cost_bps per unit of turnover (each $ bought or sold is one half-turn).
    exit_k: keep held names until they leave the top/bottom exit_k (see banded_weights).
    """
    known = fwd.dropna().index.get_level_values("date").unique()      # the last date has no next-day return yet
    scores = scores[scores.index.get_level_values("date").isin(known)]
    w = daily_weights(scores, k) if exit_k is None or exit_k <= k else banded_weights(scores, k, exit_k)
    W = w.unstack("symbol").fillna(0.0)
    R = fwd.reindex(w.index).unstack("symbol").reindex(index=W.index, columns=W.columns).fillna(0.0)
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4
    long_leg = (W.clip(lower=0) * R).sum(axis=1)
    short_leg = (W.clip(upper=0) * R).sum(axis=1)
    return LongShort(gross, net, turnover, long_leg, short_leg, cost_bps)


def perf_stats(daily: pd.Series) -> Dict[str, float]:
    """Plain performance numbers for a daily return series (simple returns, 252 days a year)."""
    daily = daily.dropna()
    eq = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS_PER_YEAR
    sd = daily.std()
    return {"mean daily": float(daily.mean()),
            "annual return": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan"),
            "annual vol": float(sd * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "Sharpe": float(daily.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else float("nan"),
            "max drawdown": float((eq / eq.cummax() - 1).min()),
            "hit rate": float((daily > 0).sum() / max((daily != 0).sum(), 1)),    # share of days with a position that made money
            "days": int(len(daily))}


def by_period(daily: pd.Series, cuts: Mapping[str, Tuple[str, str]]) -> pd.DataFrame:
    """perf_stats for named date ranges, e.g. {"GFC": ("2008-09-01", "2009-12-31")}."""
    rows = {}
    for name, (a, b) in cuts.items():
        part = daily.loc[a:b]
        if len(part) > 20:
            rows[name] = perf_stats(part)
    return pd.DataFrame(rows).T


def yearly(daily: pd.Series) -> pd.Series:
    """Compounded return per calendar year."""
    return (1 + daily).groupby(daily.index.year).prod() - 1


# --------------------------------------------------------------------------- signal research

def daily_ic(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    """Spearman rank correlation between score and next-day return, one number per date (dates with < 5 stocks dropped)."""
    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    g = df.groupby(level="date")
    rk = g.rank()                                   # ranks within each date
    rk = rk - rk.groupby(level="date").transform("mean")
    num = (rk["s"] * rk["r"]).groupby(level="date").sum()
    den = np.sqrt((rk["s"] ** 2).groupby(level="date").sum() * (rk["r"] ** 2).groupby(level="date").sum())
    ic = num / den.replace(0, np.nan)
    return ic[g.size() >= 5].dropna()


def decile_returns(scores: pd.Series, fwd: pd.Series, n: int = 10, relative: bool = True) -> pd.Series:
    """Mean next-day return of each score decile, deciles formed within each date (1 = lowest score).

    relative=True subtracts each day's average return first, so the bars show return RELATIVE to the
    other stocks that day (what a long/short book earns) rather than the market's general drift.
    """
    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    if relative:
        df["r"] = df["r"] - df.groupby(level="date")["r"].transform("mean")
    rank = df.groupby(level="date")["s"].rank(method="first")
    size = df.groupby(level="date")["s"].transform("size")
    df["decile"] = np.ceil(rank / size * n).clip(1, n)
    return df[size >= n].groupby("decile")["r"].mean().rename(index=int)


# --------------------------------------------------------------------------- live use

@dataclass
class LiveStrategy:
    """What the dashboard needs: a trained model and the rule that turns its scores into weights."""
    model: object
    model_name: str
    k: int = 10
    gross_exposure: float = 1.0       # live: $0.50 long and $0.50 short per $1 of equity, unlike the backtest's $1 + $1
    features: List[str] = field(default_factory=lambda: list(FEATURES))
    trained_from: str = ""
    trained_to: str = ""
    symbols: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def history_days(self) -> int:
        """Trading days of prices needed so the longest lag (R240) exists on the last day."""
        return max(int(f[1:]) for f in self.features) + 1

    def scores(self, prices: pd.DataFrame) -> pd.Series:
        """P(beats tomorrow's median) for every symbol on the LAST date of `prices` (dates down, symbols across)."""
        lags = [int(f[1:]) for f in self.features]
        X = lagged_returns(prices, lags)
        if X.empty:
            raise ValueError(f"not enough history: need {self.history_days} trading days of prices")
        last = X.index.get_level_values("date").max()
        rows = X.xs(last, level="date")[self.features]
        p = self.model.predict_proba(rows.to_numpy())[:, 1]
        return pd.Series(p, index=rows.index, name=last).sort_values(ascending=False)

    def weights(self, scores: pd.Series) -> pd.Series:
        """Target weights (fraction of equity) for one day's scores: +g/2k for the top k, -g/2k for the bottom k."""
        s = scores.dropna().sort_values(kind="mergesort")
        if len(s) < 2 * self.k:
            raise ValueError(f"need at least {2 * self.k} scored stocks, got {len(s)}")
        w = pd.Series(0.0, index=s.index)
        each = self.gross_exposure / (2 * self.k)
        w[s.index[-self.k:]] = each
        w[s.index[:self.k]] = -each
        return w.sort_values(ascending=False)


def train_live(panel: pd.DataFrame, model_name: str = "forest", train_days: int = TRAIN_DAYS, k: int = 10,
               preset: str = "workshop", seed: int = 0, gross_exposure: float = 1.0) -> LiveStrategy:
    """Train on the most recent `train_days` labelled dates, as the paper would for the next trading block."""
    labelled = panel.dropna(subset=["y"])
    dates = labelled.index.get_level_values("date").unique().sort_values()[-train_days:]
    train = labelled[labelled.index.get_level_values("date").isin(dates)]
    m = make_model(model_name, preset, seed)
    m.fit(train[FEATURES].to_numpy(), train["y"].to_numpy().astype(int))
    syms = sorted(train.index.get_level_values("symbol").unique())
    return LiveStrategy(m, model_name, k=k, gross_exposure=gross_exposure, trained_from=str(dates[0].date()),
                        trained_to=str(dates[-1].date()), symbols=syms)


def save_strategy(strategy: LiveStrategy, path: Union[str, Path]) -> Path:
    import joblib
    import sklearn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"strategy": strategy, "sklearn": sklearn.__version__}, path)
    return path


def load_strategy(path: Union[str, Path]) -> LiveStrategy:
    """Load a strategy saved by save_strategy. Only load files you made yourself: joblib can run code."""
    import joblib
    import sklearn
    blob = joblib.load(path)
    if blob.get("sklearn") != sklearn.__version__:
        import warnings
        warnings.warn(f"strategy was saved with scikit-learn {blob.get('sklearn')}, this is {sklearn.__version__}")
    return blob["strategy"]


CACHE_FILE = DATA_FILE.parent / "predictions_workshop.csv.gz"


def load_cached_predictions(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Walk-forward predictions saved by workshop3/build_cache.py, plus the `ensemble` column (forest, boosting, mlp)."""
    df = pd.read_csv(path or CACHE_FILE, parse_dates=["date"]).set_index(["date", "symbol"]).sort_index()
    df["ensemble"] = df[list(ENSEMBLE_MODELS)].mean(axis=1)
    return df


def market_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualised volatility of the equal-weight average of all stocks, over the last `window` days, known at t."""
    mkt = prices.pct_change(fill_method=None).mean(axis=1)
    return mkt.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
