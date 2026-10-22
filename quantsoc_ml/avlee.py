"""Workshop 4: a replication of Avellaneda & Lee (2010), "Statistical arbitrage in the US equities market",
Quantitative Finance 10(7), 761-782. Free working paper: https://math.nyu.edu/~avellane/AvellanedaLeeStatArb071008.pdf

The paper's recipe (section numbers refer to the 2008 working paper):

1. Factors (sec. 2). Either PCA: the correlation matrix of standardised daily returns over the last
   252 days, its top eigenvectors v(j), and "eigenportfolios" holding v(j)_i / sigma_i dollars of
   stock i (15 of them, or as many as explain 55% of the variance). Or ETFs: each stock against
   its sector SPDR fund.
2. Residuals (appendix). Regress each stock's last 60 daily returns on the factor returns, with an
   intercept. Sum the residuals into X_1..X_60 (so X_60 = 0 by construction).
3. Ornstein-Uhlenbeck fit (appendix). Fit X_{n+1} = a + b X_n + zeta. Then kappa = -ln(b) * 252,
   m = a / (1 - b), sigma_eq = sqrt(Var(zeta) / (1 - b^2)). Only stocks with kappa > 252/30
   (mean reversion within half the window) are traded.
4. s-score (eq. 22): s = (X_60 - m) / sigma_eq = -m / sigma_eq, with m centred by subtracting the
   average m across the stocks whose model is accepted that day (see s_scores for why).
5. Trading (eq. 16): open long if s < -1.25, open short if s > +1.25, close a short if s < +0.75,
   close a long if s > -0.50. Each position is long (short) the stock and short (long) its factor
   hedge, sized as a fixed fraction of equity. 5 bps per trade.

Choices this replication makes where the paper is silent or our setting differs, stated:
* Universe: today's S&P 100 (about 100 stocks) instead of every US stock above $1bn (about 1,000).
  With 100 stocks the default is the 55%-variance rule; 15 factors out of 100 would leave little residual.
* A held stock whose s-score cannot be computed (no valid fit) is closed. A stock that fails the
  kappa filter cannot open a new trade but an existing trade still closes by its s-score.
* Positions are held at a constant fraction of equity while open; the hedge is recomputed every
  day (rehedge="entry" keeps each trade's opening hedge instead, an improvement to try). Costs are charged on every dollar the combined (netted) book changes from one day to the
  next, hedges included, which is slightly stricter than the paper's formula.
* No interest on cash: the book is market neutral, so the Sharpe ratio barely changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_FILE = ROOT / "workshop3" / "data" / "sp100_daily.csv.gz"
ETF_FILE = ROOT / "workshop4" / "data" / "etf_daily.csv.gz"
SECTOR_FILE = ROOT / "workshop4" / "data" / "sectors.csv"
TRADING_DAYS = 252


@dataclass
class Config:
    """Every knob of the strategy, with the paper's values as defaults."""
    factors: str = "pca"            # "pca" or "etf"
    var_explained: float = 0.55     # PCA: keep factors until they explain this share of variance
    n_factors: Optional[int] = None  # PCA: a fixed number instead (the paper's other choice is 15)
    corr_window: int = 252          # PCA: days of returns for the correlation matrix
    window: int = 60                # days for the factor regression and the OU fit
    min_kappa: float = 252 / 30     # mean reversion faster than half of the 60-day window
    s_open: float = 1.25            # open when |s| > 1.25
    s_close_short: float = 0.75     # close a short when s < 0.75
    s_close_long: float = 0.50      # close a long when s > -0.50
    centre_means: bool = True       # eq. 22: subtract the average m across stocks
    position: float = 0.05          # each open trade is 5% of equity in the stock (plus its hedge)
    rehedge: str = "daily"          # "daily" (this replication's default) or "entry": keep each trade's opening hedge
    cost_bps: float = 5.0           # per dollar traded


# --------------------------------------------------------------------------- data

def load_prices(path=None) -> pd.DataFrame:
    """Stocks: dates down, symbols across (dividend- and split-adjusted closes)."""
    df = pd.read_csv(path or STOCK_FILE, parse_dates=["date"])
    return df.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def load_volume(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or STOCK_FILE, parse_dates=["date"])
    return df.pivot(index="date", columns="symbol", values="volume").sort_index()


def load_etfs(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or ETF_FILE, parse_dates=["date"])
    return df.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def load_sectors(path=None) -> pd.Series:
    """symbol -> sector SPDR ticker."""
    s = pd.read_csv(path or SECTOR_FILE)
    return s.set_index("symbol")["etf"]


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


# --------------------------------------------------------------------------- factors

def pca_eigenportfolios(R: np.ndarray, var_explained: Optional[float] = 0.55,
                        n_factors: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Eigenportfolios from a (days x stocks) return window with no missing values.

    Returns (Q, eigenvalues, m): Q is (stocks x m), Q[i, j] = v(j)_i / sigma_i, the dollars held in
    stock i per unit of factor j, so factor returns are R @ Q. Eigenvalues are of the correlation
    matrix, largest first, and sum to the number of stocks.
    """
    sd = R.std(axis=0, ddof=1)
    Y = (R - R.mean(axis=0)) / sd
    rho = Y.T @ Y / (len(R) - 1)
    vals, vecs = np.linalg.eigh(rho)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    if n_factors is not None:
        m = int(n_factors)
    else:
        m = int(np.searchsorted(np.cumsum(vals) / vals.sum(), var_explained) + 1)
    vecs = vecs * np.sign(vecs.sum(axis=0, keepdims=True) + 1e-12)   # sign convention: mostly-positive weights
    return vecs[:, :m] / sd[:, None], vals, m


# --------------------------------------------------------------------------- residuals and the OU fit

@dataclass
class Fit:
    """One day's estimates for every stock (arrays aligned with `symbols`)."""
    symbols: List[str]
    s: np.ndarray            # s-score (NaN where no valid fit)
    kappa: np.ndarray        # speed of mean reversion, per year
    m: np.ndarray            # equilibrium level of the cumulative residual (before centring)
    sigma_eq: np.ndarray     # equilibrium standard deviation of the residual
    beta: np.ndarray         # (stocks x factors) loadings
    alpha: np.ndarray        # intercept of the factor regression, per day
    r2: np.ndarray           # how much of the stock's 60-day return variance the factors explain
    eligible: np.ndarray     # kappa filter passed (may open new trades)
    n_factors: int = 0


def ou_fit(Rs: np.ndarray, F: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Regress each stock (columns of Rs, window x stocks) on factor returns F (window x k) with an
    intercept, then fit the AR(1) to the cumulative residuals. Vectorised over stocks."""
    n = len(Rs)
    X = np.column_stack([np.ones(n), F])
    B, *_ = np.linalg.lstsq(X, Rs, rcond=None)
    resid = Rs - X @ B
    ss_tot = ((Rs - Rs.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - (resid ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, np.nan)
    return {"beta": B[1:].T, "alpha": B[0], "r2": r2, **_ar1(resid)}       # X_60 = 0 up to rounding


def ou_fit_single(Rs: np.ndarray, Fs: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Like ou_fit, but stock i is regressed on its OWN factor column Fs[:, i] (one ETF per stock)."""
    fm, rm = Fs.mean(axis=0), Rs.mean(axis=0)
    beta = ((Fs - fm) * (Rs - rm)).sum(axis=0) / ((Fs - fm) ** 2).sum(axis=0)
    alpha = rm - beta * fm
    resid = Rs - (alpha + beta * Fs)
    ss_tot = ((Rs - rm) ** 2).sum(axis=0)
    r2 = 1 - (resid ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, np.nan)
    out = _ar1(resid)
    return {"beta": beta[:, None], "alpha": alpha, "r2": r2, **out}


def _ar1(resid: np.ndarray) -> Dict[str, np.ndarray]:
    """AR(1) fit X_{n+1} = a + b X_n + zeta on the cumulative residuals, per column."""
    Xc = np.cumsum(resid, axis=0)
    x, y = Xc[:-1], Xc[1:]
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    b = ((x - xm) * (y - ym)).sum(axis=0) / ((x - xm) ** 2).sum(axis=0)
    a = ym - b * xm
    var_z = (y - (a + b * x)).var(axis=0, ddof=1)
    ok = (b > 0) & (b < 1) & np.isfinite(b) & (var_z > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(ok, -np.log(np.where(ok, b, 0.5)) * TRADING_DAYS, np.nan)
        m = np.where(ok, a / (1 - b), np.nan)
        sigma_eq = np.where(ok, np.sqrt(var_z / (1 - b ** 2)), np.nan)
    return {"kappa": kappa, "m": m, "sigma_eq": sigma_eq, "x_last": Xc[-1], "a": a, "b": b}


def s_scores(fit: Dict[str, np.ndarray], cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """(s, eligible). s = (X_60 - m_centred) / sigma_eq, centred by the average m over the stocks
    that pass the kappa filter.

    Eq. 18 and 22 average "over different stocks". The appendix says that when b is too close to 1
    "the model is rejected for the stock under consideration", so this replication averages over the
    stocks where the model is accepted. It matters: m = a / (1 - b) explodes as b nears 1, and one
    rejected stock (QCOM on 31 January 2018, m about 490 typical sigmas) would otherwise shift every
    other s-score by several sigmas and open trades on one side only."""
    eligible = np.isfinite(fit["kappa"]) & (fit["kappa"] > cfg.min_kappa)
    m = fit["m"].copy()
    if cfg.centre_means and eligible.any():
        m = m - np.nanmean(m[eligible])
    s = (fit["x_last"] - m) / fit["sigma_eq"]
    return s, eligible


def estimate_day(R: pd.DataFrame, t: int, cfg: Config, etf_R: Optional[pd.DataFrame] = None,
                 sectors: Optional[pd.Series] = None) -> Optional[Fit]:
    """All estimates using returns up to and including row t of R (known at that day's close)."""
    long_w = cfg.corr_window if cfg.factors == "pca" else cfg.window
    if t + 1 < long_w:
        return None
    block = R.iloc[t + 1 - long_w: t + 1]
    cols = block.columns[block.notna().all(axis=0)]
    if len(cols) < 10:
        return None
    if cfg.factors == "pca":
        Q, vals, k = pca_eigenportfolios(block[cols].to_numpy(), cfg.var_explained, cfg.n_factors)
        Rs = block[cols].to_numpy()[-cfg.window:]
        F = Rs @ Q
        f = ou_fit(Rs, F, cfg)
        beta = f["beta"]                                  # stocks x k
    else:
        win = block.iloc[-cfg.window:]
        etf_win = etf_R.loc[win.index]
        usable = set(etf_win.columns[etf_win.notna().all(axis=0)])
        # a sector fund that was not trading yet (XLRE before 2015, XLC before 2018): use the market
        hedge = np.array([e if e in usable else "SPY" for e in sectors.reindex(cols).fillna("SPY")])
        f = ou_fit_single(win[cols].to_numpy(), etf_win[hedge].to_numpy(), cfg)
        beta = f["beta"]
        k = 1
        f["hedge_etf"] = hedge
    s, eligible = s_scores(f, cfg)
    fit = Fit(list(cols), s, f["kappa"], f["m"], f["sigma_eq"], beta, f["alpha"], f["r2"], eligible, k)
    if cfg.factors == "pca":
        fit.Q = Q                                         # stocks x k, for the hedge
        fit.eigenvalues = vals
    else:
        fit.hedge_etf = f["hedge_etf"]
    return fit


# --------------------------------------------------------------------------- signals

def next_state(state: int, s: float, eligible: bool, cfg: Config) -> int:
    """Eq. 16. state: +1 long, -1 short, 0 flat."""
    if not np.isfinite(s):
        return 0                                          # no valid fit: close whatever is open
    if state == 0:
        if not eligible:
            return 0
        if s < -cfg.s_open:
            return 1
        if s > cfg.s_open:
            return -1
        return 0
    if state == 1 and s > -cfg.s_close_long:
        return 0
    if state == -1 and s < cfg.s_close_short:
        return 0
    return state


def book_weights(fit: Fit, states: Dict[str, int], cfg: Config) -> pd.Series:
    """Dollar weights (fraction of equity) of the whole book: each trade's stock leg plus its hedge.

    PCA: the hedge for a long in stock i is short beta_ij units of eigenportfolio j, i.e. short
    beta_ij * Q[:, j] dollars of every stock, so the book is a vector over stocks only.
    ETF: the hedge is short beta_i dollars of the stock's sector fund.
    """
    idx = {s: i for i, s in enumerate(fit.symbols)}
    pos = np.zeros(len(fit.symbols))
    for sym, st in states.items():
        if st and sym in idx:
            pos[idx[sym]] = st * cfg.position
    w = pd.Series(pos, index=fit.symbols)
    if not pos.any():
        return w
    if cfg.factors == "pca":
        hedge = -(fit.Q @ (fit.beta.T @ pos))             # stocks: sum_i pos_i * (-sum_j beta_ij Q_.j)
        return w + pd.Series(hedge, index=fit.symbols)
    etf = pd.Series(-pos * fit.beta[:, 0], index=fit.hedge_etf).groupby(level=0).sum()
    return pd.concat([w, etf]).groupby(level=0).sum()


# --------------------------------------------------------------------------- backtest

@dataclass
class Backtest:
    gross: pd.Series                 # daily return before costs
    net: pd.Series                   # after costs
    turnover: pd.Series              # dollars traded per dollar of equity
    n_long: pd.Series
    n_short: pd.Series
    n_factors: pd.Series
    trades: pd.DataFrame             # one row per closed trade
    weights: Dict[pd.Timestamp, pd.Series] = field(repr=False, default_factory=dict)
    cfg: Config = field(default_factory=Config)

    def window(self, start=None, end=None) -> "Backtest":
        cut = lambda s: s.loc[start:end]
        tr = self.trades
        if len(tr):
            keep = pd.Series(True, index=tr.index)
            if start is not None:
                keep &= tr["entry"] >= pd.Timestamp(start)
            if end is not None:
                keep &= tr["exit"] <= pd.Timestamp(end)
            tr = tr[keep]
        return Backtest(cut(self.gross), cut(self.net), cut(self.turnover), cut(self.n_long), cut(self.n_short),
                        cut(self.n_factors), tr, {}, self.cfg)

    def stats(self) -> Dict[str, float]:
        from .statarb import perf_stats
        g, n = perf_stats(self.gross), perf_stats(self.net)
        return {"Sharpe (before costs)": g["Sharpe"], "annual return (before costs)": g["annual return"],
                "Sharpe": n["Sharpe"], "annual return": n["annual return"], "annual vol": n["annual vol"],
                "max drawdown": n["max drawdown"], "daily turnover": float(self.turnover.mean()),
                "open trades (avg)": float((self.n_long + self.n_short).mean()),
                "trades": int(len(self.trades)),
                "avg holding days": float(self.trades["days"].mean()) if len(self.trades) else float("nan"),
                "days": int(len(self.net))}


def run(prices: pd.DataFrame, cfg: Optional[Config] = None, etf_prices: Optional[pd.DataFrame] = None,
        sectors: Optional[pd.Series] = None, start=None, end=None, keep_weights: bool = False,
        volume: Optional[pd.DataFrame] = None, progress=None) -> Backtest:
    """Daily walk through history: estimate at the close of day t, trade at that close, earn day t+1.

    Every estimate uses returns up to day t only. `volume` switches on the paper's "trading time"
    returns (sec. 6) for the estimation, so the hedge ratio also comes from volume-scaled stock
    returns (the paper does not say either way); P&L always uses ordinary returns. Booked returns
    fall on dates from `start` to `end` inclusive.
    """
    cfg = cfg or Config()
    R = returns(prices)
    R_signal = trading_time_returns(R, volume) if volume is not None else R
    etf_R = returns(etf_prices) if etf_prices is not None else None
    allR = R if etf_R is None else R.join(etf_R, how="left")
    dates = R.index
    t0 = dates.searchsorted(pd.Timestamp(start)) if start is not None else 0
    t1 = dates.searchsorted(pd.Timestamp(end), side="right") if end is not None else len(dates)
    states: Dict[str, int] = {}
    open_at: Dict[str, Tuple[pd.Timestamp, int, float]] = {}
    prev_w = pd.Series(dtype=float)
    out_g, out_n, out_t, nl, ns, nf, trades, saved = {}, {}, {}, {}, {}, {}, [], {}
    resid_pnl: Dict[str, float] = {}
    entry_info: Dict[str, dict] = {}
    frozen: Dict[str, pd.Series] = {}
    # estimate at the close of day t, earn day t+1: so t runs from the day before `start` to the day
    # before `end`, and every booked return falls inside [start, end]
    for t in range(max(t0 - 1, 1), min(t1, len(dates)) - 1):
        d, nxt = dates[t], dates[t + 1]
        fit = estimate_day(R_signal, t, cfg, etf_R, sectors)
        if fit is None:
            continue
        s_map = dict(zip(fit.symbols, fit.s))
        el_map = dict(zip(fit.symbols, fit.eligible))
        new_states = {}
        for sym in sorted(set(states) | set(fit.symbols)):
            st = next_state(states.get(sym, 0), s_map.get(sym, np.nan), bool(el_map.get(sym, False)), cfg)
            if st:
                new_states[sym] = st
            prev = states.get(sym, 0)
            if prev and st != prev:                       # a trade closed today
                e_date, side, _ = open_at.pop(sym)
                trades.append({"symbol": sym, "side": side, "entry": e_date, "exit": d,
                               "days": int(dates.get_loc(d) - dates.get_loc(e_date)),
                               "pnl": resid_pnl.pop(sym, 0.0), **entry_info.pop(sym, {})})
                frozen.pop(sym, None)
            if st and st != prev:
                open_at[sym] = (d, st, s_map.get(sym, np.nan))
                resid_pnl[sym] = 0.0
                i = fit.symbols.index(sym)
                entry_info[sym] = {"s_entry": float(fit.s[i]), "kappa": float(fit.kappa[i]),
                                   "sigma_eq": float(fit.sigma_eq[i]), "r2": float(fit.r2[i]),
                                   "alpha": float(fit.alpha[i]), "n_factors": int(fit.n_factors)}
                if cfg.rehedge == "entry":
                    frozen[sym] = book_weights(fit, {sym: st}, cfg)
        states = new_states
        if cfg.rehedge == "entry":
            w = (pd.concat(list(frozen.values()), axis=1).fillna(0.0).sum(axis=1) if frozen
                 else pd.Series(dtype=float))
        else:
            w = book_weights(fit, states, cfg)
        r_next = allR.loc[nxt].reindex(w.index).fillna(0.0)
        aligned = pd.concat([w, prev_w], axis=1).fillna(0.0)
        turnover = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum())
        g = float((w * r_next).sum())
        out_g[nxt], out_t[nxt] = g, turnover
        out_n[nxt] = g - turnover * cfg.cost_bps / 1e4
        nl[nxt] = sum(1 for v in states.values() if v > 0)
        ns[nxt] = sum(1 for v in states.values() if v < 0)
        nf[nxt] = fit.n_factors
        # each open trade's own hedged return tomorrow: stock return minus beta times its factors' return
        if states:
            r_nxt = allR.loc[nxt]
            h = hedged_returns(fit, r_nxt)
            for sym, st in states.items():
                if cfg.rehedge == "entry" and sym in frozen:        # the hedge actually held
                    leg = frozen[sym]
                    resid_pnl[sym] = resid_pnl.get(sym, 0.0) + float((leg * r_nxt.reindex(leg.index).fillna(0.0)).sum()) / cfg.position
                else:
                    resid_pnl[sym] = resid_pnl.get(sym, 0.0) + st * float(h.get(sym, 0.0))
        if keep_weights:
            saved[d] = w
        prev_w = w
        if progress and t % 250 == 0:
            progress(f"{d.date()}  open trades: {len(states)}")
    S = lambda d_: pd.Series(d_, dtype=float).sort_index()
    return Backtest(S(out_g), S(out_n), S(out_t), S(nl), S(ns), S(nf), pd.DataFrame(trades), saved, cfg)


def hedged_returns(fit: Fit, r: pd.Series) -> pd.Series:
    """One day's return of $1 long each stock hedged with its factors (what one long trade earns)."""
    rs = r.reindex(fit.symbols).fillna(0.0).to_numpy()
    if hasattr(fit, "Q"):
        f = rs @ fit.Q
        return pd.Series(rs - fit.beta @ f, index=fit.symbols)
    re = r.reindex(fit.hedge_etf).fillna(0.0).to_numpy()
    return pd.Series(rs - fit.beta[:, 0] * re, index=fit.symbols)


def trading_time_returns(R: pd.DataFrame, volume: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Sec. 6: R * <V> / V, where <V> is the average daily volume over the previous `window` days.

    Low-volume moves are scaled up, high-volume moves scaled down. The paper does not state the
    averaging window; 10 days is this replication's choice.
    """
    avg = volume.rolling(window).mean().shift(1)
    return R * (avg / volume.replace(0, np.nan))


# --------------------------------------------------------------------------- live use

def live_signal(prices: pd.DataFrame, held: Dict[str, int], cfg: Optional[Config] = None,
                etf_prices: Optional[pd.DataFrame] = None, sectors: Optional[pd.Series] = None
                ) -> Tuple[Fit, Dict[str, int], pd.Series]:
    """For the dashboard: today's fit, the new trade states, and the target book weights.

    `held` maps stock symbol -> +1 (long trade open) or -1 (short trade open), read from the account.
    Uses the last row of `prices` as today's close.
    """
    cfg = cfg or Config()
    R = returns(prices)
    etf_R = returns(etf_prices) if etf_prices is not None else None
    fit = estimate_day(R, len(R) - 1, cfg, etf_R, sectors)
    if fit is None:
        raise ValueError("not enough history to estimate the model")
    s_map, el_map = dict(zip(fit.symbols, fit.s)), dict(zip(fit.symbols, fit.eligible))
    states = {}
    for sym in sorted(set(held) | set(fit.symbols)):
        st = next_state(int(held.get(sym, 0)), s_map.get(sym, np.nan), bool(el_map.get(sym, False)), cfg)
        if st:
            states[sym] = st
    return fit, states, book_weights(fit, states, cfg)


# --------------------------------------------------------------------------- cached backtests

DAILY_CACHE = ROOT / "workshop4" / "data" / "backtests_daily.csv.gz"
TRADES_CACHE = ROOT / "workshop4" / "data" / "backtests_trades.csv.gz"


def load_cached(variant: str) -> Backtest:
    """A Backtest rebuilt from workshop4/build_cache.py's files (no weights kept)."""
    d = pd.read_csv(DAILY_CACHE, parse_dates=["date"])
    d = d[d["variant"] == variant].set_index("date").sort_index()
    if d.empty:
        raise ValueError(f"no cached variant {variant!r}")
    t = pd.read_csv(TRADES_CACHE, parse_dates=["entry", "exit"])
    t = t[t["variant"] == variant].drop(columns="variant").reset_index(drop=True)
    return Backtest(d["gross"], d["net"], d["turnover"], d["n_long"], d["n_short"], d["n_factors"], t)
