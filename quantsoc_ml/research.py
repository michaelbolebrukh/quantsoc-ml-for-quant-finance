"""Workshop 3 research tools: purged k-fold cross-validation and the deflated Sharpe ratio.

* Purged k-fold with embargo: Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 7.
  Folds are blocks of consecutive DATES (never single rows), so one day's stocks never sit on both
  sides. Training dates whose label window overlaps the test block are removed (purge), and so are
  the `embargo` dates right after it, because features built from overlapping windows carry
  information across the boundary.
* Deflated Sharpe ratio: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5).
  The probability that the true Sharpe ratio is above zero, after allowing for having picked the
  best of N tried strategies and for skewed, fat-tailed returns.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- cross-validation

def purged_kfold(dates: Sequence, n_splits: int = 5, label_horizon: int = 1,
                 embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_mask, test_mask) boolean arrays over `dates` (one entry per ROW, e.g. the panel's date level).

    The unique dates are cut into `n_splits` consecutive blocks. For each test block:
      * purge: drop training dates within `label_horizon` days BEFORE the block, whose labels
        look into the block's prices;
      * embargo: drop `embargo` training dates right AFTER the block.
    With label_horizon=0 and embargo=0 this is plain k-fold over date blocks (no shuffling).
    """
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    uniq = d.unique().sort_values()
    pos = uniq.get_indexer(d)                       # each row's position in the date calendar
    blocks = np.array_split(np.arange(len(uniq)), n_splits)
    for b in blocks:
        lo, hi = b[0], b[-1]
        test = (pos >= lo) & (pos <= hi)
        banned = (pos >= lo - label_horizon) & (pos <= hi + embargo)
        yield ~banned, test


def shuffled_kfold(n_rows: int, n_splits: int = 5, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """The textbook k-fold: rows shuffled, so neighbouring days land on both sides. For comparison only."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(n_rows) % n_splits)
    for f in range(n_splits):
        yield fold != f, fold == f


def cv_scores(model_factory, X: np.ndarray, y: np.ndarray, splits, scorer) -> np.ndarray:
    """Fit a fresh model per split and score it on the test part."""
    out = []
    for train, test in splits:
        m = model_factory()
        m.fit(X[train], y[train])
        out.append(scorer(y[test], m.predict_proba(X[test])[:, 1]))
    return np.array(out)


def forward_median_target(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """1 if the stock's return over the next `horizon` days beats the cross-sectional median, else 0.

    With horizon > 1, the labels of neighbouring days share most of their window: they OVERLAP.
    That is the case where shuffled k-fold leaks and purging matters.
    """
    r = prices.shift(-horizon) / prices - 1.0
    y = r.gt(r.median(axis=1), axis=0).astype(float)
    return y.where(r.notna())


# --------------------------------------------------------------------------- Sharpe ratios

def sharpe(daily: pd.Series) -> float:
    """Per-period (daily, not annualised) Sharpe ratio: mean / standard deviation."""
    daily = pd.Series(daily).dropna()
    return float(daily.mean() / daily.std(ddof=1))


def probabilistic_sharpe(daily: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), allowing for skewness and fat tails (Bailey & LdP 2012)."""
    r = pd.Series(daily).dropna()
    n = len(r)
    sr = sharpe(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))        # 3 for a normal distribution
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sharpe_var: float) -> float:
    """Expected best per-period Sharpe among `n_trials` strategies whose TRUE Sharpe is zero.

    sharpe_var is the variance of the Sharpe ratios across the trials you ran. Bailey & LdP (2014), eq. 2.
    """
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(daily: pd.Series, trial_sharpes: Sequence[float], n_trials: Optional[int] = None) -> dict:
    """Deflated Sharpe ratio of the strategy you picked, given the Sharpe ratios of everything you tried.

    trial_sharpes: per-period Sharpe ratios of every variant tested (the picked one included).
    n_trials: how many independent tries that represents; defaults to len(trial_sharpes).
    Returns the observed Sharpe, the hurdle (expected best of n zero-skill tries) and the DSR, a
    probability: 0.95 or more is the usual bar.
    """
    tr = np.asarray(trial_sharpes, dtype=float)
    n = int(n_trials or len(tr))
    hurdle = expected_max_sharpe(n, float(np.var(tr, ddof=1)) if len(tr) > 1 else 0.0)
    return {"sharpe_daily": sharpe(daily), "sharpe_annual": sharpe(daily) * np.sqrt(252),
            "trials": n, "hurdle_daily": hurdle, "hurdle_annual": hurdle * np.sqrt(252),
            "PSR_vs_zero": probabilistic_sharpe(daily, 0.0), "DSR": probabilistic_sharpe(daily, hurdle)}
