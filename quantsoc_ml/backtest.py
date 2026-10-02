"""Daily backtest: forecasts -> weights -> daily portfolio returns, with costs.

Timing (the simplification the workshop states out loud):

* The forecast dated t uses features known at the close of t. On a rebalance day the new
  weights are formed from it and traded AT THE CLOSE OF t. A real system would fill at the
  next open; trading at the close we computed on is the simplification.
* Those weights earn the return of day t + 1, never of day t:
      portfolio_return[t] = sum_i weight_i[t-1] * daily_return_i[t]
  where weight[t-1] is the position held at the close of t-1.
* Between rebalances the positions are not touched, so weights drift with prices:
      weight_i[t] = weight_i[t-1] * (1 + r_i[t]) / (1 + portfolio_return[t]).
  Uninvested money (and the proceeds of short sales) earns 0.
* Cost on a rebalance day = cost_bps / 10,000 * sum_i |new_weight_i - drifted_weight_i|,
  taken out of equity at that close: equity *= (1 - cost).
* Rebalances are scheduled on the first date of the forecast panel and then every
  `rebalance_days` trading days; a scheduled date with fewer than n_long + n_short finite
  forecasts is skipped (the book is held), so the first actual trade is the first scheduled
  date that has enough forecasts. The equal-weight benchmark goes through the same engine
  with the same cost.
* If equity reaches 0 or below at a close (a short that doubles with all the capital in it),
  the book is blown up: that day's return is recorded as -100%, equity stays at 0, nothing
  more is traded, later daily returns are NaN and stats carry blown_up = True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .portfolio import StrategyConfig, weights_from_forecasts

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series          # value of 1 unit of starting capital at each close (after costs)
    daily_returns: pd.Series   # net return of each day, after costs
    turnover: pd.Series        # sum |weight change| traded at each close (0 on non-rebalance days)
    weights: pd.DataFrame      # weights held after each close (drifted between rebalances)
    stats: Dict[str, float]
    costs: pd.Series = field(default=None)  # fraction of equity paid in costs at each close


def _simulate(targets: pd.DataFrame, returns: pd.DataFrame, cost_bps: float) -> BacktestResult:
    """Core engine. `targets` has a row per date: target weights on rebalance days, all NaN otherwise."""
    dates = targets.index
    R = returns.reindex(index=dates, columns=targets.columns).to_numpy(dtype=float)
    R = np.nan_to_num(R, nan=0.0)          # a missing return is treated as no move (stated)
    T = targets.to_numpy(dtype=float)
    n_days, n_sym = T.shape
    rate = float(cost_bps) / 1e4
    w = np.zeros(n_sym)                    # weights held coming into the day
    held = np.zeros((n_days, n_sym))
    gross = np.zeros(n_days)
    cost = np.zeros(n_days)
    turn = np.zeros(n_days)
    blown_at = None
    for i in range(n_days):
        if i > 0:                          # earn today's return on yesterday's close weights
            g = float(w @ R[i])
            gross[i] = g
            if 1.0 + g <= 0.0:             # equity gone: no drift, no trade, nothing after
                blown_at = i
                break
            w = w * (1.0 + R[i]) / (1.0 + g)
        if not np.isnan(T[i]).all():       # rebalance at today's close
            new = np.nan_to_num(T[i])
            turn[i] = float(np.abs(new - w).sum())
            cost[i] = rate * turn[i]
            if cost[i] >= 1.0:
                blown_at = i
                break
            w = new
        held[i] = w
    net = (1.0 + gross) * (1.0 - cost) - 1.0
    if blown_at is not None:
        net[blown_at] = -1.0
        net[blown_at + 1:] = np.nan
        held[blown_at:] = 0.0
        turn[blown_at:] = 0.0
        cost[blown_at + 1:] = 0.0
    equity = pd.Series(np.cumprod(1.0 + np.nan_to_num(net)), index=dates, name="equity")
    daily = pd.Series(net, index=dates, name="daily_return")
    turnover = pd.Series(turn, index=dates, name="turnover")
    rebalanced = ~np.isnan(T).all(axis=1)
    if blown_at is not None:
        rebalanced[blown_at:] = False      # nothing is traded from the blow-up close on
    stats = compute_stats(equity, daily, turnover[rebalanced], blown_up=blown_at is not None)
    return BacktestResult(equity=equity, daily_returns=daily, turnover=turnover,
                          weights=pd.DataFrame(held, index=dates, columns=targets.columns), stats=stats,
                          costs=pd.Series(cost, index=dates, name="cost"))


def compute_stats(equity: pd.Series, daily: pd.Series, rebalance_turnover: pd.Series,
                  blown_up: bool = False) -> Dict[str, float]:
    """Headline numbers; years are 252 trading days.

    sharpe is the simple arithmetic form mean(daily) / std(daily) * sqrt(252) with a 0%
    risk-free rate. A blown-up book reports total_return -100% and NaN for sharpe, annual_vol
    and max_drawdown, because statistics of a path that ended at zero would be misleading.
    """
    n = len(daily)
    if blown_up:
        return {"total_return": -1.0, "annual_return": -1.0, "annual_vol": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "avg_turnover": float(rebalance_turnover.mean()) if len(rebalance_turnover) else 0.0,
                "n_rebalances": int(len(rebalance_turnover)), "blown_up": True}
    total = float(equity.iloc[-1] - 1.0)
    vol = float(daily.std(ddof=1)) if n > 1 else float("nan")
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])
    dd = np.r_[1.0, equity.to_numpy()] / peak - 1.0
    return {
        "total_return": total,
        "annual_return": float((1.0 + total) ** (TRADING_DAYS / n) - 1.0) if n else float("nan"),
        "annual_vol": vol * np.sqrt(TRADING_DAYS),
        "sharpe": float(daily.mean() / vol * np.sqrt(TRADING_DAYS)) if vol and vol > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "avg_turnover": float(rebalance_turnover.mean()) if len(rebalance_turnover) else 0.0,
        "n_rebalances": int(len(rebalance_turnover)),
        "blown_up": False,
    }


def _rebalance_mask(n: int, every: int) -> np.ndarray:
    if int(every) < 1:
        raise ValueError("rebalance_days must be at least 1")
    return (np.arange(n) % int(every)) == 0


def run_backtest(pred_panel: pd.DataFrame, returns_wide: pd.DataFrame,
                 config: Optional[StrategyConfig] = None) -> BacktestResult:
    """Backtest a date x symbol forecast panel against DAILY returns (see the module docstring).

    The run covers the dates of `pred_panel`. On a rebalance day with too few finite forecasts
    the book is simply held (no trade, not counted as a rebalance).
    """
    config = config or StrategyConfig()
    missing = sorted(map(str, set(pred_panel.columns) - set(returns_wide.columns)))
    if missing:
        raise ValueError(f"these symbols have forecasts but no daily returns, so they cannot be traded: {', '.join(missing)}")
    pred_panel = pred_panel.sort_index()
    pred_panel = pred_panel.reindex(columns=sorted(set(pred_panel.columns) | set(returns_wide.columns)))
    mask = _rebalance_mask(len(pred_panel), config.rebalance_days)
    targets = pd.DataFrame(np.nan, index=pred_panel.index, columns=pred_panel.columns)
    need = int(config.n_long) + int(config.n_short)
    for date in pred_panel.index[mask]:
        row = pred_panel.loc[date]
        if np.isfinite(row.to_numpy(dtype=float)).sum() >= need:
            targets.loc[date] = weights_from_forecasts(row, config.n_long, config.n_short).to_numpy()
    return _simulate(targets, returns_wide, config.cost_bps)


def run_benchmark(returns_wide: pd.DataFrame, dates, config: Optional[StrategyConfig] = None) -> BacktestResult:
    """Equal-weight long-only benchmark (1/N in every symbol), same rebalancing and same cost."""
    config = config or StrategyConfig()
    dates = pd.DatetimeIndex(dates).sort_values()
    cols = list(returns_wide.columns)
    mask = _rebalance_mask(len(dates), config.rebalance_days)
    targets = pd.DataFrame(np.nan, index=dates, columns=cols)
    targets.loc[dates[mask]] = 1.0 / len(cols)
    return _simulate(targets, returns_wide, config.cost_bps)


def benchmark_equal_weight(returns_wide: pd.DataFrame, dates, config: Optional[StrategyConfig] = None) -> pd.Series:
    """Equity curve of the equal-weight benchmark (use run_benchmark for its stats too)."""
    return run_benchmark(returns_wide, dates, config).equity.rename("benchmark")


def stats_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Side-by-side stats, one column per named result."""
    return pd.DataFrame({k: v.stats for k, v in results.items()})
