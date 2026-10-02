"""Reference answers for the five copy-and-change tasks, and the reference strategy.

They are shown on request (`ex.show_solution(n)`) and used by `ex.use_reference(n)` or, when a
task has not been passed, by `ex.result(n)`. Either way the tracker prints a [ref] line and the
final knob table and the export manifest record that a knob came from here. Nothing is ever
swapped in silently.

REFERENCE_STRATEGY is the design fixed in the build contract (decision D7): rank-normalised
features, the 2015 to 2019 test period, a 200-tree depth-6 forest, 4 long and 4 short, a
21-day (monthly) rebalance and 5 basis points of cost.
"""
from __future__ import annotations

from dataclasses import replace

from .features import past_return, rolling_vol
from .models import fit_forest
from .portfolio import StrategyConfig, weights_from_forecasts
from .splits import time_split

REFERENCE_STRATEGY = StrategyConfig(momentum_lookback=21, extra_feature="vol_63", train_end="2014-12-31",
                                    test_start="2015-01-12", n_estimators=200, max_depth=6, n_long=4, n_short=4,
                                    rebalance_days=21, cost_bps=5.0)

# Which StrategyConfig fields each task sets.
TASK_KNOBS = {
    "1": ("momentum_lookback",),
    "2": ("extra_feature",),
    "3": ("train_end", "test_start"),
    "4": ("n_estimators", "max_depth"),
    "5": ("n_long", "n_short", "rebalance_days"),
}


def reference_strategy() -> StrategyConfig:
    """A fresh copy, so changing one notebook's strategy never changes the reference."""
    return replace(REFERENCE_STRATEGY)


# --- Task 1: momentum lookback --------------------------------------------------
def task1(prices):
    lookback = REFERENCE_STRATEGY.momentum_lookback
    return past_return(prices, lookback)


# --- Task 2: the extra feature --------------------------------------------------
def task2(returns):
    window = 63
    return rolling_vol(returns, window)


# --- Task 3: the split dates ----------------------------------------------------
def task3(ds):
    train_end, test_start = REFERENCE_STRATEGY.train_end, REFERENCE_STRATEGY.test_start
    return time_split(ds, train_end, test_start)


# --- Task 4: the forest ---------------------------------------------------------
def task4(X_quarter, y_quarter):
    n_estimators, max_depth = REFERENCE_STRATEGY.n_estimators, REFERENCE_STRATEGY.max_depth
    return fit_forest(X_quarter, y_quarter, n_estimators=n_estimators, max_depth=max_depth)


# --- Task 5: the portfolio ------------------------------------------------------
def task5(pred_today):
    n_long, n_short = REFERENCE_STRATEGY.n_long, REFERENCE_STRATEGY.n_short
    return weights_from_forecasts(pred_today, n_long=n_long, n_short=n_short)


SOLUTION_SOURCE = {
    "1": 'lookback = 21                                  # about one month of trading days\n'
         'my_momentum = past_return(prices, lookback)\n'
         'ex.check(1, my_momentum, lookback)',
    "2": 'window = 63                                    # about three months of trading days\n'
         'my_extra = rolling_vol(returns, window)\n'
         'ex.check(2, my_extra, window)',
    "3": 'train_end, test_start = "2014-12-31", "2015-01-12"\n'
         'train, test = time_split(ds, train_end, test_start)\n'
         'ex.check(3, (train, test), train_end, test_start)',
    "4": 'n_estimators, max_depth = 200, 6\n'
         'my_forest = make_forest(X_quarter, y_quarter, n_estimators, max_depth)\n'
         'ex.check(4, my_forest, n_estimators, max_depth)',
    "5": 'n_long, n_short, rebalance_days = 4, 4, 21          # 21 trading days is about a month\n'
         'my_weights = weights_from_forecasts(pred_today, n_long=n_long, n_short=n_short)\n'
         'ex.check(5, my_weights, n_long, n_short, rebalance_days)',
}
