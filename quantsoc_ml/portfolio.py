"""From forecasts to a portfolio: the five knobs, and the long/short weighting rule.

Rule (dollar-neutral long/short, unleveraged on each side):
  1. keep symbols with a finite forecast
  2. rank by forecast, highest first; ties broken by symbol name so the result is repeatable
  3. the top n_long each get +1/n_long (together 100% long)
  4. the bottom n_short each get -1/n_short (together 100% short)
  5. everyone else gets 0

Only the ORDER of the forecasts matters, not their size. That is deliberate: the forecasts are
tiny and noisy, and ranking them is far more robust than trusting their exact values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Mapping

import numpy as np
import pandas as pd

from .features import feature_names_for


@dataclass
class StrategyConfig:
    """The knobs of one strategy. Defaults are the reference strategy (contract decision D7):
    rebalance every 21 trading days, because at 5 bps per unit traded a weekly rebalance costs
    about as much as the forecasts earn."""

    momentum_lookback: int = 21       # Task 1: days in the momentum feature ret_N
    extra_feature: str = "vol_63"     # Task 2: the extra feature, e.g. vol_63
    train_end: str = "2014-12-31"     # Task 3: last training date
    test_start: str = "2015-01-12"    # Task 3: first test date (at least 5 trading days later)
    n_estimators: int = 200           # Task 4: trees in the forest
    max_depth: int = 6                # Task 4: how many questions each tree may ask
    n_long: int = 4                   # Task 5: stocks bought
    n_short: int = 4                  # Task 5: stocks sold short
    rebalance_days: int = 21          # trade every 21 trading days (about monthly), hold in between;
                                      # costs are about the size of the edge, so trading less often matters
    cost_bps: float = 5.0             # cost per unit of weight traded, in basis points (5 bps = 0.05%)
    rank_features: bool = True        # features are rank-normalised across stocks each day before modelling;
                                      # the export and the paper runner read this flag so live forecasts are
                                      # built the same way as the training data

    @property
    def feature_names(self) -> List[str]:
        """The model's feature list for these knobs (equals FEATURE_NAMES for the reference)."""
        return feature_names_for(self.momentum_lookback, self.extra_feature)

    def validate(self) -> "StrategyConfig":
        for name in ("momentum_lookback", "n_estimators", "max_depth", "rebalance_days", "n_long", "n_short"):
            v = getattr(self, name)
            if isinstance(v, bool) or float(v) != int(float(v)):
                raise ValueError(f"{name} must be a whole number, not {v!r}")
        for name in ("momentum_lookback", "n_estimators", "max_depth", "rebalance_days"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a whole number of at least 1")
        if int(self.n_long) < 0 or int(self.n_short) < 0 or int(self.n_long) + int(self.n_short) < 1:
            raise ValueError("n_long and n_short must be 0 or more, and at least one of them above 0")
        if float(self.cost_bps) < 0:
            raise ValueError("cost_bps cannot be negative")
        if pd.Timestamp(self.test_start) <= pd.Timestamp(self.train_end):
            raise ValueError("test_start must come after train_end")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "StrategyConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d}).validate()


def _count(n) -> int:
    """A stock count as an int; 4 and 4.0 pass, 1.7, -1 and "4" do not (no silent truncation)."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, float, np.integer, np.floating)) \
            or not float(n).is_integer() or n < 0:
        raise ValueError(f"n_long and n_short must be whole numbers of 0 or more, got {n!r}")
    return int(n)


def weights_from_forecasts(pred: pd.Series, n_long: int = 4, n_short: int = 4) -> pd.Series:
    """Target weights indexed like `pred`: +1/n_long for the top n_long, -1/n_short for the bottom n_short.

    Only finite forecasts are ranked: NaN, +inf and -inf all count as 'no forecast' and get 0.
    Raises ValueError when fewer than n_long + n_short symbols have a forecast (a stock would
    have to be long and short at once), when a symbol appears twice, or when a count is not a
    whole number of 0 or more.
    """
    n_long, n_short = _count(n_long), _count(n_short)
    if n_long + n_short < 1:
        raise ValueError("n_long and n_short must be 0 or more, and at least one of them above 0")
    pred = pd.Series(pred, dtype=float)
    if pred.index.has_duplicates:
        dupes = ", ".join(sorted(map(str, pred.index[pred.index.duplicated()].unique())))
        raise ValueError(f"each symbol must appear once, but these appear more than once: {dupes}")
    finite = pred[np.isfinite(pred.to_numpy())]
    if len(finite) < n_long + n_short:
        raise ValueError(f"only {len(finite)} symbols have a forecast, fewer than n_long + n_short = {n_long + n_short}")
    order = sorted(finite.index, key=lambda s: (-finite[s], str(s)))
    w = pd.Series(0.0, index=pred.index, name="weight")
    if n_long:
        w[order[:n_long]] = 1.0 / n_long
    if n_short:
        w[order[len(order) - n_short:]] = -1.0 / n_short
    return w
