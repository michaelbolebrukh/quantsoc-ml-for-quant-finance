"""Features and the target. The timing rule of the whole workshop lives in this file.

Timing rule: a feature for date t may only use rows with date <= t. Every function here is a
backward-looking pandas operation (shift by a positive number, or a trailing rolling window),
so it cannot read a later row. The target is the one thing that looks forward, on purpose:

    target[t] = adj_close[t + 5] / adj_close[t] - 1        (per symbol, t + 5 trading days)

tests/test_core_timing.py changes every price after a date t0 and checks that the features at
t0 do not move by a single bit, and that the target at t0 equals the hand-computed value.
`leaky_feature` breaks the rule deliberately, for the leakage demo, and the test checks that
it DOES move.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import HORIZON
from .data import daily_returns

# Every helper below sorts its input by date first, so it is safe to call on a table in any
# row order (the notebook's task cells call them directly). The result is in date order.

FEATURE_NAMES: List[str] = ["ret_1", "ret_5", "ret_21", "ret_63", "vol_21", "vol_63", "volume_z", "hi52"]
TARGET = "target"
LEAKY_NAME = "LEAKY_ret_next"
VOLUME_WINDOW = 63
HIGH_WINDOW = 252

FEATURE_DESCRIPTIONS = {
    "ret_N": "past return over N trading days: adj_close[t] / adj_close[t-N] - 1",
    "vol_N": "standard deviation of the last N daily returns (ending at t)",
    "volume_z": "today's volume compared with its 63-day mean, in standard deviations",
    "hi52": "adj_close[t] / highest adj_close of the last 252 days - 1 (0 at a new high, negative below)",
}

_RET = re.compile(r"^ret_(\d+)$")
_VOL = re.compile(r"^vol_(\d+)$")


def past_return(prices_wide: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return over the last n trading days, ending at t. Uses rows t and t-n only."""
    if int(n) < 1:
        raise ValueError("the lookback n must be a whole number of days, at least 1")
    prices_wide = prices_wide.sort_index()          # oldest first, so shift(n) means n days EARLIER
    return prices_wide / prices_wide.shift(int(n)) - 1


def rolling_vol(returns_wide: pd.DataFrame, n: int) -> pd.DataFrame:
    """Standard deviation of the n daily returns ending at t (NaN until n returns exist)."""
    if int(n) < 2:
        raise ValueError("a volatility window needs at least 2 days")
    return returns_wide.sort_index().rolling(int(n), min_periods=int(n)).std()


def volume_zscore(volume_wide: pd.DataFrame, window: int = VOLUME_WINDOW) -> pd.DataFrame:
    """(volume[t] - mean of the last `window` volumes) / their std. The window ends at t, inclusive.

    A day with zero spread in the window gives NaN rather than an infinite score.
    """
    v = volume_wide.sort_index().astype(float)
    roll = v.rolling(window, min_periods=window)
    z = (v - roll.mean()) / roll.std()
    return z.replace([np.inf, -np.inf], np.nan)


def high_52w(prices_wide: pd.DataFrame, window: int = HIGH_WINDOW) -> pd.DataFrame:
    """adj_close[t] / max(adj_close over the last `window` days, including t) - 1. Always <= 0."""
    prices_wide = prices_wide.sort_index()
    return prices_wide / prices_wide.rolling(window, min_periods=window).max() - 1


def compute_feature(name: str, prices_wide: pd.DataFrame, volume_wide: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One feature as a date x symbol table, by name: ret_N, vol_N, volume_z or hi52.

    Names are parsed rather than listed so that a participant's own choice (ret_10, vol_42)
    works without editing this file.
    """
    m = _RET.match(name)
    if m:
        return past_return(prices_wide, int(m.group(1)))
    m = _VOL.match(name)
    if m:
        return rolling_vol(daily_returns(prices_wide), int(m.group(1)))
    if name == "volume_z":
        if volume_wide is None:
            raise ValueError("volume_z needs the volume table")
        return volume_zscore(volume_wide.reindex(index=prices_wide.index, columns=prices_wide.columns))
    if name == "hi52":
        return high_52w(prices_wide)
    raise ValueError(f"unknown feature name {name!r}: use ret_N, vol_N, volume_z or hi52")


def feature_names_for(momentum_lookback: int = 21, extra_feature: str = "vol_63") -> List[str]:
    """The model's feature list for a strategy's knobs.

    The default list has a momentum slot (ret_21) and an extra slot (vol_63). The two knobs
    replace those slots, so with the reference knobs the list equals FEATURE_NAMES exactly.
    A choice that duplicates another feature (for example ret_5) simply appears once.
    """
    slots = ["ret_1", "ret_5", f"ret_{int(momentum_lookback)}", "ret_63", "vol_21", str(extra_feature), "volume_z", "hi52"]
    out: List[str] = []
    for s in slots:
        if s not in out:
            out.append(s)
    return out


def _long(wide: pd.DataFrame) -> np.ndarray:
    """Row-major flatten: matches the (date, symbol) order of `_panel_index`."""
    return wide.to_numpy(dtype=float).ravel()


def _panel_index(prices_wide: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([prices_wide.index, prices_wide.columns], names=["date", "symbol"])


def build_features(prices_wide: pd.DataFrame, volume_wide: Optional[pd.DataFrame] = None,
                   names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Feature table with a (date, symbol) MultiIndex and one column per feature.

    `names` defaults to FEATURE_NAMES; pass `feature_names_for(...)` for a strategy's own list.
    The list used is stored in `.attrs["features"]` so `models.xy` picks the right columns.
    Rows keep their NaNs (warm-up at the start); `make_dataset` drops them.
    """
    names = list(names) if names is not None else list(FEATURE_NAMES)
    prices_wide = prices_wide.sort_index()
    data = {n: _long(compute_feature(n, prices_wide, volume_wide)) for n in names}
    out = pd.DataFrame(data, index=_panel_index(prices_wide))
    out.attrs["features"] = names
    return out


def rank_normalise(feats: pd.DataFrame) -> pd.DataFrame:
    """Replace each feature by its rank among the stocks on the same date, centred on 0.

    value = (rank - 0.5) / count - 0.5, per date and per feature, so the lowest stock is near
    -0.5, the highest near +0.5 and the date's mean is exactly 0 (ties share the average rank).
    Why: a long/short book only needs to know which stocks look better than the others TODAY,
    and ranks throw away the market-wide swings and outliers that otherwise dominate the
    raw numbers (Gu, Kelly and Xiu do the same). It only compares stocks within one date, so it
    keeps the timing rule. NaN stays NaN and is left out of the count.
    """
    names = list(feats.attrs.get("features", feats.columns))
    cols = [c for c in feats.columns if c in names]
    by_date = feats[cols].groupby(level="date")
    out = feats.copy()
    out[cols] = (by_date.rank(method="average") - 0.5) / by_date.transform("count") - 0.5
    out.attrs["features"] = names
    return out


def forward_return(prices_wide: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """The target: adj_close[t + horizon] / adj_close[t] - 1 per symbol (NaN in the last rows).

    This is the ONLY forward-looking calculation in the package. It is what we try to predict,
    never an input to the model.
    """
    prices_wide = prices_wide.sort_index()
    return prices_wide.shift(-int(horizon)) / prices_wide - 1


def leaky_feature(prices_wide: pd.DataFrame) -> pd.Series:
    """DELIBERATELY WRONG: tomorrow's return, adj_close[t+1] / adj_close[t] - 1, as a 'feature'.

    It is part of the target itself, so a model fed it looks brilliant in research and is
    impossible to trade. Returned as a long Series named LEAKY_ret_next on the same
    (date, symbol) index as `build_features`, so it can be added with `features.join(...)`.
    """
    prices_wide = prices_wide.sort_index()
    leak = prices_wide.shift(-1) / prices_wide - 1
    return pd.Series(_long(leak), index=_panel_index(prices_wide), name=LEAKY_NAME)


def make_dataset(features: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Rows (date, symbol), columns = the features + "target", with every incomplete row dropped.

    The target table is wide (date x symbol) and is aligned onto the feature index, so a row
    pairs the features known at the close of t with the return from t to t + horizon.
    """
    names = list(features.attrs.get("features", features.columns))
    tgt = target.reindex(index=features.index.get_level_values("date").unique(), columns=features.index.get_level_values("symbol").unique())
    stacked = pd.Series(tgt.to_numpy(dtype=float).ravel(),
                        index=pd.MultiIndex.from_product([tgt.index, tgt.columns], names=["date", "symbol"]))
    ds = features[names].copy()
    ds[TARGET] = stacked.reindex(ds.index).to_numpy()
    ds = ds.dropna()
    ds.attrs["features"] = names
    return ds
