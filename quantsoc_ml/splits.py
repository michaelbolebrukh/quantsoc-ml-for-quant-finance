"""Split the dataset by time, with an embargo between training and test.

Why an embargo: the target on the last training day t looks 5 trading days ahead, so it is
only known at the close of t + 5. A test that starts earlier would overlap the training
targets, and the model could only have been trained after the test had already begun. So
every test row must be dated at least `horizon` trading days after the last training day.

`random_split` is the Workshop 1 style shuffled split. It is here ONLY for the demo of why a
random split lies on time series: neighbouring days share most of their 5-day target, so a
shuffled test set is full of near copies of training rows.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from . import HORIZON


def _dates(df: pd.DataFrame) -> pd.Index:
    return df.index.get_level_values("date")


def _calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.unique(_dates(df).to_numpy())).sort_values()


def _like(ts, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """A boundary date in the dataset's own time zone convention, so comparisons never fail.

    A plain date given for timezone-aware data is read in the data's time zone; an aware date
    given for plain data is converted to UTC and then made plain.
    """
    ts = pd.Timestamp(ts)
    tz = calendar.tz
    if tz is not None:
        return ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts


def _keep_attrs(src: pd.DataFrame, *frames: pd.DataFrame):
    for f in frames:
        f.attrs.update(src.attrs)
    return frames


def time_split(dataset: pd.DataFrame, train_end, test_start, horizon: int = HORIZON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): train = rows dated <= train_end, test = rows dated >= test_start.

    Trading days are counted on the dataset's own dates. Raises ValueError with a plain
    sentence if the test would start less than `horizon` trading days after training ends,
    or if either side would be empty. Rows in the gap belong to neither side.
    """
    dates = _dates(dataset)
    calendar = _calendar(dataset)
    train_end, test_start = _like(train_end, calendar), _like(test_start, calendar)
    if test_start <= train_end:
        raise ValueError(f"test_start ({test_start.date()}) must come after train_end ({train_end.date()}).")
    train_days = calendar[calendar <= train_end]
    if len(train_days) == 0:
        raise ValueError(f"no data on or before train_end ({train_end.date()}), so the training set would be empty.")
    last_train_pos = len(train_days) - 1
    earliest_pos = last_train_pos + int(horizon)
    if earliest_pos >= len(calendar):
        raise ValueError(f"train_end ({train_end.date()}) leaves no room for a test set after a {horizon}-day embargo.")
    earliest = calendar[earliest_pos]
    if test_start < earliest:
        raise ValueError(
            f"test_start ({test_start.date()}) is too close to train_end ({train_end.date()}): the last training "
            f"target is only known {horizon} trading days later, so the test must start on or after {earliest.date()}.")
    train = dataset[dates <= train_end].copy()
    test = dataset[dates >= test_start].copy()
    if len(test) == 0:
        raise ValueError(f"no data on or after test_start ({test_start.date()}), so the test set would be empty.")
    return _keep_attrs(dataset, train, test)


def embargo_start(dataset: pd.DataFrame, train_end, horizon: int = HORIZON) -> pd.Timestamp:
    """Earliest allowed test_start for a given train_end (handy for a checker or a hint)."""
    calendar = _calendar(dataset)
    train_end = _like(train_end, calendar)
    if train_end < calendar[0]:
        raise ValueError(f"train_end ({train_end.date()}) is before the first date in the data ({calendar[0].date()}).")
    if train_end >= calendar[-1]:
        raise ValueError(f"train_end ({train_end.date()}) leaves no dates for testing: the data ends {calendar[-1].date()}.")
    pos = int((calendar <= train_end).sum()) - 1 + int(horizon)
    return calendar[min(pos, len(calendar) - 1)]


def random_split(dataset: pd.DataFrame, test_size: float = 0.25, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """WRONG ON PURPOSE: a shuffled split (sklearn train_test_split), for the 'random splits lie' demo."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(dataset, test_size=test_size, shuffle=True, random_state=seed)
    return _keep_attrs(dataset, train.sort_index(), test.sort_index())


def split_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """One row per side: first and last date, trading days, rows, and the gap between them."""
    rows = {}
    for name, df in (("train", train_df), ("test", test_df)):
        d = _dates(df)
        rows[name] = {"first_date": d.min().date(), "last_date": d.max().date(),
                      "trading_days": int(pd.Index(d).nunique()), "rows": int(len(df))}
    out = pd.DataFrame(rows).T
    out.index.name = "split"
    gap = (pd.Timestamp(rows["test"]["first_date"]) - pd.Timestamp(rows["train"]["last_date"])).days
    out["calendar_days_since_train_end"] = [None, gap]
    return out
