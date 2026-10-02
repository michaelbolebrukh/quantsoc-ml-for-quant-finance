"""The timing rule, tested by breaking the future.

If a feature at t0 secretly read a later row, changing every row after t0 would move it.
So we change every row after t0 (prices AND volumes, in the long file format, before any
reshaping) and demand that the features up to t0 stay bit-identical. The target is checked
against a hand computation from the price 5 trading days ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import data, features as F, splits

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
HOLE = ("DDD", 350, 357)        # DDD has no rows on days 350..356 (a 7-day gap in the file)
LATE = ("FFF", 280)             # FFF starts trading on day 280


def synthetic_long(n_days=420, seed=1) -> pd.DataFrame:
    """Random-walk prices and random volumes in the same long format as data/prices.csv.

    Real files have gaps, so this one does too: a 7-day hole in one symbol and one symbol
    that starts late. A leak through gap filling (for example a backward fill) only shows up
    on rows like these.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2001-01-01", periods=n_days)
    rows = []
    for k, s in enumerate(SYMBOLS):
        px = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days)
        frame = pd.DataFrame({"date": dates, "symbol": s, "open": px, "high": px * 1.01, "low": px * 0.99,
                              "close": px, "adj_close": px, "volume": vol})
        if s == HOLE[0]:
            frame = frame.drop(frame.index[HOLE[1]:HOLE[2]])
        if s == LATE[0]:
            frame = frame.iloc[LATE[1]:]
        rows.append(frame)
    return pd.concat(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def features_of(long: pd.DataFrame, names=None) -> pd.DataFrame:
    return F.build_features(data.to_wide(long), data.to_wide(long, "volume"), names=names)


def scramble_after(long: pd.DataFrame, t0, seed=7) -> pd.DataFrame:
    """Change every numeric value of every row dated after t0."""
    rng = np.random.default_rng(seed)
    out = long.copy()
    later = out["date"] > t0
    for c in ["open", "high", "low", "close", "adj_close"]:
        out.loc[later, c] = out.loc[later, c] * rng.uniform(0.3, 3.0, later.sum())
    out.loc[later, "volume"] = rng.integers(1, 10_000_000, later.sum())
    return out


@pytest.mark.parametrize("pos", [0, 20, 62, 251, 252, 270, 279, 285, 300, 349, 352, 356, 357, 380, 418])
def test_features_never_read_a_later_row(pos):
    long = synthetic_long()
    dates = np.sort(long["date"].unique())
    t0 = dates[pos]
    names = F.FEATURE_NAMES + ["ret_10", "vol_42"]       # also a participant's own choices
    before = features_of(long, names)
    after = features_of(scramble_after(long, t0), names)
    upto = before.index.get_level_values("date") <= t0
    assert upto.sum() == (pos + 1) * len(SYMBOLS)
    pd.testing.assert_frame_equal(before[upto], after[upto], check_exact=True)
    # and the scramble really did change the future, so the check above has teeth
    if pos + 1 < len(dates):
        assert not before[~upto]["ret_1"].equals(after[~upto]["ret_1"])


def test_the_synthetic_panel_really_has_gaps():
    P = data.to_wide(synthetic_long())
    assert P[HOLE[0]].iloc[HOLE[1]:HOLE[2]].isna().all() and P[HOLE[0]].iloc[HOLE[2]:].notna().all()
    assert P[LATE[0]].iloc[:LATE[1]].isna().all() and P[LATE[0]].iloc[LATE[1]:].notna().all()


@pytest.mark.parametrize("name", ["ret_5", "vol_21", "hi52"])
def test_features_are_nan_where_their_window_touches_a_gap(name):
    f = features_of(synthetic_long(), F.FEATURE_NAMES)
    dates = np.sort(synthetic_long()["date"].unique())
    sym, start, end = HOLE
    assert np.isnan(f.loc[(dates[start + 2], sym), name])        # inside the hole: NaN, nothing invented
    assert np.isnan(f.loc[(dates[LATE[1] - 1], LATE[0]), name])   # before the late symbol starts: NaN


def test_features_at_t0_on_real_data_ignore_the_future():
    try:
        long = data.load_prices()
    except FileNotFoundError:
        pytest.skip("data/prices.csv(.gz) is not in this checkout")
    long = long[long["date"] <= "2006-12-31"]
    t0 = pd.Timestamp("2005-06-30")
    before = features_of(long)
    after = features_of(scramble_after(long, t0))
    assert np.array_equal(before.loc[t0].to_numpy(), after.loc[t0].to_numpy(), equal_nan=True)
    assert before.loc[t0].notna().all().all()


def test_leaky_feature_does_read_the_future():
    long = synthetic_long()
    dates = np.sort(long["date"].unique())
    t0 = dates[300]
    before = F.leaky_feature(data.to_wide(long))
    after = F.leaky_feature(data.to_wide(scramble_after(long, t0)))
    assert before.name == "LEAKY_ret_next"
    assert not np.allclose(before.loc[t0].to_numpy(), after.loc[t0].to_numpy())
    # it is exactly tomorrow's return, i.e. the first day of the target
    P = data.to_wide(long)
    sym = "CCC"
    assert before.loc[(t0, sym)] == pytest.approx(P[sym].iloc[301] / P[sym].iloc[300] - 1, rel=1e-12)


def test_target_is_the_return_from_t0_to_five_trading_days_later():
    long = synthetic_long()
    P = data.to_wide(long)
    ds = F.make_dataset(features_of(long), F.forward_return(P))
    dates = P.index
    for pos, sym in [(260, "AAA"), (333, "DDD"), (414, "EEE")]:
        t0 = dates[pos]
        by_hand = P[sym].iloc[pos + 5] / P[sym].iloc[pos] - 1
        assert ds.loc[(t0, sym), "target"] == pytest.approx(by_hand, rel=1e-12)
    # the last 5 dates have no target, so they are not in the dataset
    last = ds.index.get_level_values("date").max()
    assert last == dates[-6]


def test_target_moves_with_day_t_plus_5_and_not_with_day_t_plus_6():
    long = synthetic_long()
    P = data.to_wide(long)
    pos, sym = 300, "BBB"
    base = F.forward_return(P).iloc[pos][sym]
    bumped6 = P.copy(); bumped6.iloc[pos + 6, 1] *= 2
    bumped5 = P.copy(); bumped5.iloc[pos + 5, 1] *= 2
    assert F.forward_return(bumped6).iloc[pos][sym] == base
    assert F.forward_return(bumped5).iloc[pos][sym] != base


def test_feature_values_match_their_definitions():
    long = synthetic_long()
    P, V = data.to_wide(long), data.to_wide(long, "volume")
    f = F.build_features(P, V)
    pos, sym = 300, "AAA"
    t0 = P.index[pos]
    p = P[sym].to_numpy()
    r = p[1:] / p[:-1] - 1                          # r[k] is the return of day k + 1
    v = V[sym].to_numpy().astype(float)
    row = f.loc[(t0, sym)]
    assert row["ret_21"] == pytest.approx(p[pos] / p[pos - 21] - 1, rel=1e-12)
    assert row["vol_21"] == pytest.approx(np.std(r[pos - 21:pos], ddof=1), rel=1e-9)
    w = v[pos - 62:pos + 1]
    assert row["volume_z"] == pytest.approx((v[pos] - w.mean()) / w.std(ddof=1), rel=1e-9)
    assert row["hi52"] == pytest.approx(p[pos] / p[pos - 251:pos + 1].max() - 1, rel=1e-12)


def test_reference_knobs_give_the_default_feature_list():
    assert F.feature_names_for(21, "vol_63") == F.FEATURE_NAMES
    assert F.feature_names_for(10, "vol_42") == ["ret_1", "ret_5", "ret_10", "ret_63", "vol_21", "vol_42", "volume_z", "hi52"]
    assert F.feature_names_for(5, "ret_1") == ["ret_1", "ret_5", "ret_63", "vol_21", "volume_z", "hi52"]


def _dataset():
    long = synthetic_long()
    P = data.to_wide(long)
    return F.make_dataset(features_of(long), F.forward_return(P)), P.index


def test_time_split_enforces_the_embargo():
    ds, dates = _dataset()
    cal = pd.DatetimeIndex(np.unique(ds.index.get_level_values("date")))
    train_end = cal[60]
    with pytest.raises(ValueError, match="too close"):
        splits.time_split(ds, train_end, cal[64])
    with pytest.raises(ValueError, match="after"):
        splits.time_split(ds, train_end, train_end)
    train, test = splits.time_split(ds, train_end, cal[65])     # exactly 5 trading days later is allowed
    assert train.index.get_level_values("date").max() == train_end
    assert test.index.get_level_values("date").min() == cal[65]
    assert splits.embargo_start(ds, train_end) == cal[65]
    assert train.attrs["features"] == ds.attrs["features"]


def test_time_split_counts_trading_days_not_calendar_days():
    ds, _ = _dataset()
    cal = pd.DatetimeIndex(np.unique(ds.index.get_level_values("date")))
    # a Friday train_end: 5 trading days later is the next Friday, 7 calendar days on
    fri = [d for d in cal[50:80] if d.dayofweek == 4][0]
    pos = cal.get_loc(fri)
    with pytest.raises(ValueError):
        splits.time_split(ds, fri, fri + pd.Timedelta(days=6))
    train, test = splits.time_split(ds, fri, cal[pos + 5])
    assert (test.index.get_level_values("date") >= cal[pos + 5]).all()


@pytest.mark.parametrize("helper, arg", [("past_return", 5), ("rolling_vol", 21), ("volume_zscore", 63), ("high_52w", 252)])
def test_feature_helpers_are_safe_on_a_descending_index(helper, arg):
    long = synthetic_long()
    table = data.to_wide(long, "volume") if helper == "volume_zscore" else data.to_wide(long)
    if helper == "rolling_vol":
        table = data.daily_returns(table)
    fn = getattr(F, helper)
    expected = fn(table, arg)
    got = fn(table.iloc[::-1], arg)                # same data, newest row first
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


def test_time_split_accepts_timezone_aware_dates():
    ds, _ = _dataset()
    cal = pd.DatetimeIndex(np.unique(ds.index.get_level_values("date")))
    aware = ds.copy()
    aware.index = aware.index.set_levels(aware.index.levels[0].tz_localize("UTC"), level="date")
    train, test = splits.time_split(aware, cal[60], cal[65])
    assert len(train) == len(splits.time_split(ds, cal[60], cal[65])[0])
    with pytest.raises(ValueError, match="too close"):
        splits.time_split(aware, cal[60], cal[64])


def test_embargo_start_refuses_a_train_end_before_the_data():
    ds, _ = _dataset()
    with pytest.raises(ValueError, match="before the first date"):
        splits.embargo_start(ds, "1990-01-01")


def test_rank_normalise():
    f = features_of(synthetic_long(), F.FEATURE_NAMES)
    r = F.rank_normalise(f)
    assert r.index.equals(f.index) and list(r.columns) == list(f.columns)
    assert r.attrs["features"] == F.FEATURE_NAMES
    vals = r.to_numpy()
    assert np.nanmin(vals) >= -0.5 and np.nanmax(vals) <= 0.5
    means = r.groupby(level="date").mean().dropna(how="all")
    np.testing.assert_allclose(means.fillna(0).to_numpy(), 0.0, atol=1e-12)     # centred on every date
    pd.testing.assert_frame_equal(r.isna(), f.isna())                          # NaN stays NaN, nothing filled
    # only the order within a date matters: a monotone transform of a feature changes nothing
    g = f.copy()
    g["ret_21"] = np.exp(3 * g["ret_21"]) + 7
    pd.testing.assert_frame_equal(F.rank_normalise(g), r, check_exact=True)
    # and it is per date: one date's values never affect another date's ranks
    h = f.copy()
    last = h.index.get_level_values("date") == h.index.get_level_values("date").max()
    h.loc[last, "ret_5"] = -h.loc[last, "ret_5"]
    pd.testing.assert_frame_equal(F.rank_normalise(h)[~last], r[~last], check_exact=True)


def test_compute_feature_vol_is_independent_of_row_order():
    P = data.to_wide(synthetic_long())
    a = F.compute_feature("vol_21", P, None)
    b = F.compute_feature("vol_21", P.sample(frac=1.0, random_state=3), None)
    pd.testing.assert_frame_equal(a, b.sort_index())


def test_embargo_start_refuses_a_train_end_after_the_data():
    ds, _ = _dataset()
    with pytest.raises(ValueError, match="no dates for testing"):
        splits.embargo_start(ds, pd.Timestamp("2099-01-01"))
