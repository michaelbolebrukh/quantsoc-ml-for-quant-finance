"""The backtest, checked by hand on a 5-day, 3-stock panel, plus the 'no credit before the weights' rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import backtest as B
from quantsoc_ml.portfolio import StrategyConfig, weights_from_forecasts

DATES = pd.bdate_range("2020-01-06", periods=5)
RET = pd.DataFrame({"A": [0.0, 0.10, -0.10, 0.0, 0.05],
                    "B": [0.0, -0.05, 0.20, 0.0, -0.05],
                    "C": [0.0, 0.0, 0.05, 0.10, 0.0]}, index=DATES)
PRED = pd.DataFrame({"A": [3, 9, 1, 9, 3], "B": [1, 9, 3, 9, 1], "C": [2, 9, 2, 9, 2]}, index=DATES, dtype=float)
CFG = StrategyConfig(n_long=1, n_short=1, rebalance_days=2, cost_bps=10)


def test_weights_from_forecasts():
    w = weights_from_forecasts(pd.Series({"X": 0.3, "Y": -0.1, "Z": 0.2, "W": np.nan, "V": 0.0}), 2, 1)
    assert w.to_dict() == {"X": 0.5, "Y": -1.0, "Z": 0.5, "W": 0.0, "V": 0.0}
    tie = weights_from_forecasts(pd.Series({"B": 1.0, "A": 1.0, "C": 0.0}), 1, 1)
    assert tie["A"] == 1.0 and tie["C"] == -1.0           # ties go to the earlier name
    with pytest.raises(ValueError):
        weights_from_forecasts(pd.Series({"A": 1.0, "B": np.nan}), 1, 1)
    # +inf (like NaN) counts as no forecast, so it is never ranked
    assert weights_from_forecasts(pd.Series({"A": np.inf, "B": 1.0, "C": 0.0}), 1, 1).to_dict() == {"A": 0.0, "B": 1.0, "C": -1.0}


def test_weights_refuse_duplicate_symbols():
    with pytest.raises(ValueError, match="more than once: A"):
        weights_from_forecasts(pd.Series([1.0, 2.0, 0.0], index=["A", "A", "B"]), 1, 1)


@pytest.mark.parametrize("n_long, n_short", [(1.7, 1), (1, -1), (2, 0.5), ("2", 1)])
def test_weights_refuse_counts_that_are_not_whole_numbers(n_long, n_short):
    with pytest.raises(ValueError, match="whole numbers"):
        weights_from_forecasts(pd.Series({"A": 1.0, "B": 0.5, "C": 0.0}), n_long, n_short)


def test_long_short_equity_by_hand():
    res = B.run_backtest(PRED, RET, CFG)
    # Track holdings in money, starting with 1 of capital (short proceeds sit in cash, earning 0).
    # d0 close: long A 1, short B 1; cost 0.001 * 2
    c0 = 0.001 * 2
    # d1: A 1 -> 1.10, B -1 -> -0.95: value 1 + 1.10 - 0.95 = 1.15
    # d2: A -> 0.99, B -> -1.14: value 0.85, so the day's return is 0.85 / 1.15 - 1
    #     drifted weights A 0.99/0.85, B -1.14/0.85; new: long B, short A
    c2 = 0.001 * ((0.99 / 0.85 + 1) + (1.14 / 0.85 + 1))
    # d3: A and B do not move: return 0. d4: short A -5%, long B -5%: return -0.10
    #     drifted weights A -1.05/0.9, B 0.95/0.9; new: long A, short B
    c4 = 0.001 * ((1.05 / 0.9 + 1) + (0.95 / 0.9 + 1))
    expected = [1 - c0, 1.15, (0.85 / 1.15) * (1 - c2), 1.0, 0.90 * (1 - c4)]
    np.testing.assert_allclose(1 + res.daily_returns.to_numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(res.equity.iloc[-1], np.prod(expected), rtol=1e-12)
    np.testing.assert_allclose(res.weights.iloc[1].to_numpy(), [1.10 / 1.15, -0.95 / 1.15, 0.0], rtol=1e-12)
    np.testing.assert_allclose(res.weights.iloc[3].to_numpy(), [-1.0, 1.0, 0.0])
    s = res.stats
    assert s["n_rebalances"] == 3
    assert s["avg_turnover"] == pytest.approx((2 + c2 / 0.001 + c4 / 0.001) / 3, rel=1e-12)
    assert s["total_return"] == pytest.approx(np.prod(expected) - 1, rel=1e-12)
    eq = np.cumprod(expected)
    assert s["max_drawdown"] == pytest.approx(min(eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1), rel=1e-12)


def test_equal_weight_benchmark_by_hand():
    res = B.run_benchmark(RET, DATES, CFG)
    third = 1 / 3
    # d1 holdings A 1.10/3, B 0.95/3, C 1/3; d2 A 0.99/3, B 1.14/3, C 1.05/3 (value 1.06)
    v1, v2 = (1.10 + 0.95 + 1.0) / 3, (0.99 + 1.14 + 1.05) / 3
    c2 = 0.001 * sum(abs(third - h / 3 / v2) for h in (0.99, 1.14, 1.05))
    # d3: only C moves +10%: holdings 0.99, 1.14, 1.155 (out of 3) after the rebalance back to thirds
    v3 = third * (1 + 1 + 1.10)
    # d4: drift from d2 close thirds: A 1/3 * 1.0 * 1.05, B 1/3 * 1.0 * 0.95, C 1/3 * 1.10 * 1.0
    h4 = np.array([1.05, 0.95, 1.10]) / 3
    v4 = h4.sum()
    c4 = 0.001 * np.abs(third - h4 / v4).sum()
    expected = [1 - 0.001, v1, v2 / v1 * (1 - c2), v3, v4 / v3 * (1 - c4)]
    np.testing.assert_allclose(1 + res.daily_returns.to_numpy(), expected, rtol=1e-12)
    eq = B.benchmark_equal_weight(RET, DATES, CFG)
    np.testing.assert_allclose(eq.to_numpy(), np.cumprod(expected), rtol=1e-12)


def test_returns_are_never_credited_before_the_weights_exist():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=60)
    cols = list("ABCDEFGH")
    ret = pd.DataFrame(rng.normal(0, 0.02, (60, 8)), index=dates, columns=cols)
    pred = pd.DataFrame(rng.normal(size=(60, 8)), index=dates, columns=cols)
    cfg = StrategyConfig(n_long=2, n_short=2, rebalance_days=3, cost_bps=0)
    base = B.run_backtest(pred, ret, cfg).daily_returns
    for k in (10, 30, 45):
        changed = pred.copy()
        changed.iloc[k:] = rng.normal(size=changed.iloc[k:].shape)
        new = B.run_backtest(changed, ret, cfg).daily_returns
        # forecasts from day k on can only affect returns from day k + 1 on
        pd.testing.assert_series_equal(base.iloc[:k + 1], new.iloc[:k + 1], check_exact=True)


def test_foresight_of_tomorrow_pays_and_of_today_does_not():
    # iid returns over 1000 days: a forecast equal to TODAY's return (already earned when the
    # weights are formed) has no edge, so its Sharpe is noise (standard error about 0.5)
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2010-01-01", periods=1000)
    ret = pd.DataFrame(rng.normal(0, 0.02, (1000, 8)), index=dates, columns=list("ABCDEFGH"))
    daily = StrategyConfig(n_long=2, n_short=2, rebalance_days=1, cost_bps=0)
    tomorrow = B.run_backtest(ret.shift(-1), ret, daily)
    today = B.run_backtest(ret, ret, daily)
    assert tomorrow.stats["sharpe"] > 10
    assert abs(today.stats["sharpe"]) < 2


def test_costs_scale_with_turnover():
    free = B.run_backtest(PRED, RET, StrategyConfig(n_long=1, n_short=1, rebalance_days=2, cost_bps=0))
    paid = B.run_backtest(PRED, RET, CFG)
    np.testing.assert_allclose((1 + paid.daily_returns) / (1 + free.daily_returns), 1 - paid.costs, rtol=1e-12)
    np.testing.assert_allclose(paid.costs, 0.001 * paid.turnover, rtol=1e-12)


@pytest.mark.parametrize("short_gain", [1.0, 1.5])
def test_a_short_that_doubles_or_more_blows_up_the_book(short_gain):
    # long A, short B with all the capital; B rises by 100% (or 150%) on day 1: equity hits 0 (or below)
    dates = pd.bdate_range("2020-01-06", periods=5)
    ret = pd.DataFrame({"A": [0.0, 0.0, 0.01, 0.01, 0.01], "B": [0.0, short_gain, 0.01, -0.02, 0.01]}, index=dates)
    pred = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=dates)
    res = B.run_backtest(pred, ret, StrategyConfig(n_long=1, n_short=1, rebalance_days=2, cost_bps=10))
    assert res.equity.iloc[0] == pytest.approx(1 - 0.002)
    assert (res.equity.iloc[1:] == 0).all()                  # wiped out, and it stays wiped out
    assert res.daily_returns.iloc[1] == -1.0                 # the losing day is recorded as a total loss
    assert res.daily_returns.iloc[2:].isna().all()           # no return on capital that no longer exists
    assert (res.weights.iloc[1:] == 0).all().all()           # no positions and no further trading
    assert (res.turnover.iloc[1:] == 0).all()
    s = res.stats
    assert s["blown_up"] is True
    assert s["total_return"] == -1.0
    for k in ("sharpe", "annual_vol", "max_drawdown"):
        assert np.isnan(s[k]), k


def test_a_normal_run_is_not_blown_up():
    assert B.run_backtest(PRED, RET, CFG).stats["blown_up"] is False


def test_a_forecast_symbol_with_no_returns_is_refused():
    pred = PRED.assign(ZZZ=5.0, YYY=4.0)
    with pytest.raises(ValueError, match="YYY, ZZZ"):
        B.run_backtest(pred, RET, CFG)


def test_rank_features_flag_round_trips():
    assert StrategyConfig().rank_features is True
    off = StrategyConfig(rank_features=False)
    assert off.to_dict()["rank_features"] is False
    back = StrategyConfig.from_dict(off.to_dict())
    assert back == off and back.rank_features is False
    assert StrategyConfig.from_dict(StrategyConfig().to_dict()) == StrategyConfig()


def test_a_cost_at_or_above_the_whole_book_blows_up():
    """backtest.py's cost branch: a rebalance whose cost is 100% of equity ends the book."""
    dates = pd.bdate_range("2020-01-06", periods=5)
    ret = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, 0.01], "B": [0.0, 0.0, 0.01, -0.02, 0.01]}, index=dates)
    pred = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=dates)
    # the first rebalance trades 2 units of weight (1 long + 1 short): 2 * 4999 bps < 100%, 2 * 5000 bps = 100%
    ok = B.run_backtest(pred, ret, StrategyConfig(n_long=1, n_short=1, rebalance_days=1, cost_bps=4999))
    assert ok.stats["blown_up"] is False
    dead = B.run_backtest(pred, ret, StrategyConfig(n_long=1, n_short=1, rebalance_days=1, cost_bps=5000))
    assert dead.stats["blown_up"] is True
    assert (dead.equity == 0).all() and (dead.weights == 0).all().all()
