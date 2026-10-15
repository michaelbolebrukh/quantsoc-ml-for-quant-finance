"""Workshop 3 strategy and research code: no look-ahead, the paper's rules, and the textbook formulas."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import research as rs
from quantsoc_ml import statarb as sa

ROOT = Path(__file__).resolve().parent.parent


def toy_prices(n_days=400, n_stocks=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    r = rng.normal(0, 0.01, (n_days, n_stocks))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates,
                        columns=[f"S{i:02d}" for i in range(n_stocks)])


def test_features_use_no_future_prices():
    P = toy_prices()
    t = P.index[300]
    a = sa.lagged_returns(P).xs(t, level="date")
    Q = P.copy()
    Q.loc[Q.index > t] *= 3.0                       # rewrite the future
    b = sa.lagged_returns(Q).xs(t, level="date")
    pd.testing.assert_frame_equal(a, b)


def test_feature_values_match_definition():
    P = toy_prices()
    t = P.index[300]
    f = sa.lagged_returns(P).loc[(t, "S03")]
    assert f["R1"] == pytest.approx(P.loc[t, "S03"] / P["S03"].iloc[299] - 1)
    assert f["R240"] == pytest.approx(P.loc[t, "S03"] / P["S03"].iloc[300 - 240] - 1)
    assert list(f.index) == sa.FEATURES and len(sa.FEATURES) == 31


def test_target_is_beats_tomorrows_median():
    P = toy_prices()
    t, t1 = P.index[100], P.index[101]
    r = P.loc[t1] / P.loc[t] - 1
    y = sa.median_target(P).loc[t]
    assert (y == (r > r.median()).astype(float)).all()
    assert sa.median_target(P).iloc[-1].isna().all()        # tomorrow unknown on the last day


def test_study_periods_partition_trading_days():
    dates = pd.bdate_range("2010-01-01", periods=1900)
    ps = sa.study_periods(dates)
    traded = pd.DatetimeIndex(np.concatenate([dates[(dates >= p.trade_start) & (dates <= p.trade_end)] for p in ps]))
    assert traded.is_unique and traded.equals(dates[750:])
    for p in ps:
        assert p.train_end < p.trade_start
        assert len(dates[(dates >= p.train_start) & (dates <= p.train_end)]) == 750


def test_period_predictions_ignore_the_future():
    P = toy_prices(n_days=1100)
    panel = sa.make_panel(P)
    p = sa.study_periods(panel.index.get_level_values("date").unique(), train_days=500, trade_days=100)[0]
    a, _ = sa.fit_predict_period(panel, p, ["boosting"])
    Q = P.copy()
    Q.loc[Q.index > p.trade_start] *= np.random.default_rng(1).uniform(0.5, 1.5, Q.loc[Q.index > p.trade_start].shape)
    b, _ = sa.fit_predict_period(sa.make_panel(Q), p, ["boosting"])
    first = a.index.get_level_values("date") == p.trade_start
    np.testing.assert_allclose(a.loc[first, "boosting"], b.loc[first, "boosting"])


def test_daily_weights_and_long_short_arithmetic():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=2), list("ABCDEF")], names=["date", "symbol"])
    scores = pd.Series([6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    fwd = pd.Series([0.03, 0.01, 0, 0, -0.01, -0.02, 0.0, 0.0, 0, 0, 0.01, 0.02], index=idx)
    w = sa.daily_weights(scores, k=2)
    assert w.groupby(level="date").sum().abs().max() < 1e-12
    assert (w.groupby(level="date").apply(lambda s: (s != 0).sum()) == 4).all()
    ls = sa.long_short(scores, fwd, k=2, cost_bps=10)
    # day 1: long A,B (0.03, 0.01), short E,F (-0.01, -0.02): mean long 0.02 - mean short -0.015 = 0.035
    assert ls.gross.iloc[0] == pytest.approx(0.035)
    assert ls.turnover.iloc[0] == pytest.approx(2.0) and ls.turnover.iloc[1] == pytest.approx(4.0)
    assert ls.net.iloc[1] == pytest.approx(ls.gross.iloc[1] - 4.0 * 10 / 1e4)


def test_banded_weights_reduce_turnover_and_match_daily_when_exit_equals_k():
    P = toy_prices(n_days=300)
    panel = sa.make_panel(P, lags=[1, 5])
    scores = -panel["R1"]
    pd.testing.assert_series_equal(sa.banded_weights(scores, 5, 5).sort_index(), sa.daily_weights(scores, 5).sort_index(),
                                   check_names=False)
    fast = sa.long_short(scores, panel["fwd"], k=5).turnover.mean()
    slow = sa.long_short(scores, panel["fwd"], k=5, exit_k=12).turnover.mean()
    assert slow < fast
    w = sa.banded_weights(scores, 5, 12)
    assert (w.groupby(level="date").apply(lambda s: (s > 0).sum()) == 5).all()
    assert (w.groupby(level="date").apply(lambda s: (s < 0).sum()) == 5).all()


def test_daily_ic_matches_pandas_spearman():
    P = toy_prices()
    panel = sa.make_panel(P, lags=[1, 3]).dropna()
    ic = sa.daily_ic(panel["R3"], panel["fwd"])
    d = panel.index.get_level_values("date").unique()[10]
    g = panel.xs(d, level="date")
    assert ic[d] == pytest.approx(g["R3"].rank().corr(g["fwd"].rank()))


def test_live_strategy_weights_and_history():
    P = toy_prices(n_days=1000, n_stocks=40)
    strat = sa.train_live(sa.make_panel(P), "boosting", train_days=300, k=5, gross_exposure=1.0)
    s = strat.scores(P)
    assert s.name == P.index[-1] and len(s) == 40
    w = strat.weights(s)
    assert w.abs().sum() == pytest.approx(1.0) and w.sum() == pytest.approx(0.0)
    assert (w > 0).sum() == 5 and set(w[w > 0].index) == set(s.index[:5])
    assert strat.history_days == 241


def test_expected_max_sharpe_and_psr():
    assert rs.expected_max_sharpe(1000, 1.0) == pytest.approx(3.26, abs=0.01)   # Bailey & Lopez de Prado (2014)
    assert rs.expected_max_sharpe(1, 1.0) == 0.0
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 5000))
    assert rs.probabilistic_sharpe(r, rs.sharpe(r)) == pytest.approx(0.5)
    d = rs.deflated_sharpe(r, [rs.sharpe(r), 0.0, -0.01, 0.01])
    assert d["DSR"] < d["PSR_vs_zero"]


def test_purged_kfold_removes_overlap_and_embargo():
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=20), 3)
    folds = list(rs.purged_kfold(dates, 4, label_horizon=2, embargo=1))
    days = pd.DatetimeIndex(dates)
    cal = days.unique()
    for train, test in folds:
        assert not (train & test).any()
        t_dates = cal.get_indexer(days[test])
        tr_dates = cal.get_indexer(days[train])
        lo, hi = t_dates.min(), t_dates.max()
        assert not ((tr_dates >= lo - 2) & (tr_dates <= hi + 2)).any()     # purged on both sides
    assert [len(np.unique(days[tr])) for tr, _ in folds] == [13, 11, 11, 13]
    # a larger embargo extends the gap after the block only
    folds = list(rs.purged_kfold(dates, 4, label_horizon=1, embargo=3))
    assert [len(np.unique(days[tr])) for tr, _ in folds] == [12, 11, 11, 14]


def test_cache_covers_every_trading_day_once():
    pred = sa.load_cached_predictions()
    assert pred.index.is_unique
    panel_dates = sa.make_panel(sa.wide(sa.load_prices())).index.get_level_values("date").unique()
    periods = sa.study_periods(panel_dates)
    assert pred.index.get_level_values("date").min() == periods[0].trade_start
    assert pred.index.get_level_values("date").max() == periods[-1].trade_end
    assert set(pred["period"].unique()) == {p.number for p in periods}


def test_notebook_text_has_no_dashes_and_no_outputs():
    nb = json.loads((ROOT / "workshop3" / "workshop3.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        assert not re.search("[–—]", text), text[:80]
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_last_day_without_a_next_day_return_is_not_traded():
    P = toy_prices(n_days=300)
    panel = sa.make_panel(P, lags=[1, 5])
    ls = sa.long_short(-panel["R1"], panel["fwd"], k=5)
    assert ls.gross.index[-1] == P.index[-2]          # P.index[-1] has no tomorrow yet


def test_hit_rate_ignores_flat_days_and_window_slices():
    r = pd.Series([0.01, -0.01, 0.0, 0.0, 0.02], index=pd.bdate_range("2020-01-01", periods=5))
    assert sa.perf_stats(r)["hit rate"] == pytest.approx(2 / 3)
    P = toy_prices(n_days=300)
    panel = sa.make_panel(P, lags=[1, 5])
    ls = sa.long_short(-panel["R1"], panel["fwd"], k=5)
    w = ls.window(end=P.index[200])
    assert w.net.index.max() <= P.index[200] and w.net.equals(ls.net.loc[:P.index[200]])
