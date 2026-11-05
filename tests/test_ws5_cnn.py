"""Workshop 5 (Jiang, Kelly & Xiu): the charts are drawn the paper's way, the networks have the paper's
sizes, the portfolio arithmetic is right, and nothing looks ahead."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import cnn

torch = pytest.importorskip("torch", reason="PyTorch is not installed in this environment")


# --------------------------------------------------------------------------- the paper's sizes

@pytest.mark.parametrize("days", [5, 20, 60])
def test_network_sizes_match_the_paper(days):
    model = cnn.build_cnn(days)
    assert cnn.count_params(model) == cnn.PAPER_PARAMS[days]
    fc_inputs = {5: 15_360, 20: 46_080, 60: 184_320}                 # sec. 3.1, as printed
    assert model[-1].in_features == fc_inputs[days]
    c, h, w = cnn.block_shapes(days)[-1]
    assert c * h * w == fc_inputs[days] and w == 3 * days          # the width is never pooled
    assert len(cnn.block_shapes(days)) == {5: 2, 20: 3, 60: 4}[days]


def test_picture_sizes_match_table_1():
    for days, (h, w, pr, vr) in cnn.SHAPES.items():
        assert w == 3 * days and h == pr + 1 + vr
    assert cnn.SHAPES[5][:2] == (32, 15) and cnn.SHAPES[20][:2] == (64, 60) and cnn.SHAPES[60][:2] == (96, 180)


def test_figure_5_convolution_and_pooling_numbers():
    x = torch.zeros(1, 1, 6, 6)
    x[0, 0, :4, :4] = 255.0                                          # white 4x4 corner, black elsewhere
    f1 = torch.tensor([[1.0, 0, -1]] * 3).view(1, 1, 3, 3)           # vertical edges
    f2 = f1.transpose(2, 3).contiguous()                             # rows (1,1,1), (0,0,0), (-1,-1,-1)
    out1 = torch.nn.functional.max_pool2d(torch.nn.functional.conv2d(x, f1), 2)[0, 0]
    out2 = torch.nn.functional.max_pool2d(torch.nn.functional.conv2d(x, f2), 2)[0, 0]
    assert out1[0, 1] == 765 and out1[1, 1] == 510                   # upper-right and bottom-right (paper)
    assert out2[1, 0] == 765 and out2[1, 1] == 510                   # bottom-left and bottom-right (paper)


# --------------------------------------------------------------------------- drawing

def reference_chart(close, open_, high, low, volume, days):
    """A slow, independent drawing of bars, ticks and volume (no moving average) for comparison."""
    H, W, PR, VR = cnn.SHAPES[days]
    img = np.zeros((H, W), dtype=np.uint8)
    lo, hi = min(low), max(high)
    row = lambda p: int((PR - 1) - round((p - lo) / (hi - lo) * (PR - 1)))
    vmax = max(volume)
    for d in range(days):
        for r in range(row(high[d]), row(low[d]) + 1):
            img[r, 3 * d + 1] = 255
        img[row(open_[d]), 3 * d] = 255
        img[row(close[d]), 3 * d + 2] = 255
        n = int(round(volume[d] / vmax * VR))
        for k in range(n):
            img[H - 1 - k, 3 * d + 1] = 255
    return img


def toy_inputs(days=5, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, days))
    open_ = close * (1 + rng.normal(0, 0.01, days))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.02, days))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.02, days))
    volume = rng.uniform(1e6, 5e6, days)
    return close, open_, high, low, volume


@pytest.mark.parametrize("days,seed", [(5, 0), (5, 1), (20, 2), (60, 3)])
def test_bars_ticks_and_volume_match_an_independent_drawing(days, seed):
    close, open_, high, low, volume = toy_inputs(days, seed)
    ma = np.full(days, np.nan)
    img = cnn.draw_images(close[None], open_[None], high[None], low[None], ma[None], volume[None], days)[0]
    np.testing.assert_array_equal(img, reference_chart(close, open_, high, low, volume, days))
    H, W, PR, VR = cnn.SHAPES[days]
    assert img[0].any() and img[PR - 1].any()                        # highest high on the top row, lowest low on the bottom row
    assert not img[PR].any()                                         # the empty row between prices and volume
    assert img[H - VR, 1::3].any()                                   # the biggest volume day fills the strip


def test_moving_average_line_is_connected_and_inside_the_scale():
    close, open_, high, low, volume = toy_inputs(20, 4)
    ma = np.convolve(close, np.ones(3) / 3, mode="same")
    img = cnn.draw_images(close[None], open_[None], high[None], low[None], ma[None], volume[None], 20,
                          volume_bars=False)[0]
    no_ma = cnn.draw_images(close[None], open_[None], high[None], low[None], np.full((1, 20), np.nan),
                            volume[None], 20, volume_bars=False)[0]
    line = (img > 0) & ~(no_ma > 0)
    hi, lo = max(high.max(), ma.max()), min(low.min(), ma.min())
    for d in range(20):                                              # every day's dot is drawn
        r = int(round((63) - (ma[d] - lo) / (hi - lo) * 63))
        assert img[r, 3 * d + 1] == 255
    for c in range(1, 3 * 19 + 2):                                   # every column between dots has a line pixel
        assert (img[:, c] > 0).any()
    assert line.any()


def test_missing_high_low_leaves_the_day_blank_and_missing_open_drops_the_tick():
    close, open_, high, low, volume = toy_inputs(5, 5)
    high2, open2 = high.copy(), open_.copy()
    high2[2] = np.nan
    open2[3] = np.nan
    img = cnn.draw_images(close[None], open2[None], high2[None], low[None], np.full((1, 5), np.nan),
                          volume[None], 5, volume_bars=False)[0]
    assert not img[:, 6:9].any()                                     # day 3: no bar, no ticks
    assert not img[:, 9].any() and img[:, 10].any() and img[:, 11].any()   # day 4: bar and close, no open


def test_charts_are_scale_free():
    close, open_, high, low, volume = toy_inputs(20, 6)
    ma = np.full(20, np.nan)
    a = cnn.draw_images(close[None], open_[None], high[None], low[None], ma[None], volume[None], 20)
    b = cnn.draw_images(3 * close[None], 3 * open_[None], 3 * high[None], 3 * low[None], ma[None], 7 * volume[None], 20)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- data in, no look-ahead

def toy_market(n_days=120, n=25, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2006-01-02", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n)]
    r = rng.normal(0.0005, 0.02, (n_days, n))
    adj = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=cols)
    o = pd.DataFrame(1 + rng.normal(0, 0.005, adj.shape), index=dates, columns=cols)
    h = np.maximum(o, 1) * (1 + rng.uniform(0, 0.01, adj.shape))
    lo = np.minimum(o, 1) * (1 - rng.uniform(0, 0.01, adj.shape))
    v = pd.DataFrame(rng.uniform(1e6, 2e6, adj.shape), index=dates, columns=cols)
    return cnn.Market(adj, o, h, lo, v)


def test_chart_inputs_rebuild_the_paper_price_path():
    mkt = toy_market()
    t = mkt.adj_close.index[60]
    x = cnn.chart_inputs(mkt, 20, [t], ["S03"])
    path = mkt.adj_close["S03"].iloc[41:61].to_numpy()
    np.testing.assert_allclose(x["close"][0], path / path[0])        # first close in the chart is 1
    np.testing.assert_allclose(x["high"][0], mkt.high_r["S03"].iloc[41:61].to_numpy() * path / path[0])
    ma = mkt.adj_close["S03"].rolling(20).mean().iloc[41:61].to_numpy() / path[0]
    np.testing.assert_allclose(x["ma"][0], ma)


def test_chart_ignores_the_future_and_label_uses_only_the_next_h_days():
    mkt = toy_market()
    t = mkt.adj_close.index[70]
    a = cnn.make_images(mkt, 20, 5, start=t, end=t)
    later = mkt.adj_close.copy()
    later.iloc[71:] *= np.random.default_rng(1).uniform(0.5, 1.5, later.iloc[71:].shape)
    b = cnn.make_images(cnn.Market(later, mkt.open_r, mkt.high_r, mkt.low_r, mkt.volume), 20, 5, start=t, end=t)
    np.testing.assert_array_equal(a.images, b.images)
    fwd = mkt.adj_close.iloc[75] / mkt.adj_close.iloc[70] - 1
    got = a.meta.set_index("symbol")["fwd_return"]
    np.testing.assert_allclose(got.to_numpy(), fwd[got.index].to_numpy())
    assert (a.meta["label"] == (a.meta["fwd_return"] > 0)).all()


def test_training_window_never_sees_returns_after_its_end():
    mkt = toy_market()
    end = mkt.adj_close.index[90]
    s = cnn.make_images(mkt.window(None, end), 5, 5)
    assert s.meta["date"].max() <= mkt.adj_close.index[85]           # last label ends on `end`


def test_charts_spanning_a_listing_are_dropped():
    mkt = toy_market()
    adj = mkt.adj_close.copy()
    adj.iloc[:50, 0] = np.nan                                        # S00 lists on day 50
    mkt2 = cnn.Market(adj, mkt.open_r, mkt.high_r, mkt.low_r, mkt.volume)
    ok = cnn.eligible(mkt2, 20)["S00"]
    assert not ok.iloc[:50 + 2 * 20 - 2].any() and ok.iloc[50 + 2 * 20 - 2]


# --------------------------------------------------------------------------- portfolios

def test_decile_backtest_matches_a_hand_calculation():
    mkt = toy_market(n_days=60, n=20)
    adj = mkt.adj_close
    scores = pd.DataFrame(np.tile(np.arange(20.0), (60, 1)), index=adj.index, columns=adj.columns)
    bt = cnn.decile_backtest(scores, adj, horizon=5, start=adj.index[10], cost_bps=10)
    first = bt.period.iloc[0]
    t = adj.index.get_loc(first["date"])
    h = adj.iloc[t + 5] / adj.iloc[t] - 1
    longs, shorts = ["S18", "S19"], ["S00", "S01"]                   # 20 stocks: 2 per decile
    assert first["spread"] == pytest.approx(h[longs].mean() - h[shorts].mean())
    assert first["turnover"] == pytest.approx(2.0) and first["cost"] == pytest.approx(2.0 * 10 / 1e4)
    days = adj.index[t + 1:t + 6]
    assert bt.daily_gross.loc[days].sum() == pytest.approx(first["spread"])      # daily P&L adds up
    assert bt.daily_net.loc[days[0]] == pytest.approx(bt.daily_gross.loc[days[0]] - first["cost"])
    gap = np.diff([adj.index.get_loc(d) for d in bt.period["date"]])
    assert (gap == 5).all()                                          # rebalance every 5 days, no overlap
    second = bt.period.iloc[1]                                       # same names again: only the drift is traded
    assert 0 < second["turnover"] < 0.5


def test_benchmark_signs_point_towards_up():
    adj = toy_market(n_days=300).adj_close
    s = cnn.benchmark_signals(adj)
    t = adj.index[-1]
    assert s["WSTR"].loc[t, "S00"] == pytest.approx(-(adj.loc[t, "S00"] / adj["S00"].iloc[-6] - 1))
    assert s["MOM"].loc[t, "S00"] == pytest.approx(adj["S00"].iloc[-22] / adj["S00"].iloc[-253] - 1)


def test_accuracy_counts_calls():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2010-01-04")], ["A", "B", "C", "D"]], names=["date", "symbol"])
    s = pd.Series([0.6, 0.4, 0.7, 0.3], idx)
    r = pd.Series([0.01, 0.02, -0.01, -0.03], idx)
    assert cnn.period_accuracy(s, r).iloc[0] == pytest.approx(0.5)
    assert cnn.period_accuracy(s, r, threshold=None).iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- training and Grad-CAM

def test_training_learns_a_planted_pattern_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    imgs = rng.choice([0, 255], size=(1000, 32, 15), p=[0.9, 0.1]).astype(np.uint8)
    labels = rng.integers(0, 2, 1000)
    for i, y in enumerate(labels):                                   # 'up' pictures have a bright bar near the top
        r = rng.integers(2, 10) if y else rng.integers(18, 26)
        imgs[i, r:r + 4, rng.integers(0, 12):][:, :3] = 255
    t = cnn.train_cnn(imgs, labels, 5, 5, seed=0, lr=1e-3, max_epochs=8, patience=3, device="cpu", verbose=False)
    assert t.history["val_acc"].max() > 0.8
    p = cnn.predict([t], imgs[:50], device="cpu")
    path = cnn.save_trained(t, tmp_path / "I5R5_seed0.pt")
    back = cnn.load_ensemble("I5R5", model_dir=tmp_path)
    np.testing.assert_allclose(cnn.predict(back, imgs[:50], device="cpu"), p, atol=5e-3)    # 16-bit weights


def test_grad_cam_is_a_heatmap_of_the_picture():
    t = cnn.Trained(20, 20, cnn.build_cnn(20, seed=1).eval(), 30.0, 80.0)
    img = np.zeros((64, 60), dtype=np.uint8)
    img[10:40, 1::3] = 255
    for block in (0, 1, 2):
        cam = cnn.grad_cam(t, img, block=block)
        assert cam.shape == (64, 60) and cam.min() >= 0 and cam.max() <= 1


# --------------------------------------------------------------------------- caches and notebook

def test_cached_predictions_are_out_of_sample():
    if not cnn.PRED_CACHE.exists():
        pytest.skip("workshop5/data/predictions.csv.gz is not in this checkout")
    for name in ("I5R5", "I20R20"):
        p = cnn.load_predictions(name)
        dates = p.index.get_level_values("date")
        assert dates.min() >= pd.Timestamp("2008-01-01") and p.between(0, 1).all()
        assert len(cnn.load_ensemble(name)) == 5


def test_cached_ensembles_have_five_final_networks_each():
    if not cnn.HISTORY_CACHE.exists():
        pytest.skip("workshop5/data/training_history.csv is not in this checkout")
    h = pd.read_csv(cnn.HISTORY_CACHE)
    final = h[h["run"] == "final"]
    for name in ("I5R5", "I20R20"):
        assert final[final["model"] == name]["seed"].nunique() == 5
        assert len(list(cnn.MODEL_DIR.glob(f"{name}_seed*.pt"))) == 5


def test_build_cache_training_labels_end_inside_2007():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_cache", Path(cnn.ROOT) / "workshop5" / "build_cache.py")
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)
    mkt = cnn.load_market()
    data = bc.training_images("I20R20", mkt)
    idx = mkt.adj_close.index
    last_label_day = idx[idx.get_indexer(data.meta["date"]) + 20].max()
    assert last_label_day <= pd.Timestamp("2007-12-31") and data.meta["date"].min() >= pd.Timestamp("2000-01-01")


def test_workshop5_notebook_has_no_dashes_and_no_outputs():
    path = Path(cnn.ROOT) / "workshop5" / "workshop5.ipynb"
    if not path.exists():
        pytest.skip("notebook not built yet")
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        assert not re.search("[–—]", text), text[:80]
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


# --------------------------------------------------------------------------- guards added after review

def test_moving_average_dots_and_line_alone():
    close, open_, high, low, volume = toy_inputs(20, 8)
    ma = 100 + np.cumsum(np.random.default_rng(3).normal(0, 2, 20))
    nan = np.full((1, 20), np.nan)
    img = cnn.draw_images(close[None], open_[None], nan, nan, ma[None], volume[None], 20, volume_bars=False)[0]
    hi, lo = ma.max(), ma.min()
    for d in range(20):
        r = int(round(63 - (ma[d] - lo) / (hi - lo) * 63))
        assert img[r, 3 * d + 1] == 255                               # the dot sits in the day's middle column
    on = (img > 0).any(axis=0)
    assert on[1:3 * 19 + 2].all()                                    # connected from the first dot to the last
    assert not on[0] and not on[-1]                                  # nothing before the first dot or after the last
    assert img[0].any() and img[63].any()                            # the line alone sets the scale


def test_open_or_close_outside_the_range_stays_in_the_picture():
    close, open_, high, low, volume = toy_inputs(5, 9)
    close = close.copy(); close[1] = 1.5 * high[1]
    img = cnn.draw_images(close[None], open_[None], high[None], low[None], np.full((1, 5), np.nan), volume[None], 5)[0]
    assert img.shape == (32, 15) and img[0, 5] == 255                 # clipped to the top row


def test_a_missing_high_does_not_squash_the_scale():
    close, open_, high, low, volume = toy_inputs(5, 10)
    high2, low2 = high.copy(), low.copy()
    high2[2] = np.nan; low2[2] = 1.0                                  # a stray low with no high
    img = cnn.draw_images(close[None], open_[None], high2[None], low2[None], np.full((1, 5), np.nan), volume[None], 5)[0]
    ok = [0, 1, 3, 4]
    rows = lambda p: (24 - np.rint((p - low[ok].min()) / (high[ok].max() - low[ok].min()) * 24)).astype(int)
    assert img[rows(low[ok].min()), 1::3].any() and img[0, 1::3].any()


def test_backtest_reads_each_rebalance_days_own_score():
    mkt = toy_market(n_days=80, n=20)
    adj = mkt.adj_close
    rng = np.random.default_rng(2)
    scores = pd.DataFrame(rng.normal(size=adj.shape), index=adj.index, columns=adj.columns)
    a = cnn.decile_backtest(scores, adj, 5, start=adj.index[10])
    other = scores.copy()
    reb = set(a.period["date"])
    other.loc[[d for d in adj.index if d not in reb]] = rng.normal(size=(len(adj.index) - len(reb), adj.shape[1]))
    b = cnn.decile_backtest(other, adj, 5, start=adj.index[10])
    pd.testing.assert_series_equal(a.daily_net, b.daily_net)          # other days' scores are never read
    other.loc[a.period["date"].iloc[3]] *= -1
    c = cnn.decile_backtest(other, adj, 5, start=adj.index[10])
    assert c.period["spread"].iloc[3] == pytest.approx(-a.period["spread"].iloc[3])


def test_backtest_window_and_annualisation():
    mkt = toy_market(n_days=120, n=20)
    adj = mkt.adj_close
    scores = pd.DataFrame(np.random.default_rng(4).normal(size=adj.shape), index=adj.index, columns=adj.columns)
    bt = cnn.decile_backtest(scores, adj, 5, start=adj.index[10])
    w = bt.window(adj.index[40], adj.index[80])
    assert w.period["date"].min() >= adj.index[40] and w.period["date"].max() <= adj.index[80]
    assert w.daily_gross.sum() == pytest.approx(w.period["spread"].sum())
    r = w.period["net"]
    assert w.paper_stats()["Sharpe (after costs)"] == pytest.approx(r.mean() / r.std() * np.sqrt(252 / 5))
    weekly = cnn.decile_backtest(scores, adj, 5, start=adj.index[10], bars_per_year=52)
    assert weekly.paper_stats()["Sharpe (after costs)"] == pytest.approx(
        bt.paper_stats()["Sharpe (after costs)"] * np.sqrt(52 / 252))


def test_training_uses_training_pixels_only_and_keeps_the_best_epoch():
    rng = np.random.default_rng(1)
    imgs = rng.choice([0, 255], size=(400, 32, 15), p=[0.8, 0.2]).astype(np.uint8)
    imgs[:120] = 0                                                    # make the two halves' pixel means differ
    labels = rng.integers(0, 2, 400)
    t = cnn.train_cnn(imgs, labels, 5, 5, seed=3, lr=1e-2, max_epochs=6, patience=6, device="cpu", verbose=False)
    idx = np.random.default_rng(3).permutation(400)
    val, tr = idx[:120], idx[120:]
    assert t.pixel_mean == pytest.approx(imgs[tr].mean()) and t.pixel_mean != pytest.approx(imgs.mean())
    p = cnn.predict([t], imgs[val], device="cpu")
    loss = -np.mean(labels[val] * np.log(p) + (1 - labels[val]) * np.log(1 - p))
    assert loss == pytest.approx(t.history["val_loss"].min(), rel=1e-4)   # the kept weights are the best epoch's


def test_predict_normalises_like_training_and_save_keeps_mean_and_std(tmp_path):
    model = cnn.build_cnn(5, seed=0).eval()
    t = cnn.Trained(5, 5, model, 40.0, 90.0)
    img = np.random.default_rng(0).choice([0, 255], size=(6, 32, 15)).astype(np.uint8)
    x = (torch.tensor(img, dtype=torch.float32)[:, None] - 40.0) / 90.0
    with torch.no_grad():
        manual = torch.softmax(model(x), 1)[:, 1].numpy()
    np.testing.assert_allclose(cnn.predict([t], img, device="cpu"), manual, rtol=1e-5)
    back = cnn.load_trained(cnn.save_trained(t, tmp_path / "m.pt"))
    assert (back.pixel_mean, back.pixel_std) == (40.0, 90.0)
    f = img.astype(np.float32)
    cnn.predict([t], f, device="cpu")
    np.testing.assert_array_equal(f, img.astype(np.float32))         # the caller's array is untouched


def test_reversal_signals_and_scores():
    adj = toy_market(n_days=300).adj_close
    s = cnn.benchmark_signals(adj)
    t = adj.index[-1]
    assert s["STR"].loc[t, "S01"] == pytest.approx(-(adj.loc[t, "S01"] / adj["S01"].iloc[-22] - 1))
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2010-01-04")], list("ABCDEFGHIJKL")], names=["date", "symbol"])
    x = pd.Series(np.arange(12.0), idx)
    up = pd.Series(np.exp(np.arange(12.0)) - 50, idx)                 # rises with x, but not in a straight line
    assert cnn.period_rank_corr(x, up).iloc[0] == pytest.approx(1.0)
    assert cnn.period_rank_corr(-x, up).iloc[0] == pytest.approx(-1.0)
    assert cnn.period_corr(x, up).iloc[0] < 0.9                       # Pearson is not the rank correlation
    probs = pd.Series([0.9] * 3 + [0.1] * 9, idx)
    rets = pd.Series([0.01] * 3 + [0.0] + [-0.01] * 8, idx)           # a return of exactly 0 is not 'up'
    assert cnn.period_accuracy(probs, rets).iloc[0] == pytest.approx(1.0)
    assert cnn.period_accuracy(1 - probs, rets).iloc[0] == pytest.approx(0.0)


def test_grad_cam_depends_on_the_class_asked_for():
    t = cnn.Trained(20, 20, cnn.build_cnn(20, seed=2).eval(), 30.0, 80.0)
    img = np.zeros((64, 60), dtype=np.uint8)
    img[5:30, 1::3] = 255
    img[40:50, 4::6] = 255
    assert not np.allclose(cnn.grad_cam(t, img, 1, target=1), cnn.grad_cam(t, img, 1, target=0))
    assert not t.model.training


def test_resample_market_builds_four_day_bars():
    mkt = toy_market(n_days=42, n=3)
    w = cnn.resample_market(mkt, 4)
    assert len(w.adj_close) == 10 and w.adj_close.index[-1] == mkt.adj_close.index[-1]   # 2 leftover days dropped
    days = mkt.adj_close.index[-4:]
    lvl = lambda r: (r * mkt.adj_close).loc[days, "S00"]
    assert w.adj_close["S00"].iloc[-1] == mkt.adj_close.loc[days[-1], "S00"]
    assert w.open_r["S00"].iloc[-1] * w.adj_close["S00"].iloc[-1] == pytest.approx(lvl(mkt.open_r).iloc[0])
    assert w.high_r["S00"].iloc[-1] * w.adj_close["S00"].iloc[-1] == pytest.approx(lvl(mkt.high_r).max())
    assert w.low_r["S00"].iloc[-1] * w.adj_close["S00"].iloc[-1] == pytest.approx(lvl(mkt.low_r).min())
    assert w.volume["S00"].iloc[-1] == pytest.approx(mkt.volume.loc[days, "S00"].sum())


def test_make_images_every_and_offset_pick_the_dates():
    mkt = toy_market()
    s = cnn.make_images(mkt, 5, 5, start=mkt.adj_close.index[30], every=7, offset=2)
    want = mkt.adj_close.loc[mkt.adj_close.index[30]:].index[2::7]
    assert set(s.meta["date"]) <= set(want) and s.meta["date"].nunique() > 5
