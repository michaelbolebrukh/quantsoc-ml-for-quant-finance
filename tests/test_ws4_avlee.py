"""Workshop 4 (Avellaneda & Lee): the OU fit recovers known parameters, the paper's formulas hold,
the hedge is factor-neutral, the trading rules match eq. 16, and nothing looks ahead."""
import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import avlee as al


def simulate_ou(kappa, m, sigma, n, seed=0, dt=1 / 252):
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    x = np.empty(n)
    x[0] = m
    for i in range(1, n):
        x[i] = m + b * (x[i - 1] - m) + sd * rng.normal()
    return x


def test_ar1_recovers_known_ou_parameters():
    kappa, m, sigma = 40.0, 0.02, 0.3
    x = simulate_ou(kappa, m, sigma, 20000)
    resid = np.diff(np.concatenate([[0.0], x]))[:, None]   # cumulative sum of these is x again
    out = al._ar1(resid)
    assert out["kappa"][0] == pytest.approx(kappa, rel=0.15)
    assert out["m"][0] == pytest.approx(m, abs=0.01)
    assert out["sigma_eq"][0] == pytest.approx(sigma / np.sqrt(2 * kappa), rel=0.05)


def test_s_score_matches_equation_22():
    rng = np.random.default_rng(1)
    Rs = rng.normal(0, 0.01, (60, 8))
    F = rng.normal(0, 0.01, (60, 2))
    cfg = al.Config()
    f = al.ou_fit(Rs, F, cfg)
    s, elig = al.s_scores(f, cfg)
    a, b = f["a"], f["b"]
    Xc = np.cumsum(Rs - np.column_stack([np.ones(60), F]) @ np.linalg.lstsq(np.column_stack([np.ones(60), F]), Rs, rcond=None)[0], axis=0)
    var_z = (Xc[1:] - (a + b * Xc[:-1])).var(axis=0, ddof=1)
    mbar = np.nanmean((a / (1 - b))[elig])
    eq22 = -a * np.sqrt(1 - b ** 2) / ((1 - b) * np.sqrt(var_z)) + mbar * np.sqrt((1 - b ** 2) / var_z)
    ok = np.isfinite(s)
    np.testing.assert_allclose(s[ok], eq22[ok], rtol=1e-6, atol=1e-6)
    assert abs(Xc[-1]).max() < 1e-12                                   # X_60 = 0, as the paper notes


def test_eigenportfolio_returns_are_uncorrelated_and_variance_rule_counts():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, (300, 3))
    R = common @ rng.normal(1, 0.5, (3, 40)) + rng.normal(0, 0.01, (300, 40))
    Q, vals, m = al.pca_eigenportfolios(R, var_explained=0.55)
    assert vals.sum() == pytest.approx(40) and m == int(np.searchsorted(np.cumsum(vals) / 40, 0.55) + 1)
    F = R @ Q
    c = np.corrcoef(F.T)
    assert np.abs(c - np.eye(m)).max() < 1e-8
    Q15, _, m15 = al.pca_eigenportfolios(R, n_factors=15)
    assert m15 == 15 and Q15.shape == (40, 15)


def _toy_prices(n_days=400, n=30, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mkt = rng.normal(0, 0.01, n_days)
    r = mkt[:, None] * rng.uniform(0.5, 1.5, n) + rng.normal(0, 0.01, (n_days, n))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=[f"S{i:02d}" for i in range(n)])


def test_pca_book_has_zero_factor_exposure_in_window():
    P = _toy_prices()
    cfg = al.Config(corr_window=252)
    R = al.returns(P)
    t = 300
    fit = al.estimate_day(R, t, cfg)
    states = {fit.symbols[i]: (1 if i % 3 == 0 else -1) for i in range(0, 12)}
    w = al.book_weights(fit, states, cfg)
    win = R.iloc[t + 1 - cfg.window: t + 1][fit.symbols].to_numpy()
    F = win @ fit.Q
    port = win @ w.reindex(fit.symbols).to_numpy()
    X = np.column_stack([np.ones(len(F)), F])
    beta = np.linalg.lstsq(X, port, rcond=None)[0][1:]
    assert np.abs(beta).max() < 1e-8


def test_etf_hedge_is_minus_beta_in_the_sector_fund():
    P = _toy_prices(n=12)                     # the estimator needs at least 10 stocks
    etf = pd.DataFrame({"SPY": P.mean(axis=1), "XLK": P.iloc[:, :3].mean(axis=1)}, index=P.index)
    sectors = pd.Series({f"S{i:02d}": ("XLK" if i < 3 else "XLF") for i in range(12)})
    cfg = al.Config(factors="etf")
    fit = al.estimate_day(al.returns(P), 200, cfg, al.returns(etf), sectors)
    assert list(fit.hedge_etf[:3]) == ["XLK"] * 3 and list(fit.hedge_etf[3:]) == ["SPY"] * 9   # XLF missing: market
    w = al.book_weights(fit, {"S00": 1, "S04": -1}, cfg)
    assert w["S00"] == pytest.approx(cfg.position) and w["S04"] == pytest.approx(-cfg.position)
    assert w["XLK"] == pytest.approx(-cfg.position * fit.beta[0, 0])
    assert w["SPY"] == pytest.approx(cfg.position * fit.beta[4, 0])


@pytest.mark.parametrize("state,s,eligible,expected", [
    (0, -1.3, True, 1), (0, 1.3, True, -1), (0, 1.2, True, 0), (0, -1.3, False, 0),
    (1, -0.6, True, 1), (1, -0.4, True, 0), (1, -0.4, False, 0), (1, -0.6, False, 1),
    (-1, 0.8, True, -1), (-1, 0.7, True, 0), (-1, float("nan"), True, 0), (1, 5.0, True, 0)])
def test_trading_rules_eq16(state, s, eligible, expected):
    assert al.next_state(state, s, eligible, al.Config()) == expected


def test_signal_ignores_future_prices():
    P = _toy_prices()
    cfg = al.Config()
    t = 320
    a = al.estimate_day(al.returns(P), t, cfg)
    Q = P.copy()
    Q.iloc[t + 1:] *= np.random.default_rng(9).uniform(0.5, 1.5, Q.iloc[t + 1:].shape)
    b = al.estimate_day(al.returns(Q), t, cfg)
    np.testing.assert_allclose(a.s, b.s)


def test_backtest_pnl_uses_next_day_returns():
    P = _toy_prices(n_days=360)
    cfg = al.Config(corr_window=252)
    bt = al.run(P, cfg, keep_weights=True)
    R = al.returns(P)
    d = list(bt.weights)[20]
    nxt = R.index[R.index.get_loc(d) + 1]
    w = bt.weights[d]
    assert bt.gross[nxt] == pytest.approx(float((w * R.loc[nxt].reindex(w.index)).sum()))


def test_cache_has_every_variant_over_the_same_dates():
    import json
    from pathlib import Path
    days = None
    for v in ("etf", "etf_trading_time", "pca_55", "pca_15"):
        bt = al.load_cached(v)
        idx = bt.net.index
        assert idx.min() >= pd.Timestamp("2001-01-01") and idx.is_monotonic_increasing and idx.is_unique
        days = idx if days is None else days
        assert idx.equals(days)
        assert {"s_entry", "kappa", "sigma_eq", "r2", "pnl"} <= set(bt.trades.columns)


def test_workshop4_notebook_has_no_dashes_and_no_outputs():
    import json
    import re
    from pathlib import Path
    nb = json.loads((Path(al.ROOT) / "workshop4" / "workshop4.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        assert not re.search("[–—]", text), text[:80]
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
