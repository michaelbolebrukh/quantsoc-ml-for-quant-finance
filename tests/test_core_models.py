"""Models and scores. Score values are computed by hand in the comments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantsoc_ml import evaluate as E
from quantsoc_ml import models as M
from quantsoc_ml.features import FEATURE_NAMES



def _needs_sklearn():
    pytest.importorskip("sklearn", reason="scikit-learn is not installed in this environment")


def test_score_by_hand():
    y = np.array([0.02, -0.01, 0.03, 0.0])
    p = np.array([0.01, -0.02, -0.01, 0.01])
    s = E.score(y, p)
    # errors 0.01, 0.01, 0.04, -0.01 -> mse 19e-4 / 4; predicting zero: 14e-4 / 4
    assert s["mse"] == pytest.approx(4.75e-4)
    assert s["r2_vs_zero"] == pytest.approx(1 - 4.75 / 3.5)
    # three rows moved; signs right on the first two only
    assert s["hit_rate"] == pytest.approx(2 / 3)
    # ranks p = [3.5, 1, 2, 3.5], y = [3, 1, 4, 2]: Pearson on ranks = 1.5 / sqrt(4.5 * 5)
    assert s["ic"] == pytest.approx(1.5 / np.sqrt(22.5))


def test_zero_model_scores():
    y = np.array([0.02, -0.01, 0.03])
    z = M.ZeroModel().fit(np.zeros((3, 2)), y)
    s = E.score(y, z.predict(np.zeros((3, 2))))
    assert s["r2_vs_zero"] == 0.0
    assert np.isnan(s["ic"]) and np.isnan(s["hit_rate"])


def test_decile_table_rises_for_a_perfect_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    t = E.decile_table(y, y, n=10)
    assert list(t.index) == list(range(1, 11))
    assert (t["rows"] == 100).all()
    assert t["mean_target"].is_monotonic_increasing


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = 0.5 * X[:, 0] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    return df


def test_xy_uses_attrs_when_set():
    df = _frame(10)
    X, y = M.xy(df)
    assert X.shape == (10, len(FEATURE_NAMES))
    df.attrs["features"] = ["ret_5", "ret_1"]
    X2, _ = M.xy(df)
    np.testing.assert_array_equal(X2, df[["ret_5", "ret_1"]].to_numpy())


def test_model_settings():
    _needs_sklearn()
    df = _frame(500)
    X, y = M.xy(df)
    lin = M.fit_linear(X, y)
    assert [type(s).__name__ for _, s in lin.steps] == ["StandardScaler", "Ridge"]
    assert lin.steps[1][1].alpha == 1.0
    rf = M.fit_forest(X, y, n_estimators=10)
    assert (rf.max_depth, rf.min_samples_leaf, rf.n_jobs, rf.random_state) == (6, 50, -1, 0)
    rf2 = M.fit_forest(X, y, n_estimators=10)
    assert M.fit_forest(X, y, n_estimators=2, max_depth=None, min_samples_leaf=1).max_depth is None
    # repeatable: the fitted trees are identical, split for split
    for a, b in zip(rf.estimators_, rf2.estimators_):
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
    # predictions agree to rounding only: with n_jobs=-1 sklearn adds the tree outputs in
    # whatever order the threads finish, so the last bit can differ (measured: 1.1e-16)
    np.testing.assert_allclose(rf.predict(X), rf2.predict(X), rtol=0, atol=1e-12)


def test_noise_experiment_and_importance():
    _needs_sklearn()
    train, test = _frame(2000, 1), _frame(1000, 2)
    t = E.noise_experiment(train, test, n_noise=5, n_estimators=20)
    assert list(t.index) == ["real", "real+noise"]
    assert list(t.columns) == ["train_r2", "test_r2"]
    assert t.loc["real", "test_r2"] > 0.3                   # the planted signal is found
    X, y = M.xy(train)
    rf = M.fit_forest(X, y, n_estimators=20)
    Xt, yt = M.xy(test)
    imp = E.permutation_importance_table(rf, Xt, yt, FEATURE_NAMES, n_repeats=3)
    assert imp.iloc[0]["feature"] == FEATURE_NAMES[0]       # the only real signal ranks first


def test_compare_table_shape():
    y = np.array([0.01, -0.02, 0.03, 0.01])
    t = E.compare_table({"zero": np.zeros(4), "perfect": y}, y)
    assert list(t.columns) == ["mse", "r2_vs_zero", "ic", "hit_rate"]
    assert t.loc["perfect", "r2_vs_zero"] == 1.0


def test_fractional_forest_settings_are_refused_not_truncated():
    import numpy as np
    from quantsoc_ml import models as M
    from quantsoc_ml.portfolio import StrategyConfig
    X, y = np.random.default_rng(0).normal(size=(60, 3)), np.random.default_rng(1).normal(size=60)
    with pytest.raises(ValueError, match="max_depth"):
        M.fit_forest(X, y, n_estimators=5, max_depth=2.7)
    with pytest.raises(ValueError, match="whole number"):
        StrategyConfig(n_long=1.7).validate()
    with pytest.raises(ValueError, match="whole number"):
        StrategyConfig(n_long=True).validate()
