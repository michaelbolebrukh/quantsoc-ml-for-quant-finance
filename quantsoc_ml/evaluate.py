"""How good is a forecast? Scores that a beginner can read, all measured against predicting zero.

- mse         : mean squared error, the thing the models minimise
- r2_vs_zero  : 1 - mse / mse_of_predicting_zero. Above 0 means better than 'nothing happens'.
                In daily equity returns a good honest value is a fraction of a percent.
- ic          : information coefficient, the Spearman rank correlation of forecast and outcome.
                It asks only 'are higher forecasts followed by higher returns?', which is what a
                long/short portfolio needs.
- hit_rate    : share of rows where the forecast has the right sign (rows with a 0 outcome are skipped).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")      # a constant forecast (the zero model) has no ranking
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except ImportError:
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def score(y_true, y_pred) -> Dict[str, float]:
    """dict(mse, r2_vs_zero, ic, hit_rate) for one set of forecasts.

    hit_rate counts a 0 forecast as a miss; it is NaN when every forecast is 0, because a model
    that never picks a direction has no hit rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    mse = float(np.mean((y_true - y_pred) ** 2))
    mse_zero = float(np.mean(y_true ** 2))
    moved = y_true != 0
    if not moved.any() or np.all(y_pred[moved] == 0):
        hit = float("nan")
    else:
        hit = float(np.mean(np.sign(y_pred[moved]) == np.sign(y_true[moved])))
    return {"mse": mse, "r2_vs_zero": 1 - mse / mse_zero if mse_zero > 0 else float("nan"),
            "ic": _spearman(y_pred, y_true), "hit_rate": hit}


def compare_table(predictions: Mapping[str, np.ndarray], y_true) -> pd.DataFrame:
    """One row per model name, columns mse / r2_vs_zero / ic / hit_rate."""
    out = pd.DataFrame({name: score(y_true, p) for name, p in predictions.items()}).T
    out.index.name = "model"
    return out


def decile_table(y_pred, y_true, n: int = 10) -> pd.DataFrame:
    """Sort rows by forecast into n equal-count groups; mean realised target per group.

    If the model has an edge, the mean target rises from decile 1 (lowest forecasts) to
    decile n. Ties are split by order of appearance so every group has the same size.
    """
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    decile = pd.qcut(y_pred.rank(method="first"), n, labels=False) + 1
    g = pd.DataFrame({"decile": decile, "forecast": y_pred, "target": y_true}).groupby("decile")
    out = pd.DataFrame({"mean_forecast": g["forecast"].mean(), "mean_target": g["target"].mean(),
                        "target_se": g["target"].std() / np.sqrt(g.size()), "rows": g.size()})
    return out


def ic_by_date(y_pred, y_true, dates) -> pd.Series:
    """Cross-sectional IC: the Spearman correlation across symbols on each date separately.

    This is the version a long/short book actually trades on (it ranks stocks against each
    other on the same day), so it is a useful second look beside the pooled ic in `score`.
    """
    df = pd.DataFrame({"p": np.asarray(y_pred, float), "y": np.asarray(y_true, float), "date": np.asarray(dates)})
    p = df["p"].groupby(df["date"]).rank()
    y = df["y"].groupby(df["date"]).rank()
    p = p - p.groupby(df["date"]).transform("mean")      # Pearson on ranks = Spearman
    y = y - y.groupby(df["date"]).transform("mean")
    num = (p * y).groupby(df["date"]).sum()
    den = np.sqrt((p * p).groupby(df["date"]).sum() * (y * y).groupby(df["date"]).sum())
    out = (num / den.replace(0, np.nan)).rename("ic")
    out.index.name = "date"
    return out


def noise_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, n_noise: int = 30, seed: int = 0,
                     **forest_kwargs) -> pd.DataFrame:
    """Fit the forest on the real features, then on real + n_noise columns of pure random noise.

    Same forest settings both times (fit_forest defaults unless overridden). Returns
    train_r2 and test_r2 (both r2_vs_zero) for 'real' and 'real+noise'. Noise cannot predict
    anything, so any rise in train_r2 from adding it is memorisation, and the test column
    shows what that memorisation is worth.
    """
    from .models import fit_forest, xy

    Xtr, ytr = xy(train_df)
    Xte, yte = xy(test_df)
    rng = np.random.default_rng(seed)
    noisy_tr = np.hstack([Xtr, rng.standard_normal((len(Xtr), n_noise))])
    noisy_te = np.hstack([Xte, rng.standard_normal((len(Xte), n_noise))])
    rows = {}
    for name, (a, b) in {"real": (Xtr, Xte), "real+noise": (noisy_tr, noisy_te)}.items():
        m = fit_forest(a, ytr, **forest_kwargs)
        rows[name] = {"train_r2": score(ytr, m.predict(a))["r2_vs_zero"],
                      "test_r2": score(yte, m.predict(b))["r2_vs_zero"]}
    out = pd.DataFrame(rows).T
    out.index.name = "features"
    return out


def permutation_importance_table(model, X, y, feature_names, n_repeats: int = 5, seed: int = 0,
                                 n_jobs: Optional[int] = None) -> pd.DataFrame:
    """How much worse the forecast gets (rise in MSE) when one feature column is shuffled.

    Shuffling breaks the link between that feature and the target while keeping its values,
    so a feature the model relies on shows a clear rise. Measured on whatever X you pass;
    use the TEST rows, because importance on training rows also rewards memorisation.
    """
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed,
                               scoring="neg_mean_squared_error", n_jobs=n_jobs)
    out = pd.DataFrame({"feature": list(feature_names), "mse_increase": r.importances_mean,
                        "mse_increase_std": r.importances_std})
    return out.sort_values("mse_increase", ascending=False).reset_index(drop=True)
