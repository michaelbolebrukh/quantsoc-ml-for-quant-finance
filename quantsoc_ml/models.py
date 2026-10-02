"""The three models of the workshop: predict zero, a linear model, a random forest.

Predicting zero is the honest baseline for daily stock returns: the average 5-day return is
tiny compared with its spread, so 'nothing happens' is hard to beat on squared error. Any model
is judged against it (see evaluate.r2_vs_zero). sklearn is imported inside the functions so
`import quantsoc_ml.models` stays quick.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, TARGET


class ZeroModel:
    """Always forecasts a 0 return. Has fit/predict so it slots in wherever a model does."""

    def fit(self, X, y=None) -> "ZeroModel":
        return self

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


def feature_list(df: pd.DataFrame) -> list:
    """The feature columns a dataset was built with (`.attrs["features"]`), else FEATURE_NAMES."""
    return list(df.attrs.get("features") or FEATURE_NAMES)


def xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(X, y) as float arrays: the feature columns in their fixed order, and the target."""
    return df[feature_list(df)].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


def fit_linear(X, y):
    """StandardScaler then Ridge(alpha=1.0).

    The scaler puts features in comparable units (learned on the training data only, because it
    is fitted here); the small ridge penalty keeps coefficients stable when features overlap,
    as ret_1, ret_5 and ret_21 do.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def _whole(name: str, value) -> int:
    """A whole number of at least 1; 2.7 is refused rather than silently cut to 2."""
    if isinstance(value, bool) or not float(value) == int(float(value)) or int(float(value)) < 1:
        raise ValueError(f"{name} must be a whole number of at least 1, not {value!r}")
    return int(float(value))


def fit_forest(X, y, n_estimators: int = 200, max_depth: Optional[int] = 6, random_state: int = 0,
               min_samples_leaf: int = 50, n_jobs: int = -1):
    """RandomForestRegressor: many shallow trees on bootstrap samples, averaged.

    max_depth and min_samples_leaf are the brakes: a leaf must hold at least 50 training rows,
    so a tree cannot memorise single noisy days. max_depth=None means no depth limit (for the
    overfitting demo). random_state fixes the fitted trees run to run.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=_whole("n_estimators", n_estimators),
                                  max_depth=None if max_depth is None else _whole("max_depth", max_depth),
                                  min_samples_leaf=_whole("min_samples_leaf", min_samples_leaf), n_jobs=n_jobs,
                                  random_state=random_state)
    return model.fit(X, y)


def predict_panel(model, df: pd.DataFrame) -> pd.DataFrame:
    """Forecasts for every (date, symbol) row of `df`, as a date x symbol table for the backtest."""
    X, _ = xy(df) if TARGET in df.columns else (df[feature_list(df)].to_numpy(dtype=float), None)
    pred = pd.Series(model.predict(X), index=df.index)
    wide = pred.unstack("symbol").sort_index()
    wide.columns.name = None
    return wide
