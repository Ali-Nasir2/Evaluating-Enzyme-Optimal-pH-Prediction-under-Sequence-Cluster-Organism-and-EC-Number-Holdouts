"""Baseline model definitions and constructors."""

from __future__ import annotations

import numpy as np
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MeanRegressor(BaseEstimator, RegressorMixin):
    """Regressor that always predicts the training-set mean target."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        if len(y) == 0:
            raise ValueError("Cannot fit MeanRegressor on an empty target array.")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise ValueError("MeanRegressor must be fitted before prediction.")
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


class MedianRegressor(BaseEstimator, RegressorMixin):
    """Regressor that always predicts the training-set median target."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MedianRegressor":
        if len(y) == 0:
            raise ValueError("Cannot fit MedianRegressor on an empty target array.")
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise ValueError("MedianRegressor must be fitted before prediction.")
        return np.full(shape=(len(X),), fill_value=self.median_, dtype=float)


def baseline_models(
    random_seed: int = 42, model_config: dict[str, dict[str, Any]] | None = None
) -> dict[str, BaseEstimator]:
    """Return the Phase 3 baseline models."""
    model_config = model_config or {}
    ridge_config = model_config.get("ridge", {})
    forest_config = model_config.get("random_forest", {})
    boosting_config = model_config.get("histogram_gradient_boosting", {})

    return {
        "mean_predictor": MeanRegressor(),
        "median_predictor": MedianRegressor(),
        "ridge_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(ridge_config.get("alpha", 1.0)))),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(forest_config.get("n_estimators", 300)),
                        max_depth=forest_config.get("max_depth"),
                        random_state=random_seed,
                        n_jobs=int(forest_config.get("n_jobs", -1)),
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=int(boosting_config.get("max_iter", 300)),
                        learning_rate=float(boosting_config.get("learning_rate", 0.05)),
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }
