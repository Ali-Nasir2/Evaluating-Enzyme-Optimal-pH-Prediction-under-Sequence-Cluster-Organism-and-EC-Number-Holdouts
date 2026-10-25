from __future__ import annotations

import numpy as np

from src.models import MeanRegressor, baseline_models


def test_mean_regressor_predicts_training_mean():
    model = MeanRegressor().fit(np.zeros((3, 2)), np.array([5.0, 7.0, 9.0]))

    predictions = model.predict(np.zeros((2, 2)))

    assert predictions.tolist() == [7.0, 7.0]


def test_baseline_models_include_mandatory_mean_predictor():
    models = baseline_models(random_seed=42)

    assert "mean_predictor" in models
    assert "ridge_regression" in models
    assert "random_forest" in models
    assert "hist_gradient_boosting" in models

