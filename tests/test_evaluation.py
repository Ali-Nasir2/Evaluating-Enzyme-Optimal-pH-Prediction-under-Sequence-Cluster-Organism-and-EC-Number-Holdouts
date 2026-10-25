from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.evaluation import (
    bootstrap_metric_intervals,
    holm_bonferroni_adjustment,
    paired_absolute_error_comparisons,
    ph_range_label,
    regression_metrics,
    summarize_predictions,
)


def test_regression_metrics_perfect_prediction():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_regression_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="empty evaluation set"):
        regression_metrics([], [])


def test_ph_range_label_boundaries():
    assert ph_range_label(5.9) == "acidic"
    assert ph_range_label(6.0) == "near_neutral"
    assert ph_range_label(8.0) == "near_neutral"
    assert ph_range_label(8.1) == "alkaline"


def test_bootstrap_metric_intervals_include_bounds():
    intervals = bootstrap_metric_intervals(
        [1.0, 2.0, 3.0],
        [1.1, 1.9, 3.2],
        n_bootstrap=10,
        random_seed=1,
    )

    assert "mae_ci_lower" in intervals
    assert "rmse_ci_upper" in intervals
    assert intervals["mae"] >= 0.0


def test_summarize_predictions_groups_and_computes_bias():
    predictions = pd.DataFrame(
        {
            "feature_group": ["a", "a", "a"],
            "split_strategy": ["random", "random", "random"],
            "model": ["ridge", "ridge", "ridge"],
            "pHopt": [7.0, 8.0, 9.0],
            "prediction": [7.5, 7.5, 9.5],
            "residual": [0.5, -0.5, 0.5],
        }
    )

    summary = summarize_predictions(
        predictions,
        group_columns=["feature_group", "split_strategy", "model"],
        target_column="pHopt",
    )

    assert summary.loc[0, "n_examples"] == 3
    assert summary.loc[0, "bias"] == pytest.approx(1 / 6)


def test_paired_absolute_error_comparisons_matches_by_row_index():
    predictions = pd.DataFrame(
        {
            "row_index": [0, 1, 0, 1],
            "feature_group": ["handcrafted", "handcrafted", "esm2_embedding", "esm2_embedding"],
            "split_strategy": ["random"] * 4,
            "model": ["ridge"] * 4,
            "evaluation_split": ["test"] * 4,
            "absolute_error": [0.5, 1.0, 0.2, 0.6],
        }
    )

    comparison = paired_absolute_error_comparisons(predictions)

    assert comparison.loc[0, "n_pairs"] == 2
    assert comparison.loc[0, "comparison_minus_baseline_mae"] == pytest.approx(-0.35)


def test_holm_bonferroni_adjustment_preserves_order_and_nan():
    adjusted = holm_bonferroni_adjustment(np.array([0.04, np.nan, 0.001, 0.2]))

    assert adjusted[0] == pytest.approx(0.08)
    assert np.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.003)
    assert adjusted[3] == pytest.approx(0.2)
