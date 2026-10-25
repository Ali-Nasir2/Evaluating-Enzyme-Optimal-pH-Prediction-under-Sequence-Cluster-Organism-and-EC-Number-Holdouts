"""Regression metrics and evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics for optimal-pH prediction."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            f"Metric inputs must have the same shape, got {y_true_array.shape} and {y_pred_array.shape}."
        )
    if y_true_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set.")

    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "spearman": _series_corr(y_true_array, y_pred_array, method="spearman"),
        "pearson": _series_corr(y_true_array, y_pred_array, method="pearson"),
    }


def _series_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    value = pd.Series(y_true).corr(pd.Series(y_pred), method=method)
    return float(value) if not pd.isna(value) else float("nan")


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """Estimate percentile bootstrap confidence intervals for regression metrics."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("Bootstrap inputs must have the same shape.")
    if y_true_array.size == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    rng = np.random.default_rng(random_seed)
    metric_samples: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "r2": [],
        "spearman": [],
        "pearson": [],
    }
    n = y_true_array.size
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        metrics = regression_metrics(y_true_array[indices], y_pred_array[indices])
        for metric_name, value in metrics.items():
            metric_samples[metric_name].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, float] = {}
    point_metrics = regression_metrics(y_true_array, y_pred_array)
    for metric_name, point_value in point_metrics.items():
        samples = np.asarray(metric_samples[metric_name], dtype=float)
        samples = samples[~np.isnan(samples)]
        intervals[metric_name] = point_value
        if samples.size == 0:
            intervals[f"{metric_name}_ci_lower"] = float("nan")
            intervals[f"{metric_name}_ci_upper"] = float("nan")
        else:
            intervals[f"{metric_name}_ci_lower"] = float(np.quantile(samples, alpha))
            intervals[f"{metric_name}_ci_upper"] = float(np.quantile(samples, 1.0 - alpha))
    return intervals


def ph_range_label(value: float) -> str:
    """Assign an optimal-pH value to a broad pH range."""
    if value < 6.0:
        return "acidic"
    if value <= 8.0:
        return "near_neutral"
    return "alkaline"


def summarize_predictions(
    predictions: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    min_count: int = 1,
) -> pd.DataFrame:
    """Summarize prediction error by arbitrary grouping columns."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        if len(group) < min_count:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = group[target_column].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)
        metrics = regression_metrics(y_true, y_pred)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "n_examples": len(group),
                "mean_residual": float(group["residual"].mean()),
                "bias": float(group["prediction"].mean() - group[target_column].mean()),
                **metrics,
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def paired_absolute_error_comparisons(
    predictions: pd.DataFrame,
    baseline_feature_group: str = "handcrafted",
    comparison_feature_group: str = "esm2_embedding",
    n_bootstrap: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compare absolute errors for paired feature groups on identical test examples."""
    test_predictions = predictions.loc[predictions["evaluation_split"] == "test"].copy()
    rows: list[dict[str, object]] = []
    for (split_strategy, model), group in test_predictions.groupby(["split_strategy", "model"]):
        left = group.loc[group["feature_group"] == baseline_feature_group, ["row_index", "absolute_error"]]
        right = group.loc[group["feature_group"] == comparison_feature_group, ["row_index", "absolute_error"]]
        if left.empty or right.empty:
            continue
        paired = left.merge(right, on="row_index", suffixes=("_baseline", "_comparison"))
        if paired.empty:
            continue
        diff = paired["absolute_error_comparison"] - paired["absolute_error_baseline"]
        diff_values = diff.to_numpy(dtype=float)
        rng = np.random.default_rng(random_seed)
        bootstrap_means = np.asarray(
            [
                rng.choice(diff_values, size=len(diff_values), replace=True).mean()
                for _ in range(n_bootstrap)
            ],
            dtype=float,
        )
        row: dict[str, object] = {
            "split_strategy": split_strategy,
            "model": model,
            "baseline_feature_group": baseline_feature_group,
            "comparison_feature_group": comparison_feature_group,
            "n_pairs": len(paired),
            "baseline_mae": float(paired["absolute_error_baseline"].mean()),
            "comparison_mae": float(paired["absolute_error_comparison"].mean()),
            "comparison_minus_baseline_mae": float(diff.mean()),
            "paired_difference_ci_lower": float(np.quantile(bootstrap_means, 0.025)),
            "paired_difference_ci_upper": float(np.quantile(bootstrap_means, 0.975)),
        }
        try:
            from scipy.stats import wilcoxon

            if np.any(diff.to_numpy(dtype=float) != 0):
                result = wilcoxon(diff.to_numpy(dtype=float), zero_method="wilcox")
                row["wilcoxon_statistic"] = float(result.statistic)
                row["wilcoxon_p_value"] = float(result.pvalue)
            else:
                row["wilcoxon_statistic"] = float("nan")
                row["wilcoxon_p_value"] = 1.0
        except Exception:
            row["wilcoxon_statistic"] = float("nan")
            row["wilcoxon_p_value"] = float("nan")
        rows.append(row)
    result_frame = pd.DataFrame.from_records(rows)
    if result_frame.empty:
        return result_frame

    result_frame["wilcoxon_p_value_holm"] = holm_bonferroni_adjustment(
        result_frame["wilcoxon_p_value"].to_numpy(dtype=float)
    )
    result_frame["holm_significant_alpha_0_05"] = (
        result_frame["wilcoxon_p_value_holm"] < 0.05
    )
    result_frame["multiple_testing_method"] = "Holm-Bonferroni"
    return result_frame


def holm_bonferroni_adjustment(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(~np.isnan(values))
    if valid_indices.size == 0:
        return adjusted

    order = valid_indices[np.argsort(values[valid_indices], kind="stable")]
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        (valid_indices.size - np.arange(valid_indices.size)) * ranked
    )
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted
