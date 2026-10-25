"""Error-analysis helpers."""

from __future__ import annotations

import pandas as pd


def mark_group_frequency(
    predictions: pd.DataFrame,
    group_column: str,
    rare_threshold: int = 5,
) -> pd.DataFrame:
    """Add group count and rare/common labels for a metadata column."""
    if group_column not in predictions.columns:
        raise ValueError(f"Missing group column: {group_column}")
    output = predictions.copy()
    counts = output[group_column].astype(str).value_counts()
    output[f"{group_column}_count"] = output[group_column].astype(str).map(counts)
    output[f"{group_column}_frequency"] = output[f"{group_column}_count"].map(
        lambda count: "rare" if int(count) <= rare_threshold else "common"
    )
    return output


def largest_errors(
    predictions: pd.DataFrame,
    n: int = 50,
    feature_group: str | None = None,
    split_strategy: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Return the largest absolute prediction errors after optional filtering."""
    filtered = predictions.copy()
    if feature_group is not None:
        filtered = filtered.loc[filtered["feature_group"] == feature_group]
    if split_strategy is not None:
        filtered = filtered.loc[filtered["split_strategy"] == split_strategy]
    if model is not None:
        filtered = filtered.loc[filtered["model"] == model]
    return filtered.sort_values("absolute_error", ascending=False).head(n).reset_index(drop=True)

