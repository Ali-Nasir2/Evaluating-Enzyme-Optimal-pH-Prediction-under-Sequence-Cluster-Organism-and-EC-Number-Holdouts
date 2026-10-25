from __future__ import annotations

import pandas as pd
import pytest

from scripts.train_baselines import build_generalization_comparison, load_split_definitions


def test_load_split_definitions_merges_generated_split_file(tmp_path):
    features = pd.DataFrame(
        {
            "Accession": ["p1", "p2", "p3"],
            "pHopt": [7.0, 8.0, 6.0],
            "Split": ["Training", "Validation", "Testing"],
            "length": [100.0, 110.0, 120.0],
        }
    )
    split_file = tmp_path / "random_split.csv"
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "split": ["train", "validation", "test"],
            "split_strategy": ["random", "random", "random"],
        }
    ).to_csv(split_file, index=False)

    definitions = load_split_definitions(
        feature_frame=features,
        dataset_split_column="Split",
        splits_dir=tmp_path,
        include_dataset_split=True,
    )

    assert set(definitions) == {"dataset_provided", "random"}
    assert definitions["random"]["_strategy_split"].tolist() == ["train", "validation", "test"]
    assert definitions["dataset_provided"]["_strategy_split"].tolist() == [
        "Training",
        "Validation",
        "Testing",
    ]


def test_build_generalization_comparison_adds_mae_drop_columns():
    metrics = pd.DataFrame(
        {
            "split_strategy": ["random", "exact_duplicate_safe", "cluster_safe_mmseqs_identity_90"],
            "model": ["ridge", "ridge", "ridge"],
            "split": ["test", "test", "test"],
            "mae": [0.5, 0.6, 0.8],
            "rmse": [0.7, 0.8, 1.0],
            "r2": [0.4, 0.3, 0.2],
            "spearman": [0.5, 0.4, 0.3],
            "pearson": [0.6, 0.5, 0.4],
        }
    )

    comparison = build_generalization_comparison(metrics)

    assert comparison.loc[0, "exact_duplicate_safe_mae_minus_random_mae"] == pytest.approx(0.1)
    assert comparison.loc[0, "cluster_safe_mmseqs_identity_90_mae_minus_random_mae"] == pytest.approx(0.3)
