"""Run grouped seed evaluation with the documented 300-tree forest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import baseline_models
from src.sequence_features import feature_column_names
from src.splitting import SplitFractions, make_balanced_grouped_split


SEEDS = [42, 43, 44, 45, 46]
OUTPUT_DIR = ROOT / "outputs" / "metrics" / "repeated_grouped_300tree_20261025"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "processed" / "cleaned_dataset.csv")
    handcrafted = pd.read_csv(ROOT / "data" / "processed" / "handcrafted_features.csv")
    embeddings = np.load(ROOT / "data" / "processed" / "embeddings" / "esm2_t6_8M_UR50D_embeddings.npy")
    feature_columns = [
        column for column in feature_column_names(handcrafted.columns)
        if pd.api.types.is_numeric_dtype(handcrafted[column]) and not handcrafted[column].isna().all()
    ]
    fractions = SplitFractions(train=0.70, validation=0.15, test=0.15)
    rows: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    for group_column, strategy in [("Organism", "organism_held_out"), ("EC Number", "ec_held_out")]:
        for seed in SEEDS:
            assignments = make_balanced_grouped_split(data, group_column, fractions, seed)
            assignment_frame = pd.DataFrame(
                {"row_index": data.index, "group": data[group_column].astype(str), "split": assignments}
            )
            assignment_frame.to_csv(OUTPUT_DIR / f"{strategy}_seed_{seed}_assignments.csv", index=False)
            train_indices = np.flatnonzero(assignments.to_numpy() == "train")
            test_indices = np.flatnonzero(assignments.to_numpy() == "test")
            y_train = data.loc[train_indices, "pHopt"].to_numpy(float)
            y_test = data.loc[test_indices, "pHopt"].to_numpy(float)
            for feature_group, matrix in [("handcrafted", handcrafted[feature_columns].to_numpy()), ("esm2_embedding", embeddings)]:
                model = baseline_models(random_seed=seed)["random_forest"]
                model.fit(matrix[train_indices], y_train)
                prediction = model.predict(matrix[test_indices])
                model_mae = float(np.mean(np.abs(y_test - prediction)))
                for baseline, value in [("training_mean", np.mean(y_train)), ("training_median", np.median(y_train))]:
                    baseline_mae = float(np.mean(np.abs(y_test - value)))
                    rows.append(
                        {
                            "strategy": strategy, "group_column": group_column, "seed": seed,
                            "feature_group": feature_group, "model": "random_forest",
                            "n_train": len(train_indices), "n_test": len(test_indices),
                            "model_mae": model_mae, "baseline": baseline,
                            "baseline_mae": baseline_mae, "mae_minus_baseline": model_mae - baseline_mae,
                        }
                    )
            distributions.append(
                {
                    "strategy": strategy, "group_column": group_column, "seed": seed,
                    "n_train": len(train_indices), "n_test": len(test_indices),
                    "train_mean_ph": np.mean(y_train), "train_sd_ph": np.std(y_train, ddof=1),
                    "test_mean_ph": np.mean(y_test), "test_sd_ph": np.std(y_test, ddof=1),
                    "test_acidic_n": int(np.sum(y_test < 6)),
                    "test_near_neutral_n": int(np.sum((y_test >= 6) & (y_test <= 8))),
                    "test_alkaline_n": int(np.sum(y_test > 8)),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "results.csv", index=False)
    pd.DataFrame(distributions).to_csv(OUTPUT_DIR / "target_distributions.csv", index=False)
    (OUTPUT_DIR / "settings.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS, "train_fraction": 0.70, "validation_fraction": 0.15,
                "test_fraction": 0.15, "model": "random_forest",
                "n_estimators": 300, "random_state": "seed per run", "n_jobs": -1,
                "preprocessing": "SimpleImputer(strategy='median')",
                "group_split_function": "make_balanced_grouped_split",
                "features": feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
