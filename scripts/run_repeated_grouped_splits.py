"""Run five-seed grouped-split variability checks with baseline-relative MAE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import baseline_models
from src.splitting import SplitFractions, make_balanced_grouped_split


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "processed" / "cleaned_dataset.csv")
    handcrafted = pd.read_csv(ROOT / "data" / "processed" / "handcrafted_features.csv")
    embeddings = np.load(ROOT / "data" / "processed" / "embeddings" / "esm2_t6_8M_UR50D_embeddings.npy")
    fractions = SplitFractions(train=0.70, validation=0.15, test=0.15)
    rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for group_column, strategy in [("Organism", "organism_held_out"), ("EC Number", "ec_held_out")]:
        for seed in SEEDS:
            assignments = make_balanced_grouped_split(data, group_column, fractions, seed)
            test_indices = np.flatnonzero(assignments.to_numpy() == "test")
            train_indices = np.flatnonzero(assignments.to_numpy() == "train")
            y_train = data.loc[train_indices, "pHopt"].to_numpy(float)
            y_test = data.loc[test_indices, "pHopt"].to_numpy(float)
            for feature_group, X in [
                ("handcrafted", handcrafted.drop(columns=["Accession", "pHopt", "Organism", "EC Number", "Split"], errors="ignore").to_numpy()),
                ("esm2_embedding", embeddings),
            ]:
                model = baseline_models(random_seed=seed)["random_forest"]
                # Keep this variability screen bounded; publication models remain 300-tree forests.
                model.set_params(model__n_estimators=20)
                model.fit(X[train_indices], y_train)
                prediction = model.predict(X[test_indices])
                for baseline_name, baseline_value in [
                    ("training_mean", np.mean(y_train)),
                    ("training_median", np.median(y_train)),
                ]:
                    model_mae = float(np.mean(np.abs(y_test - prediction)))
                    baseline_mae = float(np.mean(np.abs(y_test - baseline_value)))
                    rows.append(
                        {
                            "strategy": strategy,
                            "group_column": group_column,
                            "seed": seed,
                            "feature_group": feature_group,
                            "model": "random_forest",
                            "model_mae": model_mae,
                            "baseline": baseline_name,
                            "baseline_mae": baseline_mae,
                            "mae_minus_baseline": model_mae - baseline_mae,
                        }
                    )
            distribution_rows.append(
                {
                    "strategy": strategy,
                    "group_column": group_column,
                    "seed": seed,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "train_mean_ph": np.mean(y_train),
                    "train_sd_ph": np.std(y_train, ddof=1),
                    "test_mean_ph": np.mean(y_test),
                    "test_sd_ph": np.std(y_test, ddof=1),
                    "test_acidic_n": int(np.sum(y_test < 6)),
                    "test_near_neutral_n": int(np.sum((y_test >= 6) & (y_test <= 8))),
                    "test_alkaline_n": int(np.sum(y_test > 8)),
                }
            )

    output_dir = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(output_dir / "repeated_grouped_split_results.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(
        output_dir / "repeated_grouped_split_target_distributions.csv", index=False
    )


if __name__ == "__main__":
    main()
