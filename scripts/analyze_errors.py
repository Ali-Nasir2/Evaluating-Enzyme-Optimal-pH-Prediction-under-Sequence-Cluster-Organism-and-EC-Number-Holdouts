"""Generate focused error-analysis tables from combined predictions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation import ph_range_label, summarize_predictions
from src.explainability import largest_errors, mark_group_frequency
from src.utils import configure_logging, ensure_directory, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--rare-threshold", type=int, default=5, help="Maximum group count considered rare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    paths = config["paths"]
    dataset_config = config["dataset"]
    target_column = dataset_config["target_column"]
    organism_column = dataset_config.get("organism_column")
    ec_column = dataset_config.get("ec_column")

    predictions_path = Path(paths["outputs_dir"]) / "predictions" / "combined_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Combined predictions not found: {predictions_path}. Run scripts/evaluate_models.py first."
        )
    predictions = pd.read_csv(predictions_path, low_memory=False)
    test_predictions = predictions.loc[predictions["evaluation_split"] == "test"].copy()

    metrics_dir = ensure_directory(Path(paths["outputs_dir"]) / "metrics")
    predictions_dir = ensure_directory(Path(paths["outputs_dir"]) / "predictions")

    largest_errors(test_predictions, n=50).to_csv(predictions_dir / "top_50_errors_overall.csv", index=False)
    best_embedding = select_best_embedding_predictions(test_predictions)
    largest_errors(best_embedding, n=50).to_csv(
        predictions_dir / "top_50_errors_best_embedding_models.csv", index=False
    )

    extreme = best_embedding.copy()
    extreme["ph_range"] = extreme[target_column].map(ph_range_label)
    extreme.loc[extreme[target_column] <= 5.0].to_csv(predictions_dir / "low_ph_errors.csv", index=False)
    extreme.loc[extreme[target_column] >= 9.0].to_csv(predictions_dir / "high_ph_errors.csv", index=False)

    length_summary = summarize_by_bins(
        best_embedding,
        value_column="sequence_length",
        bins=[0, 100, 250, 500, 750, 1100],
        labels=["<=100", "101-250", "251-500", "501-750", ">750"],
        target_column=target_column,
    )
    length_summary.to_csv(metrics_dir / "error_by_sequence_length.csv", index=False)

    if organism_column and organism_column in best_embedding.columns:
        organism_marked = mark_group_frequency(best_embedding, organism_column, args.rare_threshold)
        organism_marked.to_csv(predictions_dir / "best_embedding_predictions_with_organism_frequency.csv", index=False)
        organism_frequency = summarize_predictions(
            organism_marked,
            group_columns=["split_strategy", "model", f"{organism_column}_frequency"],
            target_column=target_column,
        )
        organism_frequency.to_csv(metrics_dir / "error_by_organism_frequency.csv", index=False)

    if ec_column and ec_column in best_embedding.columns:
        ec_marked = mark_group_frequency(best_embedding, ec_column, args.rare_threshold)
        ec_frequency = summarize_predictions(
            ec_marked,
            group_columns=["split_strategy", "model", f"{ec_column}_frequency"],
            target_column=target_column,
        )
        ec_frequency.to_csv(metrics_dir / "error_by_ec_frequency.csv", index=False)

    LOGGER.info("Wrote focused error-analysis tables to %s and %s", metrics_dir, predictions_dir)


def select_best_embedding_predictions(test_predictions: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-MAE non-mean ESM-2 model for each split strategy."""
    candidates = test_predictions.loc[
        (test_predictions["feature_group"] == "esm2_embedding")
        & (test_predictions["model"] != "mean_predictor")
    ].copy()
    mae = (
        candidates.groupby(["split_strategy", "model"])["absolute_error"]
        .mean()
        .reset_index(name="mae")
        .sort_values(["split_strategy", "mae"])
    )
    best = mae.groupby("split_strategy", as_index=False).head(1)
    selected = candidates.merge(best[["split_strategy", "model"]], on=["split_strategy", "model"], how="inner")
    return selected


def summarize_by_bins(
    predictions: pd.DataFrame,
    value_column: str,
    bins: list[float],
    labels: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Summarize errors after binning a numeric prediction column."""
    if value_column not in predictions.columns:
        if value_column == "sequence_length":
            cleaned = pd.read_csv("data/processed/cleaned_dataset.csv")
            predictions = predictions.merge(
                cleaned.reset_index().rename(columns={"index": "row_index"})[["row_index", "Sequence"]],
                on="row_index",
                how="left",
            )
            predictions[value_column] = predictions["Sequence"].astype(str).map(len)
        else:
            raise ValueError(f"Missing value column: {value_column}")
    working = predictions.copy()
    working[f"{value_column}_bin"] = pd.cut(
        working[value_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(str)
    return summarize_predictions(
        working,
        group_columns=["split_strategy", "model", f"{value_column}_bin"],
        target_column=target_column,
    )


if __name__ == "__main__":
    main()
