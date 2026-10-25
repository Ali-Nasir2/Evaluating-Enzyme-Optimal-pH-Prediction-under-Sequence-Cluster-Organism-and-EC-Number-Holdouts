"""Train baseline regressors on handcrafted features across split strategies."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.models import baseline_models
from src.sequence_features import feature_column_names
from src.utils import configure_logging, ensure_directory, load_config, set_reproducible_seed

LOGGER = logging.getLogger(__name__)


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--include-dataset-split",
        action="store_true",
        help="Also train on the dataset-provided Split column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    seed = int(config.get("project", {}).get("random_seed", 42))
    set_reproducible_seed(seed)

    dataset_config = config["dataset"]
    target_column = dataset_config["target_column"]
    split_column = dataset_config.get("split_column", "Split")
    features_path = Path(config["paths"]["handcrafted_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"Handcrafted features not found: {features_path}. Run scripts/generate_features.py first."
        )

    frame = pd.read_csv(features_path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column missing from feature table: {target_column}")

    candidate_features = feature_column_names(frame.columns)
    numeric_features = [
        column
        for column in candidate_features
        if pd.api.types.is_numeric_dtype(frame[column]) and not frame[column].isna().all()
    ]
    if not numeric_features:
        raise ValueError("No numeric feature columns are available for baseline training.")

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    model_dir = ensure_directory(Path(config["paths"]["outputs_dir"]) / "models" / "baselines")
    metrics_dir = ensure_directory(Path(config["paths"]["outputs_dir"]) / "metrics")
    predictions_dir = ensure_directory(Path(config["paths"]["outputs_dir"]) / "predictions")

    split_definitions = load_split_definitions(
        feature_frame=frame,
        dataset_split_column=split_column,
        splits_dir=Path(config["paths"]["splits_dir"]),
        include_dataset_split=args.include_dataset_split,
    )
    LOGGER.info("Training baselines for split strategies: %s", list(split_definitions))

    for split_strategy, strategy_frame in split_definitions.items():
        strategy_frame = strategy_frame.copy()
        strategy_frame["_normalized_split"] = (
            strategy_frame["_strategy_split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
        )
        counts = strategy_frame["_normalized_split"].value_counts(dropna=False).to_dict()
        LOGGER.info("%s split counts: %s", split_strategy, counts)

        train = strategy_frame.loc[strategy_frame["_normalized_split"] == "train"].copy()
        validation = strategy_frame.loc[strategy_frame["_normalized_split"] == "validation"].copy()
        test = strategy_frame.loc[strategy_frame["_normalized_split"] == "test"].copy()
        if train.empty or validation.empty or test.empty:
            raise ValueError(
                f"Expected non-empty train, validation, and test sets for {split_strategy}. "
                f"Observed normalized split counts: {counts}"
            )

        X_train = train[numeric_features]
        y_train = train[target_column].to_numpy(dtype=float)
        strategy_model_dir = ensure_directory(model_dir / split_strategy)
        models = baseline_models(random_seed=seed, model_config=config.get("models", {}))

        for model_name, model in models.items():
            LOGGER.info("Training %s on %s", model_name, split_strategy)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            runtime_seconds = time.perf_counter() - start

            with (strategy_model_dir / f"{model_name}.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "model": model,
                        "feature_columns": numeric_features,
                        "target_column": target_column,
                        "split_strategy": split_strategy,
                    },
                    handle,
                )

            for split_name, split_frame in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]:
                y_true = split_frame[target_column].to_numpy(dtype=float)
                y_pred = np.asarray(model.predict(split_frame[numeric_features]), dtype=float)
                metrics = regression_metrics(y_true, y_pred)
                all_metrics.append(
                    {
                        "split_strategy": split_strategy,
                        "model": model_name,
                        "features": "handcrafted_sequence",
                        "split": split_name,
                        "n_examples": len(split_frame),
                        "runtime_seconds": runtime_seconds if split_name == "train" else 0.0,
                        **metrics,
                    }
                )

                prediction_columns = [
                    column
                    for column in [
                        "row_index",
                        dataset_config.get("id_column"),
                        dataset_config.get("organism_column"),
                        dataset_config.get("ec_column"),
                        split_column,
                        target_column,
                    ]
                    if column and column in split_frame.columns
                ]
                predictions = split_frame.loc[:, prediction_columns].copy()
                predictions["split_strategy"] = split_strategy
                predictions["model"] = model_name
                predictions["evaluation_split"] = split_name
                predictions["prediction"] = y_pred
                predictions["residual"] = y_pred - y_true
                predictions["absolute_error"] = np.abs(predictions["residual"])
                all_predictions.append(predictions)

    metrics_frame = pd.DataFrame.from_records(all_metrics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    comparison_frame = build_generalization_comparison(metrics_frame)

    metrics_path = metrics_dir / "baseline_metrics_by_strategy.csv"
    predictions_path = predictions_dir / "baseline_predictions_by_strategy.csv"
    comparison_path = metrics_dir / "baseline_generalization_comparison.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    write_comparison_markdown(comparison_frame, metrics_dir / "baseline_generalization_comparison.md")

    LOGGER.info("Wrote baseline metrics to %s", metrics_path)
    LOGGER.info("Wrote baseline predictions to %s", predictions_path)
    LOGGER.info("Wrote baseline comparison to %s", comparison_path)


def load_split_definitions(
    feature_frame: pd.DataFrame,
    dataset_split_column: str,
    splits_dir: Path,
    include_dataset_split: bool,
) -> dict[str, pd.DataFrame]:
    """Load dataset-provided and generated split definitions."""
    strategies: dict[str, pd.DataFrame] = {}
    indexed_features = feature_frame.reset_index().rename(columns={"index": "row_index"})

    if include_dataset_split:
        if dataset_split_column not in indexed_features.columns:
            raise ValueError(f"Dataset split column is missing: {dataset_split_column}")
        dataset_split = indexed_features.copy()
        dataset_split["_strategy_split"] = dataset_split[dataset_split_column]
        strategies["dataset_provided"] = dataset_split

    split_files = sorted(splits_dir.glob("*_split.csv"))
    if not split_files:
        raise FileNotFoundError(
            f"No generated split files found in {splits_dir}. Run scripts/create_splits.py first."
        )

    for split_file in split_files:
        split_definition = pd.read_csv(split_file)
        required = {"row_index", "split", "split_strategy"}
        missing = required.difference(split_definition.columns)
        if missing:
            raise ValueError(f"Split file {split_file} is missing columns: {sorted(missing)}")
        if split_definition["row_index"].duplicated().any():
            raise ValueError(f"Split file {split_file} contains duplicate row_index values.")

        merged = indexed_features.merge(
            split_definition.loc[:, ["row_index", "split", "split_strategy"]],
            on="row_index",
            how="left",
            validate="one_to_one",
        )
        if merged["split"].isna().any():
            raise ValueError(f"Split file {split_file} does not cover every feature row.")
        strategy_values = merged["split_strategy"].dropna().unique()
        if len(strategy_values) != 1:
            raise ValueError(f"Split file {split_file} must contain exactly one split_strategy value.")
        strategy_name = str(strategy_values[0])
        merged["_strategy_split"] = merged["split"]
        strategies[strategy_name] = merged

    return strategies


def build_generalization_comparison(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact test-set comparison table across split strategies."""
    test_metrics = metrics_frame.loc[metrics_frame["split"] == "test"].copy()
    value_columns = ["mae", "rmse", "r2", "spearman", "pearson"]
    pivot = test_metrics.pivot(index="model", columns="split_strategy", values=value_columns)
    pivot.columns = [f"{strategy}_{metric}" for metric, strategy in pivot.columns]
    comparison = pivot.reset_index()

    if "random_mae" in comparison.columns:
        for column in list(comparison.columns):
            if column.endswith("_mae") and column != "random_mae":
                strategy = column.removesuffix("_mae")
                comparison[f"{strategy}_mae_minus_random_mae"] = comparison[column] - comparison["random_mae"]
    return comparison


def write_comparison_markdown(comparison_frame: pd.DataFrame, path: Path) -> None:
    """Write a readable Markdown version of the baseline comparison."""
    rounded = comparison_frame.round(4)
    columns = list(rounded.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |"
        for _, row in rounded.iterrows()
    ]
    lines = [
        "# Baseline Generalization Comparison",
        "",
        "All values are test-set metrics from models trained only on each strategy's training split.",
        "",
        header,
        separator,
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
