"""Train regressors on frozen protein embeddings across split strategies."""

from __future__ import annotations

import argparse
import json
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

from scripts.train_baselines import (
    SPLIT_ALIASES,
    build_generalization_comparison,
    load_split_definitions,
    write_comparison_markdown,
)
from src.evaluation import regression_metrics
from src.models import baseline_models
from src.utils import configure_logging, ensure_directory, load_config, set_reproducible_seed

LOGGER = logging.getLogger(__name__)


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

    paths = config["paths"]
    dataset_config = config["dataset"]
    target_column = dataset_config["target_column"]
    split_column = dataset_config.get("split_column", "Split")

    cleaned_path = Path(paths["cleaned_dataset"])
    embedding_path = Path(paths["embedding_file"])
    metadata_path = Path(paths["embedding_metadata"])
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run scripts/prepare_data.py first."
        )
    if not embedding_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Embedding cache is missing. Run scripts/generate_embeddings.py before training "
            "embedding-based models."
        )

    frame = pd.read_csv(cleaned_path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column missing from cleaned dataset: {target_column}")
    embeddings = np.load(embedding_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(frame):
        raise ValueError(
            f"Embedding row count {embeddings.shape[0]} does not match cleaned dataset rows {len(frame)}."
        )

    split_definitions = load_split_definitions(
        feature_frame=frame,
        dataset_split_column=split_column,
        splits_dir=Path(paths["splits_dir"]),
        include_dataset_split=args.include_dataset_split,
    )

    model_dir = ensure_directory(Path(paths["outputs_dir"]) / "models" / "embedding_regressors")
    metrics_dir = ensure_directory(Path(paths["outputs_dir"]) / "metrics")
    predictions_dir = ensure_directory(Path(paths["outputs_dir"]) / "predictions")

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    feature_columns = [f"embedding_{index}" for index in range(embeddings.shape[1])]

    LOGGER.info(
        "Training embedding regressors using %s with shape %s",
        metadata.get("model_name", "unknown_model"),
        embeddings.shape,
    )

    for split_strategy, strategy_frame in split_definitions.items():
        strategy_frame = strategy_frame.copy()
        strategy_frame["_normalized_split"] = (
            strategy_frame["_strategy_split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
        )
        counts = strategy_frame["_normalized_split"].value_counts(dropna=False).to_dict()
        LOGGER.info("%s split counts: %s", split_strategy, counts)

        train_indices = strategy_frame.loc[
            strategy_frame["_normalized_split"] == "train", "row_index"
        ].to_numpy(dtype=int)
        validation_indices = strategy_frame.loc[
            strategy_frame["_normalized_split"] == "validation", "row_index"
        ].to_numpy(dtype=int)
        test_indices = strategy_frame.loc[
            strategy_frame["_normalized_split"] == "test", "row_index"
        ].to_numpy(dtype=int)
        if len(train_indices) == 0 or len(validation_indices) == 0 or len(test_indices) == 0:
            raise ValueError(f"Split strategy {split_strategy} has an empty partition: {counts}")

        strategy_model_dir = ensure_directory(model_dir / split_strategy)
        models = baseline_models(random_seed=seed, model_config=config.get("models", {}))
        X_train = embeddings[train_indices]
        y_train = frame.loc[train_indices, target_column].to_numpy(dtype=float)

        for model_name, model in models.items():
            LOGGER.info("Training %s on embeddings for %s", model_name, split_strategy)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            runtime_seconds = time.perf_counter() - start

            with (strategy_model_dir / f"{model_name}.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "model": model,
                        "feature_columns": feature_columns,
                        "target_column": target_column,
                        "split_strategy": split_strategy,
                        "embedding_metadata": metadata,
                    },
                    handle,
                )

            for split_name, row_indices in [
                ("train", train_indices),
                ("validation", validation_indices),
                ("test", test_indices),
            ]:
                y_true = frame.loc[row_indices, target_column].to_numpy(dtype=float)
                y_pred = np.asarray(model.predict(embeddings[row_indices]), dtype=float)
                metrics = regression_metrics(y_true, y_pred)
                all_metrics.append(
                    {
                        "split_strategy": split_strategy,
                        "model": model_name,
                        "features": "frozen_protein_embedding",
                        "embedding_model": metadata.get("model_name", "unknown_model"),
                        "split": split_name,
                        "n_examples": len(row_indices),
                        "runtime_seconds": runtime_seconds if split_name == "train" else 0.0,
                        **metrics,
                    }
                )

                prediction_columns = [
                    column
                    for column in [
                        dataset_config.get("id_column"),
                        dataset_config.get("organism_column"),
                        dataset_config.get("ec_column"),
                        split_column,
                        target_column,
                    ]
                    if column and column in frame.columns
                ]
                predictions = frame.loc[row_indices, prediction_columns].copy()
                predictions.insert(0, "row_index", row_indices)
                predictions["split_strategy"] = split_strategy
                predictions["model"] = model_name
                predictions["embedding_model"] = metadata.get("model_name", "unknown_model")
                predictions["evaluation_split"] = split_name
                predictions["prediction"] = y_pred
                predictions["residual"] = y_pred - y_true
                predictions["absolute_error"] = np.abs(predictions["residual"])
                all_predictions.append(predictions)

    metrics_frame = pd.DataFrame.from_records(all_metrics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    comparison_frame = build_generalization_comparison(metrics_frame)

    metrics_path = metrics_dir / "embedding_metrics_by_strategy.csv"
    predictions_path = predictions_dir / "embedding_predictions_by_strategy.csv"
    comparison_path = metrics_dir / "embedding_generalization_comparison.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    write_comparison_markdown(comparison_frame, metrics_dir / "embedding_generalization_comparison.md")

    LOGGER.info("Wrote embedding metrics to %s", metrics_path)
    LOGGER.info("Wrote embedding predictions to %s", predictions_path)
    LOGGER.info("Wrote embedding comparison to %s", comparison_path)


if __name__ == "__main__":
    main()
