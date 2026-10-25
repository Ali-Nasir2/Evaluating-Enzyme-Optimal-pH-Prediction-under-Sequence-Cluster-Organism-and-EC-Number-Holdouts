"""Create consistent evaluation tables from saved model predictions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation import (
    bootstrap_metric_intervals,
    paired_absolute_error_comparisons,
    ph_range_label,
    summarize_predictions,
)
from src.utils import configure_logging, ensure_directory, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override the configured number of bootstrap iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    paths = config["paths"]
    dataset_config = config["dataset"]
    evaluation_config = config.get("evaluation", {})
    target_column = dataset_config["target_column"]
    organism_column = dataset_config.get("organism_column")
    ec_column = dataset_config.get("ec_column")
    seed = int(config.get("project", {}).get("random_seed", 42))
    n_bootstrap = int(args.bootstrap_iterations or evaluation_config.get("bootstrap_iterations", 500))
    confidence_level = float(evaluation_config.get("bootstrap_confidence_level", 0.95))
    subgroup_min_count = int(evaluation_config.get("subgroup_min_count", 20))
    subgroup_top_n = int(evaluation_config.get("subgroup_top_n", 25))

    metrics_dir = ensure_directory(Path(paths["outputs_dir"]) / "metrics")
    predictions_dir = ensure_directory(Path(paths["outputs_dir"]) / "predictions")
    report_tables_dir = ensure_directory(Path(paths["tables_dir"]))
    paper_tables_dir = ensure_directory(Path("paper") / "tables")

    combined = load_combined_predictions(predictions_dir)
    combined_path = predictions_dir / "combined_predictions.csv"
    combined.to_csv(combined_path, index=False)
    LOGGER.info("Wrote combined predictions to %s", combined_path)

    test_predictions = combined.loc[combined["evaluation_split"] == "test"].copy()
    if test_predictions.empty:
        raise ValueError("No test predictions were found.")

    model_comparison = summarize_predictions(
        test_predictions,
        group_columns=["feature_group", "split_strategy", "model"],
        target_column=target_column,
    ).sort_values(["split_strategy", "mae", "feature_group", "model"])
    model_comparison_path = metrics_dir / "model_comparison_table.csv"
    model_comparison.to_csv(model_comparison_path, index=False)

    LOGGER.info("Computing bootstrap intervals with %s iterations", n_bootstrap)
    bootstrap_rows: list[dict[str, object]] = []
    for keys, group in test_predictions.groupby(["feature_group", "split_strategy", "model"]):
        feature_group, split_strategy, model = keys
        intervals = bootstrap_metric_intervals(
            group[target_column].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=seed,
        )
        bootstrap_rows.append(
            {
                "feature_group": feature_group,
                "split_strategy": split_strategy,
                "model": model,
                "n_examples": len(group),
                "confidence_level": confidence_level,
                "bootstrap_iterations": n_bootstrap,
                **intervals,
            }
        )
    bootstrap_frame = pd.DataFrame.from_records(bootstrap_rows).sort_values(
        ["split_strategy", "mae", "feature_group", "model"]
    )
    bootstrap_path = metrics_dir / "bootstrap_confidence_intervals.csv"
    bootstrap_frame.to_csv(bootstrap_path, index=False)

    ph_predictions = test_predictions.copy()
    ph_predictions["ph_range"] = ph_predictions[target_column].map(ph_range_label)
    ph_summary = summarize_predictions(
        ph_predictions,
        group_columns=["feature_group", "split_strategy", "model", "ph_range"],
        target_column=target_column,
    ).sort_values(["feature_group", "split_strategy", "model", "ph_range"])
    ph_summary_path = metrics_dir / "performance_by_ph_range.csv"
    ph_summary.to_csv(ph_summary_path, index=False)

    subgroup_frames: list[pd.DataFrame] = []
    for column, label in [(organism_column, "organism"), (ec_column, "ec_number")]:
        if column and column in test_predictions.columns:
            top_values = (
                test_predictions[column]
                .dropna()
                .astype(str)
                .value_counts()
                .loc[lambda counts: counts >= subgroup_min_count]
                .head(subgroup_top_n)
                .index
            )
            subgroup_input = test_predictions.loc[test_predictions[column].astype(str).isin(top_values)].copy()
            subgroup_summary = summarize_predictions(
                subgroup_input,
                group_columns=["feature_group", "split_strategy", "model", column],
                target_column=target_column,
                min_count=subgroup_min_count,
            )
            if not subgroup_summary.empty:
                subgroup_summary = subgroup_summary.rename(columns={column: "subgroup"})
                subgroup_summary.insert(0, "subgroup_type", label)
                subgroup_frames.append(subgroup_summary)

    subgroup_table = (
        pd.concat(subgroup_frames, ignore_index=True)
        if subgroup_frames
        else pd.DataFrame()
    )
    subgroup_path = metrics_dir / "performance_by_subgroup.csv"
    subgroup_table.to_csv(subgroup_path, index=False)

    top_errors = (
        test_predictions.sort_values("absolute_error", ascending=False)
        .groupby(["feature_group", "split_strategy", "model"], group_keys=False)
        .head(50)
        .sort_values(["feature_group", "split_strategy", "model", "absolute_error"], ascending=[True, True, True, False])
    )
    top_errors_path = predictions_dir / "top_50_errors_by_model.csv"
    top_errors.to_csv(top_errors_path, index=False)

    paired = paired_absolute_error_comparisons(
        test_predictions,
        n_bootstrap=n_bootstrap,
        random_seed=seed,
    )
    paired_path = metrics_dir / "paired_feature_group_comparisons.csv"
    paired.to_csv(paired_path, index=False)

    write_latex_table(
        model_comparison,
        report_tables_dir / "model_comparison_table.tex",
        columns=["feature_group", "split_strategy", "model", "n_examples", "mae", "rmse", "r2", "spearman", "pearson"],
        caption="Test-set performance by feature group, split strategy, and model.",
        label="tab:model-comparison",
    )
    write_latex_table(
        model_comparison,
        paper_tables_dir / "model_comparison_table.tex",
        columns=["feature_group", "split_strategy", "model", "n_examples", "mae", "rmse", "r2", "spearman", "pearson"],
        caption="Test-set performance by feature group, split strategy, and model.",
        label="tab:model-comparison",
    )
    write_compact_main_table(
        bootstrap_frame,
        paper_tables_dir / "compact_model_comparison_table.tex",
    )
    write_latex_table(
        ph_summary,
        report_tables_dir / "performance_by_ph_range.tex",
        columns=["feature_group", "split_strategy", "model", "ph_range", "n_examples", "mae", "rmse", "bias"],
        caption="Test-set performance stratified by experimental optimal-pH range.",
        label="tab:performance-by-ph-range",
    )
    write_latex_table(
        ph_summary,
        paper_tables_dir / "performance_by_ph_range.tex",
        columns=["feature_group", "split_strategy", "model", "ph_range", "n_examples", "mae", "rmse", "bias"],
        caption="Test-set performance stratified by experimental optimal-pH range.",
        label="tab:performance-by-ph-range",
    )

    LOGGER.info("Wrote model comparison to %s", model_comparison_path)
    LOGGER.info("Wrote bootstrap intervals to %s", bootstrap_path)
    LOGGER.info("Wrote pH-range summary to %s", ph_summary_path)
    LOGGER.info("Wrote subgroup summary to %s", subgroup_path)
    LOGGER.info("Wrote top errors to %s", top_errors_path)
    LOGGER.info("Wrote paired comparisons to %s", paired_path)


def load_combined_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load handcrafted and embedding prediction files with feature-group labels."""
    files = [
        ("handcrafted", predictions_dir / "baseline_predictions_by_strategy.csv"),
        ("esm2_embedding", predictions_dir / "embedding_predictions_by_strategy.csv"),
    ]
    frames: list[pd.DataFrame] = []
    for feature_group, path in files:
        if not path.exists():
            LOGGER.warning("Prediction file missing, skipping: %s", path)
            continue
        frame = pd.read_csv(path)
        frame["feature_group"] = feature_group
        if "embedding_model" not in frame.columns:
            frame["embedding_model"] = ""
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
    caption: str,
    label: str,
    max_rows: int | None = None,
) -> None:
    """Write a simple LaTeX table from selected columns."""
    available_columns = [column for column in columns if column in frame.columns]
    table = frame.loc[:, available_columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    numeric_columns = table.select_dtypes(include="number").columns
    table.loc[:, numeric_columns] = table.loc[:, numeric_columns].round(4)
    latex = table.to_latex(
        index=False,
        escape=True,
        longtable=True,
        caption=caption,
        label=label,
    )
    path.write_text(latex, encoding="utf-8")


def write_compact_main_table(bootstrap: pd.DataFrame, path: Path) -> None:
    """Write one readable best-model row per feature group and split strategy."""
    selected = (
        bootstrap.loc[bootstrap["model"].ne("mean_predictor")]
        .sort_values("mae")
        .groupby(["feature_group", "split_strategy"], as_index=False)
        .first()
    )
    labels = {
        "handcrafted": "Handcrafted",
        "esm2_embedding": "ESM-2 embeddings",
        "cluster_safe_mmseqs_identity_50": "MMseqs2 50\\%",
        "cluster_safe_mmseqs_identity_70": "MMseqs2 70\\%",
        "cluster_safe_mmseqs_identity_90": "MMseqs2 90\\%",
        "exact_duplicate_safe": "Exact-duplicate-safe",
        "organism_held_out": "Organism-held-out",
        "ec_held_out": "EC-number-held-out",
        "random": "Random",
        "dataset_provided": "Dataset-provided",
    }
    selected["Feature group"] = selected["feature_group"].map(
        {"handcrafted": "Handcrafted", "esm2_embedding": "ESM-2 embeddings"}
    )
    selected["Split strategy"] = selected["split_strategy"].map(labels).fillna(
        selected["split_strategy"]
    )
    selected["Model"] = selected["model"].map(
        {
            "random_forest": "Random forest",
            "hist_gradient_boosting": "Histogram gradient boosting",
            "ridge_regression": "Ridge regression",
        }
    ).fillna(selected["model"])
    selected["MAE (95\\% CI)"] = selected.apply(
        lambda row: f"{row['mae']:.3f} [{row['mae_ci_lower']:.3f}, {row['mae_ci_upper']:.3f}]",
        axis=1,
    )
    table = selected[
        ["Feature group", "Split strategy", "Model", "n_examples", "MAE (95\\% CI)"]
    ].rename(columns={"n_examples": "n examples"})
    latex = table.to_latex(
        index=False,
        escape=False,
        longtable=False,
        caption="Best non-mean model by feature group and split strategy; MAE is shown with a row-level bootstrap 95\\% confidence interval.",
        label="tab:model-comparison",
        column_format="lllrl",
    )
    path.write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
