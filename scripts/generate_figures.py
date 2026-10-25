"""Generate publication-ready figures from saved experiment outputs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils import load_config
from src.visualization import clean_label, save_figure, set_publication_style

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--pca-sample", type=int, default=5000, help="Maximum sequences for embedding PCA plot.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    dataset_config = config["dataset"]
    target_column = dataset_config["target_column"]
    sequence_column = dataset_config["sequence_column"]
    organism_column = dataset_config.get("organism_column")
    ec_column = dataset_config.get("ec_column")
    figures_dir = Path(paths["figures_dir"])
    set_publication_style()

    cleaned = pd.read_csv(paths["cleaned_dataset"])
    metrics = pd.read_csv(Path(paths["outputs_dir"]) / "metrics" / "model_comparison_table.csv")
    bootstrap = pd.read_csv(Path(paths["outputs_dir"]) / "metrics" / "bootstrap_confidence_intervals.csv")
    ph_range = pd.read_csv(Path(paths["outputs_dir"]) / "metrics" / "performance_by_ph_range.csv")
    subgroup = pd.read_csv(Path(paths["outputs_dir"]) / "metrics" / "performance_by_subgroup.csv")
    predictions = pd.read_csv(
        Path(paths["outputs_dir"]) / "predictions" / "combined_predictions.csv",
        low_memory=False,
    )

    plot_target_distribution(cleaned, target_column, figures_dir)
    plot_sequence_length_distribution(cleaned, sequence_column, figures_dir)
    plot_dataset_composition(cleaned, organism_column, ec_column, figures_dir)
    plot_model_comparison(bootstrap, figures_dir)
    plot_split_strategy_comparison(bootstrap, figures_dir)
    plot_random_vs_controlled(bootstrap, "cluster_safe_mmseqs_identity_90", figures_dir, "random_vs_cluster_mae")
    plot_random_vs_controlled(bootstrap, "organism_held_out", figures_dir, "random_vs_organism_mae")
    plot_feature_group_improvement(Path(paths["outputs_dir"]) / "metrics" / "feature_group_mae_comparison.csv", figures_dir)
    plot_best_model_prediction_diagnostics(metrics, predictions, target_column, figures_dir)
    plot_ph_range_performance(ph_range, figures_dir)
    plot_subgroup_error(subgroup, "organism", figures_dir, "error_by_organism")
    plot_subgroup_error(subgroup, "ec_number", figures_dir, "error_by_ec_number")
    plot_embedding_pca(cleaned, paths, target_column, figures_dir, sample_size=args.pca_sample)
    LOGGER.info("Generated figures in %s", figures_dir)


def plot_target_distribution(cleaned: pd.DataFrame, target_column: str, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(cleaned[target_column], bins=32, color="#4477AA", edgecolor="white")
    ax.set_xlabel("Experimental optimal pH")
    ax.set_ylabel("Number of enzymes")
    ax.set_title("Target-pH distribution")
    save_figure(fig, figures_dir, "target_ph_distribution")


def plot_sequence_length_distribution(cleaned: pd.DataFrame, sequence_column: str, figures_dir: Path) -> None:
    lengths = cleaned[sequence_column].astype(str).map(len)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(lengths, bins=45, color="#228833", edgecolor="white")
    ax.set_xlabel("Sequence length (amino acids)")
    ax.set_ylabel("Number of enzymes")
    ax.set_title("Sequence-length distribution")
    save_figure(fig, figures_dir, "sequence_length_distribution")


def plot_dataset_composition(
    cleaned: pd.DataFrame,
    organism_column: str | None,
    ec_column: str | None,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, column, title in [
        (axes[0], organism_column, "Top organisms"),
        (axes[1], ec_column, "Top EC numbers"),
    ]:
        if column and column in cleaned.columns:
            counts = cleaned[column].astype(str).value_counts().head(12).sort_values()
            ax.barh(counts.index, counts.values, color="#66CCEE")
            ax.set_xlabel("Records")
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, "Metadata unavailable", ha="center", va="center")
            ax.set_axis_off()
    save_figure(fig, figures_dir, "dataset_composition")


def plot_model_comparison(bootstrap: pd.DataFrame, figures_dir: Path) -> None:
    selected = bootstrap.loc[
        (bootstrap["split_strategy"].isin(["random", "cluster_safe_mmseqs_identity_90", "organism_held_out"]))
        & (bootstrap["model"] != "mean_predictor")
    ].copy()
    selected["label"] = selected["feature_group"].map(clean_label) + " / " + selected["model"].map(clean_label)
    for split_strategy, group in selected.groupby("split_strategy"):
        group = group.sort_values("mae")
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        y = np.arange(len(group))
        errors = np.vstack(
            [
                group["mae"] - group["mae_ci_lower"],
                group["mae_ci_upper"] - group["mae"],
            ]
        )
        ax.barh(y, group["mae"], xerr=errors, color="#4477AA", alpha=0.85, capsize=3)
        ax.set_yticks(y, group["label"])
        ax.set_xlabel("Test MAE")
        ax.set_title(f"Model comparison: {clean_label(split_strategy)}")
        save_figure(fig, figures_dir, f"model_comparison_{split_strategy}")


def plot_split_strategy_comparison(bootstrap: pd.DataFrame, figures_dir: Path) -> None:
    selected = (
        bootstrap.loc[bootstrap["model"] != "mean_predictor"]
        .sort_values(["feature_group", "split_strategy", "mae"])
        .groupby(["feature_group", "split_strategy"], as_index=False)
        .head(1)
    )
    selected["label"] = selected["feature_group"].map(clean_label) + "\n" + selected["model"].map(clean_label)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pivot = selected.pivot(index="label", columns="split_strategy", values="mae")
    pivot = pivot.loc[sorted(pivot.index)]
    pivot.plot(kind="bar", ax=ax, color=["#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377"])
    ax.set_ylabel("Test MAE")
    ax.set_xlabel("")
    ax.set_title("Best non-mean model per feature group and split strategy")
    ax.legend(title="Split strategy", labels=[clean_label(col) for col in pivot.columns])
    save_figure(fig, figures_dir, "split_strategy_comparison")


def plot_random_vs_controlled(
    bootstrap: pd.DataFrame,
    controlled_strategy: str,
    figures_dir: Path,
    name: str,
) -> None:
    selected = bootstrap.loc[
        (bootstrap["split_strategy"].isin(["random", controlled_strategy]))
        & (bootstrap["model"] != "mean_predictor")
    ].copy()
    selected["model_label"] = selected["feature_group"].map(clean_label) + " / " + selected["model"].map(clean_label)
    pivot = selected.pivot_table(index="model_label", columns="split_strategy", values="mae")
    pivot = pivot.dropna().sort_values(controlled_strategy)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(pivot["random"], pivot[controlled_strategy], color="#AA3377", s=46)
    lower = min(pivot.min().min(), 0.5)
    upper = max(pivot.max().max(), 0.9)
    ax.plot([lower, upper], [lower, upper], color="#666666", linestyle="--", linewidth=1)
    for label, row in pivot.iterrows():
        ax.annotate(label, (row["random"], row[controlled_strategy]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Random split MAE")
    ax.set_ylabel(f"{clean_label(controlled_strategy)} MAE")
    ax.set_title(f"Random vs {clean_label(controlled_strategy)}")
    save_figure(fig, figures_dir, name)


def plot_feature_group_improvement(comparison_path: Path, figures_dir: Path) -> None:
    comparison = pd.read_csv(comparison_path)
    selected = comparison.loc[comparison["model"] != "mean_predictor"].copy()
    selected["label"] = selected["split_strategy"].map(clean_label) + "\n" + selected["model"].map(clean_label)
    selected = selected.sort_values("embedding_mae_minus_handcrafted_mae")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.barh(selected["label"], selected["embedding_mae_minus_handcrafted_mae"], color="#228833")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("ESM-2 MAE minus handcrafted MAE")
    ax.set_title("Embedding feature MAE change")
    save_figure(fig, figures_dir, "embedding_vs_handcrafted_mae_change")


def plot_best_model_prediction_diagnostics(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    target_column: str,
    figures_dir: Path,
) -> None:
    generated_strategies = ["random", "exact_duplicate_safe", "cluster_safe_mmseqs_identity_90", "organism_held_out"]
    best = (
        metrics.loc[
            (metrics["feature_group"] == "esm2_embedding")
            & (metrics["model"] != "mean_predictor")
            & (metrics["split_strategy"].isin(generated_strategies))
        ]
        .sort_values(["split_strategy", "mae"])
        .groupby("split_strategy", as_index=False)
        .head(1)
    )
    selected_parts = []
    for row in best.itertuples(index=False):
        selected_parts.append(
            predictions.loc[
                (predictions["feature_group"] == row.feature_group)
                & (predictions["split_strategy"] == row.split_strategy)
                & (predictions["model"] == row.model)
                & (predictions["evaluation_split"] == "test")
            ].copy()
        )
    selected = pd.concat(selected_parts, ignore_index=True)

    strategies = list(best["split_strategy"])
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharex=True, sharey=True)
    for ax, strategy in zip(axes.ravel(), strategies):
        group = selected.loc[selected["split_strategy"] == strategy]
        ax.scatter(group[target_column], group["prediction"], s=10, alpha=0.35, color="#4477AA")
        low = min(group[target_column].min(), group["prediction"].min())
        high = max(group[target_column].max(), group["prediction"].max())
        ax.plot([low, high], [low, high], color="#333333", linewidth=1)
        model_name = group["model"].iloc[0]
        ax.set_title(f"{clean_label(strategy)}\n{clean_label(model_name)}")
        ax.set_xlabel("Actual pH")
        ax.set_ylabel("Predicted pH")
    save_figure(fig, figures_dir, "actual_vs_predicted_best_embedding_models")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharex=True)
    for ax, strategy in zip(axes.ravel(), strategies):
        group = selected.loc[selected["split_strategy"] == strategy]
        ax.hist(group["residual"], bins=35, color="#EE6677", edgecolor="white")
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(clean_label(strategy))
        ax.set_xlabel("Prediction residual")
        ax.set_ylabel("Count")
    save_figure(fig, figures_dir, "residual_distribution_best_embedding_models")

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharex=True, sharey=True)
    for ax, strategy in zip(axes.ravel(), strategies):
        group = selected.loc[selected["split_strategy"] == strategy]
        ax.scatter(group[target_column], group["absolute_error"], s=10, alpha=0.35, color="#CCBB44")
        ax.set_title(clean_label(strategy))
        ax.set_xlabel("Actual pH")
        ax.set_ylabel("Absolute error")
    save_figure(fig, figures_dir, "error_vs_true_ph_best_embedding_models")


def plot_ph_range_performance(ph_range: pd.DataFrame, figures_dir: Path) -> None:
    selected = ph_range.loc[
        (ph_range["feature_group"] == "esm2_embedding")
        & (ph_range["model"].isin(["random_forest", "hist_gradient_boosting"]))
    ].copy()
    selected["label"] = selected["split_strategy"].map(clean_label) + " / " + selected["model"].map(clean_label)
    for split_strategy, group in selected.groupby("split_strategy"):
        pivot = group.pivot(index="label", columns="ph_range", values="mae")
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        pivot.plot(kind="bar", ax=ax, color=["#4477AA", "#EE6677", "#228833"])
        ax.set_ylabel("Test MAE")
        ax.set_xlabel("")
        ax.set_title(f"Performance by pH range: {clean_label(split_strategy)}")
        ax.legend(title="pH range")
        save_figure(fig, figures_dir, f"performance_by_ph_range_{split_strategy}")


def plot_subgroup_error(subgroup: pd.DataFrame, subgroup_type: str, figures_dir: Path, name: str) -> None:
    selected = subgroup.loc[
        (subgroup["subgroup_type"] == subgroup_type)
        & (subgroup["feature_group"] == "esm2_embedding")
        & (subgroup["model"] == "random_forest")
        & (subgroup["split_strategy"] == "dataset_provided")
    ].copy()
    if selected.empty:
        return
    selected = selected.sort_values("mae", ascending=False).head(15).sort_values("mae")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(selected["subgroup"].astype(str), selected["mae"], color="#66CCEE")
    ax.set_xlabel("Test MAE")
    ax.set_title(f"Highest error {subgroup_type} groups")
    save_figure(fig, figures_dir, name)


def plot_embedding_pca(
    cleaned: pd.DataFrame,
    paths: dict[str, object],
    target_column: str,
    figures_dir: Path,
    sample_size: int,
) -> None:
    embedding_path = Path(str(paths["embedding_file"]))
    if not embedding_path.exists():
        LOGGER.warning("Embedding file missing; skipping PCA figure: %s", embedding_path)
        return
    embeddings = np.load(embedding_path, mmap_mode="r")
    n = embeddings.shape[0]
    rng = np.random.default_rng(42)
    indices = np.arange(n)
    if n > sample_size:
        indices = np.sort(rng.choice(indices, size=sample_size, replace=False))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(np.asarray(embeddings[indices], dtype=np.float32))
    values = cleaned.loc[indices, target_column].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=10, cmap="viridis", alpha=0.75)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("ESM-2 embedding PCA colored by optimal pH")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Experimental optimal pH")
    save_figure(fig, figures_dir, "embedding_pca_optimal_ph")


if __name__ == "__main__":
    main()
