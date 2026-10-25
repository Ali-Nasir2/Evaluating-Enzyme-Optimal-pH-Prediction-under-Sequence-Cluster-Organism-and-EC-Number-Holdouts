"""Publication-quality figure generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def set_publication_style() -> None:
    """Configure a restrained Matplotlib style for paper figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output_dir: str | Path, name: str) -> None:
    """Save a figure as PNG and PDF."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def clean_label(value: object) -> str:
    """Convert internal identifiers to readable plot labels."""
    return str(value).replace("_", " ")


def add_value_labels(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    """Annotate bar containers with numeric values."""
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append("" if pd.isna(value) else fmt.format(value))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def best_non_mean_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the lowest-MAE non-mean model for each feature/split strategy."""
    candidates = metrics.loc[
        (metrics["split"] == "test") & (metrics["model"] != "mean_predictor")
    ].copy()
    order = candidates.sort_values(["feature_group", "split_strategy", "mae"])
    return order.groupby(["feature_group", "split_strategy"], as_index=False).head(1)

