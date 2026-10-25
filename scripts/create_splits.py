"""Create reproducible train-validation-test split definitions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.splitting import (
    SplitFractions,
    build_split_frame,
    find_group_leakage,
    make_balanced_grouped_split,
    make_grouped_split,
    make_random_split,
    split_counts,
    write_split_summary,
)
from src.utils import configure_logging, ensure_directory, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)

    dataset_config = config["dataset"]
    split_config = config["splitting"]
    paths = config["paths"]
    seed = int(config.get("project", {}).get("random_seed", 42))

    cleaned_path = Path(paths["cleaned_dataset"])
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run scripts/prepare_data.py first."
        )

    frame = pd.read_csv(cleaned_path)
    fractions = SplitFractions(
        train=float(split_config["train_fraction"]),
        validation=float(split_config["validation_fraction"]),
        test=float(split_config["test_fraction"]),
    )

    splits_dir = ensure_directory(paths["splits_dir"])
    id_column = dataset_config["id_column"]
    sequence_column = dataset_config["sequence_column"]

    LOGGER.info("Creating row-wise random split for %s records", len(frame))
    random_assignments = make_random_split(frame, fractions=fractions, random_seed=seed)
    random_split = build_split_frame(
        frame=frame,
        split_assignments=random_assignments,
        strategy="random",
        id_column=id_column,
        sequence_column=sequence_column,
    )
    random_path = splits_dir / "random_split.csv"
    random_split.to_csv(random_path, index=False)
    generated_split_frames = [random_split]

    random_leakage = find_group_leakage(random_split, "sequence_sha256")
    LOGGER.info(
        "Random split counts: %s; exact duplicate leakage groups: %s",
        split_counts(random_split),
        len(random_leakage),
    )

    LOGGER.info("Creating exact-duplicate-safe split")
    duplicate_safe_assignments = make_grouped_split(
        frame=frame,
        group_column=sequence_column,
        fractions=fractions,
        random_seed=seed,
    )
    duplicate_safe_split = build_split_frame(
        frame=frame,
        split_assignments=duplicate_safe_assignments,
        strategy="exact_duplicate_safe",
        id_column=id_column,
        sequence_column=sequence_column,
    )
    duplicate_safe_path = splits_dir / "exact_duplicate_safe_split.csv"
    duplicate_safe_split.to_csv(duplicate_safe_path, index=False)
    generated_split_frames.append(duplicate_safe_split)

    duplicate_safe_leakage = find_group_leakage(duplicate_safe_split, "sequence_sha256")
    if not duplicate_safe_leakage.empty:
        raise ValueError("Exact-duplicate-safe split leaked duplicate sequences across splits.")

    LOGGER.info("Exact-duplicate-safe split counts: %s", split_counts(duplicate_safe_split))

    summaries: list[dict[str, object]] = [
        {
            "strategy": "random",
            "counts": split_counts(random_split),
            "group_leakage_count": len(random_leakage),
            "notes": "Row-wise random split; exact duplicate sequences may cross splits.",
        },
        {
            "strategy": "exact_duplicate_safe",
            "counts": split_counts(duplicate_safe_split),
            "group_leakage_count": len(duplicate_safe_leakage),
            "notes": "All exact duplicate sequences are assigned to one split.",
        },
    ]

    organism_column = dataset_config.get("organism_column")
    if organism_column and organism_column in frame.columns and frame[organism_column].notna().any():
        LOGGER.info("Creating organism-held-out split using %s", organism_column)
        organism_assignments = make_balanced_grouped_split(
            frame=frame,
            group_column=organism_column,
            fractions=fractions,
            random_seed=seed,
        )
        organism_split = build_split_frame(
            frame=frame,
            split_assignments=organism_assignments,
            strategy="organism_held_out",
            id_column=id_column,
            sequence_column=sequence_column,
        )
        organism_split[organism_column] = frame[organism_column].astype(str).to_numpy()
        organism_leakage = find_group_leakage(organism_split, organism_column)
        if not organism_leakage.empty:
            raise ValueError("Organism-held-out split leaked organisms across splits.")

        organism_path = splits_dir / "organism_held_out_split.csv"
        organism_split.to_csv(organism_path, index=False)
        generated_split_frames.append(organism_split)
        summaries.append(
            {
                "strategy": "organism_held_out",
                "counts": split_counts(organism_split),
                "group_leakage_count": len(organism_leakage),
                "notes": f"All records from the same {organism_column} are assigned to one split.",
            }
        )
        LOGGER.info("Organism-held-out split counts: %s", split_counts(organism_split))
    else:
        LOGGER.info("No organism metadata found; skipping organism-held-out split.")

    ec_column = dataset_config.get("ec_column")
    if ec_column and ec_column in frame.columns and frame[ec_column].notna().any():
        LOGGER.info("Creating EC-held-out split using %s", ec_column)
        ec_assignments = make_balanced_grouped_split(
            frame=frame,
            group_column=ec_column,
            fractions=fractions,
            random_seed=seed,
        )
        ec_split = build_split_frame(
            frame=frame,
            split_assignments=ec_assignments,
            strategy="ec_held_out",
            id_column=id_column,
            sequence_column=sequence_column,
        )
        ec_split[ec_column] = frame[ec_column].astype(str).to_numpy()
        ec_leakage = find_group_leakage(ec_split, ec_column)
        if not ec_leakage.empty:
            raise ValueError("EC-held-out split leaked EC groups across splits.")

        ec_path = splits_dir / "ec_held_out_split.csv"
        ec_split.to_csv(ec_path, index=False)
        generated_split_frames.append(ec_split)
        summaries.append(
            {
                "strategy": "ec_held_out",
                "counts": split_counts(ec_split),
                "group_leakage_count": len(ec_leakage),
                "notes": f"All records from the same {ec_column} are assigned to one split.",
            }
        )
        LOGGER.info("EC-held-out split counts: %s", split_counts(ec_split))
    else:
        LOGGER.info("No EC metadata found; skipping EC-held-out split.")

    cluster_assignment_files = sorted(
        Path(paths.get("clusters_dir", "data/processed/clusters")).glob(
            "cluster_assignments_*_identity_*.csv"
        )
    )
    if cluster_assignment_files:
        LOGGER.info("Creating cluster-based splits from %s assignment files", len(cluster_assignment_files))
    else:
        LOGGER.info("No cluster assignment files found; skipping cluster-based splits for now.")

    for assignment_path in cluster_assignment_files:
        assignments = pd.read_csv(assignment_path)
        if "row_index" not in assignments.columns or "cluster_id" not in assignments.columns:
            raise ValueError(f"Invalid cluster assignment file: {assignment_path}")
        assignment_lookup = assignments.set_index("row_index")
        clustered = frame.copy()
        clustered["cluster_id"] = assignment_lookup.reindex(frame.index)["cluster_id"].to_numpy()
        if clustered["cluster_id"].isna().any():
            raise ValueError(f"Cluster assignment file is missing rows: {assignment_path}")

        strategy = assignment_path.stem.replace("cluster_assignments_", "cluster_safe_")
        cluster_assignments = make_grouped_split(
            frame=clustered,
            group_column="cluster_id",
            fractions=fractions,
            random_seed=seed,
        )
        cluster_split = build_split_frame(
            frame=clustered,
            split_assignments=cluster_assignments,
            strategy=strategy,
            id_column=id_column,
            sequence_column=sequence_column,
        )
        cluster_split["cluster_id"] = clustered["cluster_id"].to_numpy()
        cluster_leakage = find_group_leakage(cluster_split, "cluster_id")
        if not cluster_leakage.empty:
            raise ValueError(f"Cluster-safe split leaked clusters across splits: {assignment_path}")

        output_path = splits_dir / f"{strategy}_split.csv"
        cluster_split.to_csv(output_path, index=False)
        generated_split_frames.append(cluster_split)
        summaries.append(
            {
                "strategy": strategy,
                "counts": split_counts(cluster_split),
                "group_leakage_count": len(cluster_leakage),
                "notes": f"Cluster-safe split from {assignment_path.name}.",
            }
        )
        LOGGER.info("Wrote %s with counts %s", output_path, split_counts(cluster_split))

    write_split_summary(summaries, splits_dir / "split_summary.md")
    write_split_target_summary(
        frame=frame,
        split_frames=generated_split_frames,
        target_column=dataset_config["target_column"],
        organism_column=dataset_config.get("organism_column"),
        output_path=splits_dir / "split_target_summary.csv",
    )

    LOGGER.info("Wrote split files to %s", splits_dir)


def write_split_target_summary(
    frame: pd.DataFrame,
    split_frames: list[pd.DataFrame],
    target_column: str,
    organism_column: str | None,
    output_path: Path,
) -> None:
    """Write target and metadata summaries for each generated split."""
    indexed = frame.reset_index().rename(columns={"index": "row_index"})
    rows: list[dict[str, object]] = []
    for split_frame in split_frames:
        columns = ["row_index", "split", "split_strategy"]
        if "cluster_id" in split_frame.columns:
            columns.append("cluster_id")
        merged = indexed.merge(split_frame.loc[:, columns], on="row_index", how="inner")
        for (strategy, split_name), group in merged.groupby(["split_strategy", "split"]):
            row: dict[str, object] = {
                "split_strategy": strategy,
                "split": split_name,
                "n_examples": len(group),
                "target_mean": group[target_column].mean(),
                "target_std": group[target_column].std(),
                "target_min": group[target_column].min(),
                "target_max": group[target_column].max(),
            }
            if organism_column and organism_column in group.columns:
                row["unique_organisms"] = group[organism_column].nunique()
            if "cluster_id" in group.columns:
                row["unique_clusters"] = group["cluster_id"].nunique()
            rows.append(row)

    pd.DataFrame.from_records(rows).sort_values(["split_strategy", "split"]).to_csv(
        output_path, index=False
    )


if __name__ == "__main__":
    main()
