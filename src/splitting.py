"""Train-validation-test split strategies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitFractions:
    """Fractions for train, validation, and test partitions."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        values = [self.train, self.validation, self.test]
        if any(value <= 0 for value in values):
            raise ValueError(f"Split fractions must be positive, got {values}")
        if not np.isclose(sum(values), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(values):.6f}")


def make_random_split(
    frame: pd.DataFrame,
    fractions: SplitFractions,
    random_seed: int,
) -> pd.Series:
    """Create a row-wise random split assignment."""
    fractions.validate()
    n_rows = len(frame)
    if n_rows < 3:
        raise ValueError("At least three records are required to create train/validation/test splits.")

    rng = np.random.default_rng(random_seed)
    shuffled_positions = rng.permutation(n_rows)
    train_end = int(round(n_rows * fractions.train))
    validation_end = train_end + int(round(n_rows * fractions.validation))

    split_values = np.empty(n_rows, dtype=object)
    split_values[shuffled_positions[:train_end]] = "train"
    split_values[shuffled_positions[train_end:validation_end]] = "validation"
    split_values[shuffled_positions[validation_end:]] = "test"
    return pd.Series(split_values, index=frame.index, name="split")


def make_grouped_split(
    frame: pd.DataFrame,
    group_column: str,
    fractions: SplitFractions,
    random_seed: int,
) -> pd.Series:
    """Create a split assignment that keeps every group in exactly one split."""
    fractions.validate()
    if group_column not in frame.columns:
        raise ValueError(f"Missing grouping column for grouped split: {group_column}")
    if frame[group_column].isna().any():
        raise ValueError(f"Grouping column contains missing values: {group_column}")

    groups = frame[group_column].astype(str)
    unique_groups = groups.drop_duplicates().to_numpy()
    if len(unique_groups) < 3:
        raise ValueError("At least three unique groups are required for a grouped split.")

    rng = np.random.default_rng(random_seed)
    shuffled_groups = rng.permutation(unique_groups)
    train_end = int(round(len(shuffled_groups) * fractions.train))
    validation_end = train_end + int(round(len(shuffled_groups) * fractions.validation))

    group_to_split: dict[str, str] = {}
    for group in shuffled_groups[:train_end]:
        group_to_split[str(group)] = "train"
    for group in shuffled_groups[train_end:validation_end]:
        group_to_split[str(group)] = "validation"
    for group in shuffled_groups[validation_end:]:
        group_to_split[str(group)] = "test"

    return groups.map(group_to_split).rename("split")


def make_balanced_grouped_split(
    frame: pd.DataFrame,
    group_column: str,
    fractions: SplitFractions,
    random_seed: int,
) -> pd.Series:
    """Create a grouped split that approximates target row counts.

    Groups are sorted by decreasing size and assigned greedily to the split that
    minimizes normalized squared deviation from target row counts. This is useful
    for organism-held-out splits where group sizes can be highly skewed.
    """
    fractions.validate()
    if group_column not in frame.columns:
        raise ValueError(f"Missing grouping column for grouped split: {group_column}")
    if frame[group_column].isna().any():
        raise ValueError(f"Grouping column contains missing values: {group_column}")

    groups = frame[group_column].astype(str)
    group_sizes = groups.value_counts(sort=False)
    if len(group_sizes) < 3:
        raise ValueError("At least three unique groups are required for a grouped split.")

    rng = np.random.default_rng(random_seed)
    group_table = pd.DataFrame(
        {
            "group": group_sizes.index.astype(str),
            "size": group_sizes.to_numpy(dtype=int),
            "tie_breaker": rng.random(len(group_sizes)),
        }
    ).sort_values(["size", "tie_breaker"], ascending=[False, True])

    targets = {
        "train": len(frame) * fractions.train,
        "validation": len(frame) * fractions.validation,
        "test": len(frame) * fractions.test,
    }
    current = {split_name: 0 for split_name in SPLIT_NAMES}
    group_to_split: dict[str, str] = {}

    for row in group_table.itertuples(index=False):
        group = str(row.group)
        size = int(row.size)
        best_split = min(
            SPLIT_NAMES,
            key=lambda split_name: _assignment_score(current, targets, split_name, size),
        )
        group_to_split[group] = best_split
        current[best_split] += size

    if any(current[split_name] == 0 for split_name in SPLIT_NAMES):
        raise ValueError(
            "Balanced grouped split produced an empty partition. "
            f"Observed counts: {current}"
        )

    return groups.map(group_to_split).rename("split")


def build_split_frame(
    frame: pd.DataFrame,
    split_assignments: pd.Series,
    strategy: str,
    id_column: str,
    sequence_column: str,
) -> pd.DataFrame:
    """Build a portable split-definition table."""
    if len(frame) != len(split_assignments):
        raise ValueError("Split assignments must have the same length as the input dataframe.")
    if id_column not in frame.columns:
        raise ValueError(f"Missing identifier column: {id_column}")
    if sequence_column not in frame.columns:
        raise ValueError(f"Missing sequence column: {sequence_column}")

    output = pd.DataFrame(
        {
            "row_index": frame.index.to_numpy(),
            id_column: frame[id_column].to_numpy(),
            "split": split_assignments.to_numpy(),
            "split_strategy": strategy,
            "sequence_sha256": frame[sequence_column].astype(str).map(stable_sequence_hash).to_numpy(),
        }
    )
    if output["split"].isna().any():
        raise ValueError(f"Split strategy '{strategy}' produced missing split assignments.")
    return output


def stable_sequence_hash(sequence: str) -> str:
    """Return a stable SHA-256 hash for a sequence string."""
    normalized = str(sequence).strip().upper()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_counts(split_frame: pd.DataFrame) -> dict[str, int]:
    """Return train/validation/test counts in stable order."""
    counts = split_frame["split"].value_counts().to_dict()
    return {split_name: int(counts.get(split_name, 0)) for split_name in SPLIT_NAMES}


def find_group_leakage(split_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return groups that appear in more than one split."""
    if group_column not in split_frame.columns:
        raise ValueError(f"Missing group column in split frame: {group_column}")
    leakage = (
        split_frame.groupby(group_column)["split"]
        .nunique()
        .reset_index(name="n_splits")
        .query("n_splits > 1")
        .sort_values(["n_splits", group_column], ascending=[False, True])
    )
    return leakage.reset_index(drop=True)


def _assignment_score(
    current: dict[str, int],
    targets: dict[str, float],
    candidate_split: str,
    group_size: int,
) -> float:
    candidate = current.copy()
    candidate[candidate_split] += group_size
    return float(
        sum(
            ((candidate[split_name] - targets[split_name]) / targets[split_name]) ** 2
            for split_name in SPLIT_NAMES
        )
    )


def write_split_summary(
    summaries: list[dict[str, object]],
    path: str | Path,
) -> None:
    """Write a Markdown summary for generated split definitions."""
    lines = [
        "# Split Summary",
        "",
        "Generated split definitions for reproducible experiments.",
        "",
        "| Strategy | Train | Validation | Test | Group leakage count | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        counts = summary["counts"]
        if not isinstance(counts, dict):
            raise TypeError("summary['counts'] must be a dictionary")
        lines.append(
            "| {strategy} | {train} | {validation} | {test} | {leakage} | {notes} |".format(
                strategy=summary["strategy"],
                train=counts.get("train", 0),
                validation=counts.get("validation", 0),
                test=counts.get("test", 0),
                leakage=summary.get("group_leakage_count", 0),
                notes=summary.get("notes", ""),
            )
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
