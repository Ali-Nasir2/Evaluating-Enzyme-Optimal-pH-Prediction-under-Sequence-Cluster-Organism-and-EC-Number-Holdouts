"""Strict validation and quality reporting for enzyme optimal-pH datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AMINO_ACIDS = set("XBZJUO")


@dataclass
class ValidationConfig:
    """Column names and thresholds used for dataset validation."""

    sequence_column: str = "sequence"
    target_column: str = "optimal_ph"
    id_column: str = "protein_id"
    organism_column: str = "organism"
    taxonomy_column: str = "taxonomy"
    ec_column: str = "ec_number"
    min_sequence_length: int = 30
    max_sequence_length: int = 5000
    min_ph: float = 0.0
    max_ph: float = 14.0
    allowed_amino_acids: set[str] = field(default_factory=lambda: set(STANDARD_AMINO_ACIDS))
    ambiguous_amino_acids: set[str] = field(default_factory=lambda: set(AMBIGUOUS_AMINO_ACIDS))
    drop_invalid_sequences: bool = True
    drop_missing_targets: bool = True
    drop_out_of_range_targets: bool = True


@dataclass
class DataQualityReport:
    """Summary of validation results and cleaning decisions."""

    original_records: int
    cleaned_records: int
    removed_missing_sequences: int
    removed_missing_targets: int
    removed_invalid_sequences: int
    removed_out_of_range_targets: int
    duplicate_identifier_count: int
    duplicate_sequence_count: int
    conflicting_duplicate_sequence_labels: int
    ambiguous_sequence_count: int
    invalid_sequence_count: int
    short_sequence_count: int
    long_sequence_count: int
    sequence_length_min: int | None
    sequence_length_median: float | None
    sequence_length_mean: float | None
    sequence_length_max: int | None
    target_min: float | None
    target_median: float | None
    target_mean: float | None
    target_max: float | None
    organism_available: bool
    organism_unique_count: int | None
    ec_available: bool
    ec_unique_count: int | None
    taxonomy_available: bool
    available_columns: list[str]
    missing_values: dict[str, int]
    organism_top_counts: dict[str, int]
    ec_top_counts: dict[str, int]

    def to_markdown(self) -> str:
        """Render the report as Markdown."""
        lines = [
            "# Data Quality Report",
            "",
            "## Record Counts",
            "",
            f"- Original records: {self.original_records}",
            f"- Cleaned records: {self.cleaned_records}",
            f"- Removed missing sequences: {self.removed_missing_sequences}",
            f"- Removed missing targets: {self.removed_missing_targets}",
            f"- Removed invalid sequences: {self.removed_invalid_sequences}",
            f"- Removed out-of-range pH targets: {self.removed_out_of_range_targets}",
            "",
            "## Columns",
            "",
            ", ".join(self.available_columns) if self.available_columns else "[No columns detected]",
            "",
            "## Missing Values",
            "",
        ]
        for column, count in self.missing_values.items():
            lines.append(f"- {column}: {count}")

        lines.extend(
            [
                "",
                "## Sequence Quality",
                "",
                f"- Duplicate protein identifiers: {self.duplicate_identifier_count}",
                f"- Exact duplicate sequences: {self.duplicate_sequence_count}",
                (
                    "- Duplicate sequences with conflicting pH labels: "
                    f"{self.conflicting_duplicate_sequence_labels}"
                ),
                f"- Sequences containing ambiguous residues: {self.ambiguous_sequence_count}",
                f"- Sequences containing invalid symbols: {self.invalid_sequence_count}",
                f"- Extremely short sequences: {self.short_sequence_count}",
                f"- Extremely long sequences: {self.long_sequence_count}",
                "",
                "## Sequence-Length Distribution",
                "",
                f"- Minimum: {self.sequence_length_min}",
                f"- Median: {self.sequence_length_median}",
                f"- Mean: {self.sequence_length_mean}",
                f"- Maximum: {self.sequence_length_max}",
                "",
                "## Target-pH Distribution",
                "",
                f"- Minimum: {self.target_min}",
                f"- Median: {self.target_median}",
                f"- Mean: {self.target_mean}",
                f"- Maximum: {self.target_max}",
                "",
                "## Metadata Availability",
                "",
                f"- Organism column available: {self.organism_available}",
                f"- Unique organisms: {self.organism_unique_count}",
                f"- Taxonomy column available: {self.taxonomy_available}",
                f"- EC-number column available: {self.ec_available}",
                f"- Unique EC numbers: {self.ec_unique_count}",
                "",
                "## Top Organisms",
                "",
            ]
        )
        if self.organism_top_counts:
            for value, count in self.organism_top_counts.items():
                lines.append(f"- {value}: {count}")
        else:
            lines.append("[No organism metadata available]")

        lines.extend(["", "## Top EC Numbers", ""])
        if self.ec_top_counts:
            for value, count in self.ec_top_counts.items():
                lines.append(f"- {value}: {count}")
        else:
            lines.append("[No EC-number metadata available]")

        lines.extend(
            [
                "",
                "## Sequence Clusters",
                "",
                "[Not computed in Phase 1. Sequence clustering will be added in a later phase.]",
            ]
        )
        return "\n".join(lines) + "\n"


def validation_config_from_project_config(config: dict[str, Any]) -> ValidationConfig:
    """Build a ValidationConfig from the project YAML config."""
    dataset = config.get("dataset", {})
    validation = config.get("validation", {})
    return ValidationConfig(
        sequence_column=dataset.get("sequence_column", "sequence"),
        target_column=dataset.get("target_column", "optimal_ph"),
        id_column=dataset.get("id_column", "protein_id"),
        organism_column=dataset.get("organism_column", "organism"),
        taxonomy_column=dataset.get("taxonomy_column", "taxonomy"),
        ec_column=dataset.get("ec_column", "ec_number"),
        min_sequence_length=int(validation.get("min_sequence_length", 30)),
        max_sequence_length=int(validation.get("max_sequence_length", 5000)),
        min_ph=float(validation.get("min_ph", 0.0)),
        max_ph=float(validation.get("max_ph", 14.0)),
        allowed_amino_acids=set(validation.get("allowed_amino_acids", "ACDEFGHIKLMNPQRSTVWY")),
        ambiguous_amino_acids=set(validation.get("ambiguous_amino_acids", "XBZJUO")),
        drop_invalid_sequences=bool(validation.get("drop_invalid_sequences", True)),
        drop_missing_targets=bool(validation.get("drop_missing_targets", True)),
        drop_out_of_range_targets=bool(validation.get("drop_out_of_range_targets", True)),
    )


def normalize_sequence(value: object) -> str | None:
    """Normalize a sequence value or return None for missing values."""
    if pd.isna(value):
        return None
    sequence = str(value).strip().upper().replace(" ", "").replace("\n", "")
    return sequence or None


def sequence_symbol_sets(sequence: str, config: ValidationConfig) -> tuple[set[str], set[str]]:
    """Return ambiguous and invalid symbols present in a sequence."""
    symbols = set(sequence)
    ambiguous = symbols.intersection(config.ambiguous_amino_acids)
    invalid = symbols.difference(config.allowed_amino_acids).difference(config.ambiguous_amino_acids)
    return ambiguous, invalid


def validate_dataset(
    frame: pd.DataFrame, config: ValidationConfig
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and clean a loaded dataset.

    The returned dataframe is a copy. Raw input data is not modified.
    """
    required = [config.sequence_column, config.target_column]
    missing_required = [column for column in required if column not in frame.columns]
    if missing_required:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Missing required dataset columns: {missing_required}. Available columns: {available}"
        )

    working = frame.copy()
    original_records = len(working)
    working[config.sequence_column] = working[config.sequence_column].map(normalize_sequence)
    working[config.target_column] = pd.to_numeric(working[config.target_column], errors="coerce")

    missing_sequence_mask = working[config.sequence_column].isna()
    missing_target_mask = working[config.target_column].isna()

    sequences = working[config.sequence_column]
    lengths = sequences.dropna().map(len)
    short_sequence_mask = sequences.map(lambda seq: len(seq) < config.min_sequence_length if seq else False)
    long_sequence_mask = sequences.map(lambda seq: len(seq) > config.max_sequence_length if seq else False)

    ambiguous_mask = sequences.map(
        lambda seq: bool(sequence_symbol_sets(seq, config)[0]) if seq else False
    )
    invalid_symbol_mask = sequences.map(
        lambda seq: bool(sequence_symbol_sets(seq, config)[1]) if seq else False
    )

    out_of_range_target_mask = (
        working[config.target_column].notna()
        & (
            (working[config.target_column] < config.min_ph)
            | (working[config.target_column] > config.max_ph)
        )
    )

    duplicate_identifier_count = 0
    if config.id_column in working.columns:
        duplicate_identifier_count = int(
            working[config.id_column].notna().sum()
            - working.loc[working[config.id_column].notna(), config.id_column].nunique()
        )

    duplicate_sequence_count = int(
        working[config.sequence_column].notna().sum()
        - working.loc[working[config.sequence_column].notna(), config.sequence_column].nunique()
    )

    conflicting_duplicate_sequence_labels = _count_conflicting_duplicate_labels(
        working, config.sequence_column, config.target_column
    )

    removal_mask = missing_sequence_mask.copy()
    if config.drop_missing_targets:
        removal_mask = removal_mask | missing_target_mask
    invalid_sequence_mask = invalid_symbol_mask | short_sequence_mask | long_sequence_mask
    if config.drop_invalid_sequences:
        removal_mask = removal_mask | invalid_sequence_mask
    if config.drop_out_of_range_targets:
        removal_mask = removal_mask | out_of_range_target_mask

    cleaned = working.loc[~removal_mask].reset_index(drop=True)

    cleaned_lengths = cleaned[config.sequence_column].map(len) if not cleaned.empty else pd.Series(dtype=float)
    cleaned_targets = cleaned[config.target_column] if not cleaned.empty else pd.Series(dtype=float)

    report = DataQualityReport(
        original_records=original_records,
        cleaned_records=len(cleaned),
        removed_missing_sequences=int(missing_sequence_mask.sum()),
        removed_missing_targets=int((missing_target_mask & removal_mask).sum()),
        removed_invalid_sequences=int((invalid_sequence_mask & removal_mask).sum()),
        removed_out_of_range_targets=int((out_of_range_target_mask & removal_mask).sum()),
        duplicate_identifier_count=duplicate_identifier_count,
        duplicate_sequence_count=duplicate_sequence_count,
        conflicting_duplicate_sequence_labels=conflicting_duplicate_sequence_labels,
        ambiguous_sequence_count=int(ambiguous_mask.sum()),
        invalid_sequence_count=int(invalid_symbol_mask.sum()),
        short_sequence_count=int(short_sequence_mask.sum()),
        long_sequence_count=int(long_sequence_mask.sum()),
        sequence_length_min=_safe_int(cleaned_lengths.min()),
        sequence_length_median=_safe_float(cleaned_lengths.median()),
        sequence_length_mean=_safe_float(cleaned_lengths.mean()),
        sequence_length_max=_safe_int(cleaned_lengths.max()),
        target_min=_safe_float(cleaned_targets.min()),
        target_median=_safe_float(cleaned_targets.median()),
        target_mean=_safe_float(cleaned_targets.mean()),
        target_max=_safe_float(cleaned_targets.max()),
        organism_available=config.organism_column in cleaned.columns,
        organism_unique_count=_unique_count(cleaned, config.organism_column),
        ec_available=config.ec_column in cleaned.columns,
        ec_unique_count=_unique_count(cleaned, config.ec_column),
        taxonomy_available=config.taxonomy_column in cleaned.columns,
        available_columns=list(map(str, working.columns)),
        missing_values={str(column): int(count) for column, count in working.isna().sum().items()},
        organism_top_counts=_top_counts(cleaned, config.organism_column),
        ec_top_counts=_top_counts(cleaned, config.ec_column),
    )
    return cleaned, report


def write_quality_report(report: DataQualityReport, path: str | Path) -> None:
    """Write a Markdown data-quality report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_markdown(), encoding="utf-8")


def _count_conflicting_duplicate_labels(
    frame: pd.DataFrame, sequence_column: str, target_column: str
) -> int:
    grouped = (
        frame.dropna(subset=[sequence_column, target_column])
        .groupby(sequence_column)[target_column]
        .nunique(dropna=True)
    )
    return int((grouped > 1).sum())


def _unique_count(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].dropna().nunique())


def _top_counts(frame: pd.DataFrame, column: str, limit: int = 10) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].dropna().astype(str).value_counts().head(limit)
    return {str(index): int(value) for index, value in counts.items()}


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(np.round(float(value), 4))


def _safe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)

