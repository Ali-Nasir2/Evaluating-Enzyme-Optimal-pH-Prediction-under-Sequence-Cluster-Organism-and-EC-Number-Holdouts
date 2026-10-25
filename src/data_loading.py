"""Dataset discovery and loading utilities.

The loader accepts common formats used for enzyme metadata and sequences. It does
not infer missing biological labels from FASTA headers; if optimal-pH values are
not present, validation will report the missing target column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".fasta",
    ".fa",
    ".faa",
    ".fas",
}


@dataclass(frozen=True)
class LoadedDataset:
    """A loaded dataset and minimal source metadata."""

    frame: pd.DataFrame
    source_path: Path
    source_format: str


def find_dataset_files(raw_data_dir: str | Path) -> list[Path]:
    """Return supported dataset files in a raw-data directory."""
    directory = Path(raw_data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def resolve_dataset_path(raw_data_dir: str | Path, configured_file: str | None = None) -> Path:
    """Resolve the dataset path from config or by discovering a single raw file."""
    if configured_file:
        path = Path(configured_file)
        if not path.is_absolute():
            path = Path(raw_data_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Configured dataset file does not exist: {path}")
        return path

    files = find_dataset_files(raw_data_dir)
    if not files:
        raise FileNotFoundError(
            "No supported dataset file found in data/raw/. Place a real dataset there "
            "or set dataset.file in config/default.yaml."
        )
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise ValueError(
            "Multiple supported dataset files were found. Set dataset.file in "
            f"config/default.yaml to choose one. Found: {names}"
        )
    return files[0]


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load a dataset from CSV, TSV, JSON, Excel, or FASTA."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported dataset extension '{suffix}' for {dataset_path}")

    if suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(dataset_path, sep="\t")
    elif suffix == ".json":
        frame = pd.read_json(dataset_path)
    elif suffix == ".jsonl":
        frame = pd.read_json(dataset_path, lines=True)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(dataset_path)
    elif suffix in {".fasta", ".fa", ".faa", ".fas"}:
        frame = load_fasta(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset extension '{suffix}' for {dataset_path}")

    if frame.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return LoadedDataset(frame=frame, source_path=dataset_path, source_format=suffix.lstrip("."))


def load_fasta(path: str | Path) -> pd.DataFrame:
    """Load a FASTA file into `protein_id`, `description`, and `sequence` columns."""
    records: list[dict[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []

    def flush_record() -> None:
        if current_header is None:
            return
        identifier = current_header.split()[0]
        records.append(
            {
                "protein_id": identifier,
                "description": current_header,
                "sequence": "".join(current_sequence),
            }
        )

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                current_header = line[1:].strip()
                current_sequence = []
            elif current_header is None:
                raise ValueError(f"FASTA sequence encountered before header at line {line_number}")
            else:
                current_sequence.append(line)
    flush_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame.from_records(records)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a helpful error if required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing}. Available columns: {available}"
        )

