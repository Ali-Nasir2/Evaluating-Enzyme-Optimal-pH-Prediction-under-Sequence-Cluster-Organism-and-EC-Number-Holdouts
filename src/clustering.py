"""Sequence clustering helpers for leakage-controlled protein splits."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ClusteringCommand:
    """A clustering command and its expected assignment output."""

    tool: str
    identity_threshold: float
    command: str
    expected_raw_output: Path
    expected_assignment_output: Path


def threshold_label(identity_threshold: float) -> str:
    """Convert 0.9 to '90' for stable filenames."""
    return str(int(round(identity_threshold * 100)))


def fasta_record_id(row_index: int, accession: object | None = None) -> str:
    """Create a FASTA record id that preserves the source row index."""
    suffix = "" if accession is None or pd.isna(accession) else f"|{_sanitize_identifier(accession)}"
    return f"seq_{row_index}{suffix}"


def row_index_from_record_id(record_id: str) -> int:
    """Extract the original dataframe row index from a FASTA record id."""
    match = re.match(r"^seq_(\d+)(?:\||$)", str(record_id))
    if not match:
        raise ValueError(f"Cannot extract row index from FASTA record id: {record_id}")
    return int(match.group(1))


def write_fasta(
    frame: pd.DataFrame,
    sequence_column: str,
    id_column: str | None,
    output_path: str | Path,
) -> Path:
    """Write sequences to FASTA for external clustering tools."""
    if sequence_column not in frame.columns:
        raise ValueError(f"Missing sequence column: {sequence_column}")
    if id_column and id_column not in frame.columns:
        raise ValueError(f"Missing identifier column: {id_column}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row_index, row in frame.iterrows():
            accession = row[id_column] if id_column else None
            record_id = fasta_record_id(int(row_index), accession)
            sequence = str(row[sequence_column]).strip().upper()
            if not sequence:
                raise ValueError(f"Empty sequence at row index {row_index}")
            handle.write(f">{record_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    return path


def tool_available(binary_name: str) -> bool:
    """Return whether a clustering executable is available on PATH."""
    return shutil.which(binary_name) is not None


def build_mmseqs_commands(
    fasta_path: str | Path,
    clusters_dir: str | Path,
    thresholds: Iterable[float],
    binary: str = "mmseqs",
    coverage: float = 0.80,
    cov_mode: int = 0,
    threads: int = 4,
) -> list[ClusteringCommand]:
    """Build MMseqs2 easy-cluster commands."""
    fasta = Path(fasta_path)
    output_dir = Path(clusters_dir)
    commands: list[ClusteringCommand] = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        prefix = output_dir / f"mmseqs_identity_{label}"
        tmp_dir = output_dir / f"tmp_mmseqs_identity_{label}"
        raw_output = Path(f"{prefix}_cluster.tsv")
        assignment_output = output_dir / f"cluster_assignments_mmseqs_identity_{label}.csv"
        command = (
            f'{binary} easy-cluster "{fasta}" "{prefix}" "{tmp_dir}" '
            f"--min-seq-id {threshold:.2f} -c {coverage:.2f} --cov-mode {cov_mode} "
            f"--threads {threads}"
        )
        commands.append(
            ClusteringCommand(
                tool="mmseqs",
                identity_threshold=threshold,
                command=command,
                expected_raw_output=raw_output,
                expected_assignment_output=assignment_output,
            )
        )
    return commands


def build_cdhit_commands(
    fasta_path: str | Path,
    clusters_dir: str | Path,
    thresholds: Iterable[float],
    binary: str = "cd-hit",
    threads: int = 4,
) -> list[ClusteringCommand]:
    """Build CD-HIT commands."""
    fasta = Path(fasta_path)
    output_dir = Path(clusters_dir)
    commands: list[ClusteringCommand] = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        output_fasta = output_dir / f"cdhit_identity_{label}.fasta"
        raw_output = Path(f"{output_fasta}.clstr")
        assignment_output = output_dir / f"cluster_assignments_cdhit_identity_{label}.csv"
        word_size = _cdhit_word_size(threshold)
        command = (
            f'{binary} -i "{fasta}" -o "{output_fasta}" -c {threshold:.2f} '
            f"-n {word_size} -d 0 -T {threads} -M 0"
        )
        commands.append(
            ClusteringCommand(
                tool="cd-hit",
                identity_threshold=threshold,
                command=command,
                expected_raw_output=raw_output,
                expected_assignment_output=assignment_output,
            )
        )
    return commands


def write_clustering_command_file(
    commands: Iterable[ClusteringCommand],
    output_path: str | Path,
) -> Path:
    """Write PowerShell commands for external clustering."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Run the commands for one clustering tool after installing MMseqs2 or CD-HIT.",
        "# Do not treat this file as evidence that clustering has already completed.",
        "# Recommended with conda/mamba:",
        "#   conda install -c bioconda -c conda-forge mmseqs2 cd-hit",
        "",
    ]
    for command in commands:
        lines.extend(
            [
                f"# {command.tool} at {threshold_label(command.identity_threshold)}% identity",
                command.command,
                (
                    "# Expected raw output: "
                    f"{command.expected_raw_output.as_posix()}"
                ),
                (
                    "# Parse to assignments with: "
                    f"python scripts/cluster_sequences.py --config config/default.yaml "
                    f"--parse-{command.tool.replace('-', '')} "
                    f"\"{command.expected_raw_output.as_posix()}\" "
                    f"--identity {command.identity_threshold:.2f}"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_mmseqs_cluster_tsv(path: str | Path, identity_threshold: float) -> pd.DataFrame:
    """Parse an MMseqs2 cluster TSV file into record-level assignments."""
    frame = pd.read_csv(path, sep="\t", header=None, names=["representative_id", "record_id"])
    if frame.empty:
        raise ValueError(f"MMseqs2 cluster TSV is empty: {path}")
    frame["row_index"] = frame["record_id"].map(row_index_from_record_id)
    frame["cluster_id"] = "mmseqs_" + frame["representative_id"].astype(str)
    frame["algorithm"] = "mmseqs"
    frame["identity_threshold"] = identity_threshold
    return frame[
        ["row_index", "record_id", "cluster_id", "representative_id", "algorithm", "identity_threshold"]
    ].drop_duplicates("row_index")


def parse_cdhit_clstr(path: str | Path, identity_threshold: float) -> pd.DataFrame:
    """Parse a CD-HIT .clstr file into record-level assignments."""
    rows: list[dict[str, object]] = []
    current_cluster: str | None = None
    representative_id: str | None = None
    pending_members: list[str] = []

    def flush_cluster() -> None:
        if current_cluster is None:
            return
        rep = representative_id or (pending_members[0] if pending_members else current_cluster)
        for record_id in pending_members:
            rows.append(
                {
                    "row_index": row_index_from_record_id(record_id),
                    "record_id": record_id,
                    "cluster_id": f"cdhit_{current_cluster}",
                    "representative_id": rep,
                    "algorithm": "cd-hit",
                    "identity_threshold": identity_threshold,
                }
            )

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">Cluster"):
            flush_cluster()
            current_cluster = line.removeprefix(">").replace(" ", "_")
            representative_id = None
            pending_members = []
            continue

        match = re.search(r">(.+?)\.\.\.", line)
        if not match:
            raise ValueError(f"Could not parse CD-HIT cluster member line: {line}")
        record_id = match.group(1)
        pending_members.append(record_id)
        if line.endswith("*"):
            representative_id = record_id

    flush_cluster()
    if not rows:
        raise ValueError(f"No CD-HIT cluster assignments parsed from {path}")
    return pd.DataFrame.from_records(rows).drop_duplicates("row_index")


def validate_cluster_assignments(assignments: pd.DataFrame, expected_rows: int) -> None:
    """Validate that clustering produced exactly one assignment per input row."""
    required = {"row_index", "record_id", "cluster_id", "representative_id"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Cluster assignments are missing columns: {sorted(missing)}")
    if assignments["row_index"].duplicated().any():
        raise ValueError("Cluster assignments contain duplicate row_index values.")
    observed = set(assignments["row_index"].astype(int))
    expected = set(range(expected_rows))
    missing_rows = sorted(expected.difference(observed))
    extra_rows = sorted(observed.difference(expected))
    if missing_rows or extra_rows:
        raise ValueError(
            "Cluster assignments do not match the cleaned dataset rows. "
            f"Missing rows: {missing_rows[:10]} Extra rows: {extra_rows[:10]}"
        )


def _cdhit_word_size(identity_threshold: float) -> int:
    if identity_threshold >= 0.70:
        return 5
    if identity_threshold >= 0.60:
        return 4
    if identity_threshold >= 0.50:
        return 3
    return 2


def _sanitize_identifier(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value))
