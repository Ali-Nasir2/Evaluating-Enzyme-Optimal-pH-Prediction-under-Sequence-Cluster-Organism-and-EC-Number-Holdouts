"""Prepare and parse external sequence clustering outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.clustering import (
    build_cdhit_commands,
    build_mmseqs_commands,
    parse_cdhit_clstr,
    parse_mmseqs_cluster_tsv,
    threshold_label,
    tool_available,
    validate_cluster_assignments,
    write_clustering_command_file,
    write_fasta,
)
from src.utils import configure_logging, ensure_directory, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--parse-mmseqs",
        metavar="PATH",
        help="Parse an MMseqs2 *_cluster.tsv file into cluster assignments.",
    )
    parser.add_argument(
        "--parse-cdhit",
        metavar="PATH",
        help="Parse a CD-HIT .clstr file into cluster assignments.",
    )
    parser.add_argument(
        "--identity",
        type=float,
        help="Identity threshold for the parsed clustering output, e.g. 0.90.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
    paths = config["paths"]
    dataset_config = config["dataset"]
    clustering_config = config["clustering"]
    thresholds = [float(value) for value in config["splitting"]["cluster_identity_thresholds"]]

    cleaned_path = Path(paths["cleaned_dataset"])
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run scripts/prepare_data.py first."
        )

    frame = pd.read_csv(cleaned_path)
    clusters_dir = ensure_directory(paths["clusters_dir"])
    fasta_path = write_fasta(
        frame=frame,
        sequence_column=dataset_config["sequence_column"],
        id_column=dataset_config.get("id_column"),
        output_path=paths["sequences_fasta"],
    )
    LOGGER.info("Wrote FASTA for clustering to %s", fasta_path)

    if args.parse_mmseqs or args.parse_cdhit:
        if args.parse_mmseqs and args.parse_cdhit:
            raise ValueError("Choose only one parser: --parse-mmseqs or --parse-cdhit.")
        if args.identity is None:
            raise ValueError("--identity is required when parsing clustering output.")
        assignments = (
            parse_mmseqs_cluster_tsv(args.parse_mmseqs, args.identity)
            if args.parse_mmseqs
            else parse_cdhit_clstr(args.parse_cdhit, args.identity)
        )
        validate_cluster_assignments(assignments, expected_rows=len(frame))
        tool_name = "mmseqs" if args.parse_mmseqs else "cdhit"
        output_path = (
            clusters_dir
            / f"cluster_assignments_{tool_name}_identity_{threshold_label(args.identity)}.csv"
        )
        assignments.to_csv(output_path, index=False)
        LOGGER.info("Wrote cluster assignments to %s", output_path)
        return

    mmseqs_binary = clustering_config.get("mmseqs_binary", "mmseqs")
    cdhit_binary = clustering_config.get("cdhit_binary", "cd-hit")
    commands = build_mmseqs_commands(
        fasta_path=fasta_path,
        clusters_dir=clusters_dir,
        thresholds=thresholds,
        binary=mmseqs_binary,
        coverage=float(clustering_config.get("coverage", 0.80)),
        cov_mode=int(clustering_config.get("cov_mode", 0)),
        threads=int(clustering_config.get("threads", 4)),
    ) + build_cdhit_commands(
        fasta_path=fasta_path,
        clusters_dir=clusters_dir,
        thresholds=thresholds,
        binary=cdhit_binary,
        threads=int(clustering_config.get("threads", 4)),
    )
    command_file = write_clustering_command_file(
        commands,
        clusters_dir / "run_clustering_commands.ps1",
    )

    LOGGER.info("MMseqs2 available: %s", tool_available(mmseqs_binary))
    LOGGER.info("CD-HIT available: %s", tool_available(cdhit_binary))
    LOGGER.info("Wrote external clustering commands to %s", command_file)
    LOGGER.info(
        "Install and run a clustering tool, then parse its output with this script before "
        "creating cluster-based splits."
    )


if __name__ == "__main__":
    main()
