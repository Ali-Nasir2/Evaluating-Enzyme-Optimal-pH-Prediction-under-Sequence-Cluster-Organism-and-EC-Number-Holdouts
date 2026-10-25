"""Audit nearest train-sequence identity for cluster-disjoint test partitions."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output", default="outputs/metrics/cluster_similarity_audit.csv")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[50, 70, 90])
    return parser.parse_args()


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row_index, row in frame.iterrows():
            handle.write(f">{row_index}|{row['Accession']}\n{row['Sequence']}\n")


def main() -> None:
    args = parse_args()
    binary = shutil.which(args.mmseqs)
    if binary is None:
        raise FileNotFoundError(
            "MMseqs2 is required for the similarity audit. Install it or pass --mmseqs "
            "with the executable path; cluster IDs alone cannot establish nearest-neighbor identity."
        )
    version = subprocess.run([binary, "version"], check=True, capture_output=True, text=True).stdout.strip()
    data = pd.read_csv(args.data)
    rows: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="mmseqs_similarity_audit_") as temporary:
        root = Path(temporary)
        for threshold in args.thresholds:
            split_path = Path(args.splits_dir) / f"cluster_safe_mmseqs_identity_{threshold}_split.csv"
            split = pd.read_csv(split_path)
            train_indices = set(split.loc[split["split"].eq("train"), "row_index"])
            test_indices = set(split.loc[split["split"].eq("test"), "row_index"])
            train = data.loc[sorted(train_indices)]
            test = data.loc[sorted(test_indices)]
            train_fasta = root / f"train_{threshold}.faa"
            test_fasta = root / f"test_{threshold}.faa"
            output_prefix = root / f"search_{threshold}"
            write_fasta(train, train_fasta)
            write_fasta(test, test_fasta)
            command = [
                binary,
                "easy-search",
                str(test_fasta),
                str(train_fasta),
                str(output_prefix),
                str(root / f"tmp_{threshold}"),
                "--min-seq-id",
                "0",
                "-c",
                "0",
                "--cov-mode",
                "0",
                "--max-seqs",
                "1",
                "--format-output",
                "query,target,pident,qcov,tcov",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            search_output = Path(f"{output_prefix}.m8")
            result = pd.read_csv(
                search_output,
                sep="\t",
                names=["query", "target", "pident", "query_coverage", "target_coverage"],
            )
            result["identity_threshold"] = threshold / 100
            result["mmseqs_version"] = version
            result["command"] = " ".join(command)
            result["exceeds_cluster_threshold"] = result["pident"] >= threshold
            rows.append(result)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
