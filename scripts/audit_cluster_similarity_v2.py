"""Audit nearest training-sequence similarity without overwriting prior outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmseqs", required=True)
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    binary = str(Path(args.mmseqs).resolve())
    if not Path(binary).exists():
        raise FileNotFoundError(binary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    version = subprocess.run([binary, "version"], check=True, capture_output=True, text=True).stdout.strip()
    thresholds = [50, 70, 90]
    records: list[pd.DataFrame] = []
    commands: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="mmseqs_similarity_audit_") as temporary:
        root = Path(temporary)
        for threshold in thresholds:
            split_path = Path(args.splits_dir) / f"cluster_safe_mmseqs_identity_{threshold}_split.csv"
            split = pd.read_csv(split_path)
            split.to_csv(output_dir / f"partition_assignments_{threshold}.csv", index=False)
            train = data.loc[sorted(set(split.loc[split["split"].eq("train"), "row_index"]))]
            test = data.loc[sorted(set(split.loc[split["split"].eq("test"), "row_index"]))]
            train_fasta = root / f"train_{threshold}.faa"
            test_fasta = root / f"test_{threshold}.faa"
            train_fasta.write_text(
                "".join(f">{i}|{row['Accession']}\n{row['Sequence']}\n" for i, row in train.iterrows()),
                encoding="utf-8",
            )
            test_fasta.write_text(
                "".join(f">{i}|{row['Accession']}\n{row['Sequence']}\n" for i, row in test.iterrows()),
                encoding="utf-8",
            )
            prefix = root / f"search_{threshold}"
            command = [
                binary, "easy-search", str(test_fasta), str(train_fasta), str(prefix),
                str(root / f"tmp_{threshold}"), "--min-seq-id", "0", "-c", "0",
                "--cov-mode", "0", "--max-seqs", "1",
                "--format-output", "query,target,pident,qcov,tcov",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            candidates = [Path(f"{prefix}.m8"), prefix]
            search_output = next((path for path in candidates if path.exists()), None)
            if search_output is None:
                raise FileNotFoundError(f"MMseqs2 produced no search output for {threshold}%: {candidates}")
            result = pd.read_csv(
                search_output,
                sep="\t",
                names=["query", "target", "pident", "query_coverage", "target_coverage"],
            )
            expected_queries = pd.DataFrame({"query": [f"{i}|{row['Accession']}" for i, row in test.iterrows()]})
            result = expected_queries.merge(result, on="query", how="left")
            result["hit_found"] = result["target"].notna()
            result["identity_threshold"] = threshold / 100
            result["identity_threshold_percent"] = threshold
            result["mmseqs_version"] = version
            result["command"] = subprocess.list2cmdline(command)
            result["exceeds_cluster_threshold"] = result["pident"] >= threshold
            records.append(result)
            commands.append(
                {
                    "threshold": threshold / 100,
                    "command": subprocess.list2cmdline(command),
                    "train_count": len(train),
                    "test_count": len(test),
                }
            )

    pd.concat(records, ignore_index=True).to_csv(output_dir / "nearest_training_hits.csv", index=False)
    (output_dir / "audit_metadata.json").write_text(
        json.dumps(
            {
                "mmseqs_binary": binary,
                "mmseqs_version": version,
                "settings": {
                    "search": "easy-search",
                    "min_seq_id": 0,
                    "coverage": 0,
                    "cov_mode": 0,
                    "max_seqs": 1,
                    "format_output": "query,target,pident,qcov,tcov",
                    "thresholds": thresholds,
                },
                "commands": commands,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
