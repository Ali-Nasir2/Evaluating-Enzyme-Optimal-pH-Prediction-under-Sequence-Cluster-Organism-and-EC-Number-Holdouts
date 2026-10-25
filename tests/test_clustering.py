from __future__ import annotations

import pandas as pd
import pytest

from src.clustering import (
    build_cdhit_commands,
    build_mmseqs_commands,
    parse_cdhit_clstr,
    parse_mmseqs_cluster_tsv,
    row_index_from_record_id,
    threshold_label,
    validate_cluster_assignments,
    write_fasta,
)


def test_threshold_label():
    assert threshold_label(0.9) == "90"
    assert threshold_label(0.7) == "70"


def test_write_fasta_uses_row_index_record_ids(tmp_path):
    frame = pd.DataFrame(
        {
            "Accession": ["P 1", "P2"],
            "Sequence": ["ACDE", "FGHI"],
        }
    )
    output = write_fasta(frame, "Sequence", "Accession", tmp_path / "seqs.fasta")

    text = output.read_text(encoding="utf-8")

    assert ">seq_0|P_1" in text
    assert ">seq_1|P2" in text
    assert "ACDE" in text


def test_row_index_from_record_id():
    assert row_index_from_record_id("seq_123|P21673") == 123

    with pytest.raises(ValueError, match="Cannot extract row index"):
        row_index_from_record_id("P21673")


def test_parse_mmseqs_cluster_tsv(tmp_path):
    path = tmp_path / "clusters.tsv"
    path.write_text(
        "seq_0|A\tseq_0|A\nseq_0|A\tseq_1|B\nseq_2|C\tseq_2|C\n",
        encoding="utf-8",
    )

    assignments = parse_mmseqs_cluster_tsv(path, identity_threshold=0.9)

    assert assignments.shape[0] == 3
    assert set(assignments["row_index"]) == {0, 1, 2}
    assert assignments.loc[assignments["row_index"] == 1, "cluster_id"].item() == "mmseqs_seq_0|A"


def test_parse_cdhit_clstr(tmp_path):
    path = tmp_path / "clusters.clstr"
    path.write_text(
        ">Cluster 0\n"
        "0 100aa, >seq_0|A... *\n"
        "1 100aa, >seq_1|B... at 95%\n"
        ">Cluster 1\n"
        "0 100aa, >seq_2|C... *\n",
        encoding="utf-8",
    )

    assignments = parse_cdhit_clstr(path, identity_threshold=0.9)

    assert assignments.shape[0] == 3
    assert set(assignments["cluster_id"]) == {"cdhit_Cluster_0", "cdhit_Cluster_1"}


def test_validate_cluster_assignments_requires_all_rows():
    assignments = pd.DataFrame(
        {
            "row_index": [0],
            "record_id": ["seq_0|A"],
            "cluster_id": ["c0"],
            "representative_id": ["seq_0|A"],
        }
    )

    with pytest.raises(ValueError, match="do not match"):
        validate_cluster_assignments(assignments, expected_rows=2)


def test_command_builders_point_to_expected_outputs(tmp_path):
    mmseqs = build_mmseqs_commands(tmp_path / "seqs.fasta", tmp_path, [0.9])
    cdhit = build_cdhit_commands(tmp_path / "seqs.fasta", tmp_path, [0.9])

    assert "--min-seq-id 0.90" in mmseqs[0].command
    assert mmseqs[0].expected_raw_output.name == "mmseqs_identity_90_cluster.tsv"
    assert "-c 0.90" in cdhit[0].command
    assert cdhit[0].expected_raw_output.name == "cdhit_identity_90.fasta.clstr"

