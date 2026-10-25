from __future__ import annotations

import pandas as pd
import pytest

from src.data_loading import find_dataset_files, load_dataset, require_columns, resolve_dataset_path


def test_load_csv_dataset(tmp_path):
    path = tmp_path / "enzymes.csv"
    frame = pd.DataFrame(
        {
            "protein_id": ["p1"],
            "sequence": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
            "optimal_ph": [7.0],
        }
    )
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded.source_format == "csv"
    assert loaded.frame.shape == (1, 3)
    assert loaded.frame.loc[0, "protein_id"] == "p1"


def test_load_fasta_dataset(tmp_path):
    path = tmp_path / "enzymes.fasta"
    path.write_text(
        ">p1 protein one\nACDEFGHIKLMNPQRSTVWY\nACDEFGHIKLMNPQRSTVWY\n",
        encoding="utf-8",
    )

    loaded = load_dataset(path)

    assert loaded.frame.loc[0, "protein_id"] == "p1"
    assert loaded.frame.loc[0, "sequence"] == "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


def test_find_dataset_files_filters_supported_files(tmp_path):
    (tmp_path / "data.csv").write_text("sequence,optimal_ph\nAAA,7\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")

    assert find_dataset_files(tmp_path) == [tmp_path / "data.csv"]


def test_resolve_dataset_path_requires_choice_for_multiple_files(tmp_path):
    (tmp_path / "a.csv").write_text("sequence,optimal_ph\nAAA,7\n", encoding="utf-8")
    (tmp_path / "b.tsv").write_text("sequence\toptimal_ph\nAAA\t7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple supported dataset files"):
        resolve_dataset_path(tmp_path)


def test_require_columns_reports_missing_columns():
    frame = pd.DataFrame({"sequence": ["ACD"]})

    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(frame, ["sequence", "optimal_ph"])

