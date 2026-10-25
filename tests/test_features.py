from __future__ import annotations

import math

import pandas as pd

from src.sequence_features import (
    AMINO_ACIDS,
    build_feature_table,
    compute_sequence_features,
    feature_column_names,
)


def test_compute_sequence_features_has_all_amino_acid_fractions():
    sequence = "".join(AMINO_ACIDS)

    features = compute_sequence_features(sequence)

    assert features["length"] == 20.0
    assert features["standard_residue_fraction"] == 1.0
    assert features["acidic_fraction"] == 2 / 20
    assert features["basic_fraction"] == 3 / 20
    assert math.isclose(features["net_charge_proxy"], 1 / 20)
    for amino_acid in AMINO_ACIDS:
        assert features[f"aa_{amino_acid}_fraction"] == 1 / 20


def test_compute_sequence_features_reports_ambiguous_fraction():
    features = compute_sequence_features("ACDX")

    assert features["length"] == 4.0
    assert features["ambiguous_residue_fraction"] == 0.25
    assert math.isnan(features["molecular_weight"])


def test_build_feature_table_preserves_metadata():
    frame = pd.DataFrame(
        {
            "Accession": ["p1"],
            "pHopt": [7.0],
            "Split": ["Training"],
            "Sequence": ["ACDEFGHIKLMNPQRSTVWY"],
        }
    )

    features = build_feature_table(
        frame,
        sequence_column="Sequence",
        metadata_columns=["Accession", "pHopt", "Split"],
    )

    assert features.loc[0, "Accession"] == "p1"
    assert features.loc[0, "pHopt"] == 7.0
    assert "aa_A_fraction" in features.columns


def test_feature_column_names_excludes_metadata_and_target():
    columns = ["row_index", "Unnamed: 0", "Accession", "pHopt", "Split", "length", "aa_A_fraction"]

    assert feature_column_names(columns) == ["length", "aa_A_fraction"]
