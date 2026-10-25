from __future__ import annotations

import pandas as pd
import pytest

from src.data_validation import (
    ValidationConfig,
    normalize_sequence,
    sequence_symbol_sets,
    validate_dataset,
)


VALID_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


def test_normalize_sequence_uppercases_and_removes_whitespace():
    assert normalize_sequence(" acd ef\n") == "ACDEF"


def test_sequence_symbol_sets_reports_ambiguous_and_invalid_symbols():
    config = ValidationConfig()

    ambiguous, invalid = sequence_symbol_sets("ACDX*", config)

    assert ambiguous == {"X"}
    assert invalid == {"*"}


def test_validate_dataset_removes_missing_target_and_invalid_sequence():
    frame = pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3"],
            "sequence": [VALID_SEQUENCE, VALID_SEQUENCE.replace("A", "*", 1), VALID_SEQUENCE],
            "optimal_ph": [7.0, 8.0, None],
            "organism": ["Org A", "Org B", "Org A"],
            "ec_number": ["1.1.1.1", "2.2.2.2", "1.1.1.1"],
        }
    )

    cleaned, report = validate_dataset(frame, ValidationConfig())

    assert len(cleaned) == 1
    assert report.original_records == 3
    assert report.cleaned_records == 1
    assert report.removed_missing_targets == 1
    assert report.removed_invalid_sequences == 1
    assert report.organism_available is True
    assert report.ec_available is True


def test_validate_dataset_reports_duplicate_sequences_and_conflicting_labels():
    frame = pd.DataFrame(
        {
            "protein_id": ["p1", "p1", "p3"],
            "sequence": [VALID_SEQUENCE, VALID_SEQUENCE, VALID_SEQUENCE.replace("C", "A", 1)],
            "optimal_ph": [7.0, 8.0, 6.5],
        }
    )

    _, report = validate_dataset(frame, ValidationConfig())

    assert report.duplicate_identifier_count == 1
    assert report.duplicate_sequence_count == 1
    assert report.conflicting_duplicate_sequence_labels == 1


def test_validate_dataset_rejects_missing_required_columns():
    frame = pd.DataFrame({"sequence": [VALID_SEQUENCE]})

    with pytest.raises(ValueError, match="Missing required dataset columns"):
        validate_dataset(frame, ValidationConfig())


def test_validate_dataset_removes_out_of_range_ph():
    frame = pd.DataFrame(
        {
            "sequence": [VALID_SEQUENCE, VALID_SEQUENCE.replace("D", "E", 1)],
            "optimal_ph": [7.0, 15.0],
        }
    )

    cleaned, report = validate_dataset(frame, ValidationConfig())

    assert len(cleaned) == 1
    assert report.removed_out_of_range_targets == 1

