from __future__ import annotations

import pandas as pd
import pytest

from src.splitting import (
    SplitFractions,
    build_split_frame,
    find_group_leakage,
    make_balanced_grouped_split,
    make_grouped_split,
    make_random_split,
    split_counts,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Accession": [f"p{i}" for i in range(10)],
            "Sequence": ["AAA", "AAA", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "KKK"],
        }
    )


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitFractions(train=0.8, validation=0.1, test=0.2).validate()


def test_random_split_is_reproducible():
    frame = _frame()
    fractions = SplitFractions(train=0.6, validation=0.2, test=0.2)

    first = make_random_split(frame, fractions=fractions, random_seed=123)
    second = make_random_split(frame, fractions=fractions, random_seed=123)

    assert first.tolist() == second.tolist()
    assert split_counts(pd.DataFrame({"split": first})) == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_grouped_split_keeps_exact_duplicate_sequences_together():
    frame = _frame()
    fractions = SplitFractions(train=0.6, validation=0.2, test=0.2)

    assignments = make_grouped_split(
        frame,
        group_column="Sequence",
        fractions=fractions,
        random_seed=123,
    )
    split_frame = build_split_frame(
        frame,
        split_assignments=assignments,
        strategy="exact_duplicate_safe",
        id_column="Accession",
        sequence_column="Sequence",
    )

    assert find_group_leakage(split_frame, "sequence_sha256").empty
    assert frame.loc[0, "Sequence"] == frame.loc[1, "Sequence"]
    assert assignments.iloc[0] == assignments.iloc[1]


def test_balanced_grouped_split_keeps_groups_together_and_uses_all_partitions():
    frame = pd.DataFrame(
        {
            "Accession": [f"p{i}" for i in range(20)],
            "Organism": ["large"] * 8
            + ["medium"] * 5
            + ["small_a"] * 3
            + ["small_b"] * 2
            + ["small_c"]
            + ["small_d"],
            "Sequence": [f"SEQ{i}" for i in range(20)],
        }
    )
    fractions = SplitFractions(train=0.6, validation=0.2, test=0.2)

    assignments = make_balanced_grouped_split(
        frame,
        group_column="Organism",
        fractions=fractions,
        random_seed=42,
    )
    split_frame = pd.DataFrame({"Organism": frame["Organism"], "split": assignments})

    assert set(assignments) == {"train", "validation", "test"}
    assert find_group_leakage(split_frame, "Organism").empty


def test_build_split_frame_rejects_missing_identifier_column():
    frame = pd.DataFrame({"Sequence": ["AAA", "CCC", "DDD"]})
    assignments = pd.Series(["train", "validation", "test"])

    with pytest.raises(ValueError, match="Missing identifier column"):
        build_split_frame(
            frame,
            split_assignments=assignments,
            strategy="random",
            id_column="Accession",
            sequence_column="Sequence",
        )
