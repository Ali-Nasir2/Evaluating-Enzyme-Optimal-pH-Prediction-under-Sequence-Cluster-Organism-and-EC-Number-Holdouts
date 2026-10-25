from __future__ import annotations

import pandas as pd
import pytest

from src.explainability import largest_errors, mark_group_frequency


def test_mark_group_frequency_labels_rare_and_common():
    frame = pd.DataFrame({"Organism": ["A", "A", "B"], "absolute_error": [1.0, 2.0, 3.0]})

    marked = mark_group_frequency(frame, "Organism", rare_threshold=1)

    assert marked.loc[0, "Organism_frequency"] == "common"
    assert marked.loc[2, "Organism_frequency"] == "rare"


def test_mark_group_frequency_rejects_missing_column():
    with pytest.raises(ValueError, match="Missing group column"):
        mark_group_frequency(pd.DataFrame({"x": [1]}), "Organism")


def test_largest_errors_filters_and_sorts():
    frame = pd.DataFrame(
        {
            "feature_group": ["a", "a", "b"],
            "split_strategy": ["random", "random", "random"],
            "model": ["ridge", "ridge", "ridge"],
            "absolute_error": [0.2, 1.5, 9.0],
        }
    )

    result = largest_errors(frame, n=1, feature_group="a")

    assert result.loc[0, "absolute_error"] == 1.5

