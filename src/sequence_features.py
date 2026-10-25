"""Handcrafted sequence-feature extraction for protein sequences."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
ACIDIC_RESIDUES = frozenset({"D", "E"})
BASIC_RESIDUES = frozenset({"H", "K", "R"})
HYDROPHOBIC_RESIDUES = frozenset({"A", "V", "I", "L", "M", "F", "W", "Y"})
POLAR_RESIDUES = frozenset({"S", "T", "N", "Q", "C", "Y"})
AROMATIC_RESIDUES = frozenset({"F", "W", "Y"})
ALIPHATIC_RESIDUES = frozenset({"A", "V", "I", "L"})
AMBIGUOUS_RESIDUES = frozenset({"X", "B", "Z", "J", "U", "O"})

KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def compute_sequence_features(sequence: str, include_dipeptides: bool = False) -> dict[str, float]:
    """Compute handcrafted features for one amino-acid sequence.

    Ambiguous residues are counted and retained in the denominator for global
    length-aware fractions, but amino-acid composition features are defined only
    for the 20 standard residues.
    """
    normalized = str(sequence).strip().upper()
    length = len(normalized)
    if length == 0:
        raise ValueError("Cannot compute features for an empty sequence.")

    counts = Counter(normalized)
    standard_count = sum(counts[aa] for aa in AMINO_ACIDS)
    ambiguous_count = sum(counts[aa] for aa in AMBIGUOUS_RESIDUES)

    features: dict[str, float] = {
        "length": float(length),
        "standard_residue_fraction": _safe_divide(standard_count, length),
        "ambiguous_residue_fraction": _safe_divide(ambiguous_count, length),
        "acidic_fraction": _fraction(counts, ACIDIC_RESIDUES, length),
        "basic_fraction": _fraction(counts, BASIC_RESIDUES, length),
        "net_charge_proxy": _fraction(counts, BASIC_RESIDUES, length)
        - _fraction(counts, ACIDIC_RESIDUES, length),
        "hydrophobic_fraction": _fraction(counts, HYDROPHOBIC_RESIDUES, length),
        "polar_fraction": _fraction(counts, POLAR_RESIDUES, length),
        "aromatic_fraction": _fraction(counts, AROMATIC_RESIDUES, length),
        "aliphatic_fraction": _fraction(counts, ALIPHATIC_RESIDUES, length),
        "gravy": _gravy(counts, standard_count),
    }

    for aa in AMINO_ACIDS:
        features[f"aa_{aa}_fraction"] = _safe_divide(counts[aa], length)

    features.update(_biopython_features(normalized))

    if include_dipeptides:
        features.update(_dipeptide_features(normalized))

    return features


def featurize_sequences(
    sequences: Iterable[str], include_dipeptides: bool = False
) -> pd.DataFrame:
    """Compute sequence features for an iterable of protein sequences."""
    return pd.DataFrame.from_records(
        compute_sequence_features(sequence, include_dipeptides=include_dipeptides)
        for sequence in sequences
    )


def build_feature_table(
    frame: pd.DataFrame,
    sequence_column: str,
    metadata_columns: Iterable[str] = (),
    include_dipeptides: bool = False,
) -> pd.DataFrame:
    """Return metadata columns plus handcrafted features for a dataframe."""
    if sequence_column not in frame.columns:
        raise ValueError(f"Missing sequence column: {sequence_column}")

    metadata = [column for column in metadata_columns if column in frame.columns]
    features = featurize_sequences(
        frame[sequence_column].astype(str), include_dipeptides=include_dipeptides
    )
    return pd.concat([frame.loc[:, metadata].reset_index(drop=True), features], axis=1)


def feature_column_names(columns: Iterable[str]) -> list[str]:
    """Return columns that should be used as numeric model features."""
    excluded_prefixes = ("protein_id", "accession", "organism", "taxonomy", "ec", "split")
    selected: list[str] = []
    for column in columns:
        normalized = str(column).lower().replace(" ", "_")
        if normalized.startswith("unnamed") or normalized == "row_index":
            continue
        if any(normalized.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if normalized in {"sequence", "optimal_ph", "phopt", "target", "label", "sample_weight"}:
            continue
        selected.append(str(column))
    return selected


def _fraction(counts: Counter[str], residues: frozenset[str], length: int) -> float:
    return _safe_divide(sum(counts[residue] for residue in residues), length)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _gravy(counts: Counter[str], standard_count: int) -> float:
    if standard_count == 0:
        return float("nan")
    total = sum(counts[aa] * KYTE_DOOLITTLE[aa] for aa in AMINO_ACIDS)
    return float(total) / float(standard_count)


def _biopython_features(sequence: str) -> dict[str, float]:
    """Compute optional Biopython physicochemical descriptors."""
    if any(residue not in AMINO_ACIDS for residue in sequence):
        return {
            "molecular_weight": float("nan"),
            "isoelectric_point": float("nan"),
            "instability_index": float("nan"),
            "aromaticity": float("nan"),
        }

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        return {
            "molecular_weight": float("nan"),
            "isoelectric_point": float("nan"),
            "instability_index": float("nan"),
            "aromaticity": float("nan"),
        }

    analysis = ProteinAnalysis(sequence)
    return {
        "molecular_weight": float(analysis.molecular_weight()),
        "isoelectric_point": float(analysis.isoelectric_point()),
        "instability_index": float(analysis.instability_index()),
        "aromaticity": float(analysis.aromaticity()),
    }


def _dipeptide_features(sequence: str) -> dict[str, float]:
    standard_pairs = ["".join(pair) for pair in product(AMINO_ACIDS, repeat=2)]
    denominator = max(len(sequence) - 1, 1)
    pair_counts = Counter(
        sequence[index : index + 2]
        for index in range(len(sequence) - 1)
        if sequence[index] in AMINO_ACIDS and sequence[index + 1] in AMINO_ACIDS
    )
    return {
        f"dipeptide_{pair}_fraction": float(np.round(pair_counts[pair] / denominator, 8))
        for pair in standard_pairs
    }
