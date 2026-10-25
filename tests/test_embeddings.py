from __future__ import annotations

import json

import numpy as np

from src.embeddings import (
    embedding_cache_valid,
    model_slug,
    sequence_checksum,
    write_embedding_metadata,
)


def test_model_slug_is_filesystem_safe():
    assert model_slug("facebook/esm2_t6_8M_UR50D") == "facebook_esm2_t6_8M_UR50D"


def test_sequence_checksum_depends_on_order():
    first = sequence_checksum(["AAA", "CCC"])
    second = sequence_checksum(["CCC", "AAA"])

    assert first != second
    assert first == sequence_checksum(["aaa", "ccc"])


def test_embedding_cache_validates_metadata_and_shape(tmp_path):
    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    sequences = ["AAA", "CCC"]
    checksum = sequence_checksum(sequences)
    np.save(embedding_path, np.zeros((2, 3), dtype=np.float32))
    write_embedding_metadata(
        metadata_path,
        model_name="test-model",
        n_sequences=2,
        sequence_checksum_value=checksum,
        embedding_dim=3,
        device="cpu",
        batch_size=2,
        max_sequence_length=10,
    )

    assert embedding_cache_valid(
        embedding_path,
        metadata_path,
        expected_model_name="test-model",
        expected_n_sequences=2,
        expected_sequence_checksum=checksum,
    )


def test_embedding_cache_rejects_wrong_model(tmp_path):
    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(embedding_path, np.zeros((2, 3), dtype=np.float32))
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "other-model",
                "n_sequences": 2,
                "sequence_checksum": "abc",
                "embedding_dim": 3,
            }
        ),
        encoding="utf-8",
    )

    assert not embedding_cache_valid(
        embedding_path,
        metadata_path,
        expected_model_name="test-model",
        expected_n_sequences=2,
        expected_sequence_checksum="abc",
    )

