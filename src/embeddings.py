"""Frozen protein-language-model embedding generation and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for frozen protein-language-model embeddings."""

    model_name: str
    batch_size: int
    max_sequence_length: int
    device: str = "auto"
    cache_dir: str | None = None


def model_slug(model_name: str) -> str:
    """Return a filesystem-safe slug for a model name."""
    return model_name.replace("/", "_").replace("-", "_")


def sequence_checksum(sequences: Iterable[str]) -> str:
    """Compute a stable checksum for an ordered sequence collection."""
    digest = hashlib.sha256()
    for sequence in sequences:
        normalized = str(sequence).strip().upper()
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve `auto` to CUDA when available, otherwise CPU."""
    if requested_device != "auto":
        return requested_device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def embedding_cache_valid(
    embedding_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_model_name: str,
    expected_n_sequences: int,
    expected_sequence_checksum: str,
) -> bool:
    """Return whether cached embeddings match the expected dataset/model."""
    embedding_file = Path(embedding_path)
    metadata_file = Path(metadata_path)
    if not embedding_file.exists() or not metadata_file.exists():
        return False

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_file, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    return (
        metadata.get("model_name") == expected_model_name
        and metadata.get("n_sequences") == expected_n_sequences
        and metadata.get("sequence_checksum") == expected_sequence_checksum
        and tuple(embeddings.shape)[0] == expected_n_sequences
        and tuple(embeddings.shape)[1] == metadata.get("embedding_dim")
    )


def write_embedding_metadata(
    metadata_path: str | Path,
    *,
    model_name: str,
    n_sequences: int,
    sequence_checksum_value: str,
    embedding_dim: int,
    device: str,
    batch_size: int,
    max_sequence_length: int,
) -> None:
    """Write metadata describing cached embedding generation."""
    metadata = {
        "model_name": model_name,
        "n_sequences": n_sequences,
        "sequence_checksum": sequence_checksum_value,
        "embedding_dim": embedding_dim,
        "device": device,
        "batch_size": batch_size,
        "max_sequence_length": max_sequence_length,
        "dtype": "float32",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def generate_esm_embeddings(
    sequences: list[str],
    config: EmbeddingConfig,
    embedding_path: str | Path,
    metadata_path: str | Path,
    force: bool = False,
) -> np.ndarray:
    """Generate or load cached frozen ESM embeddings.

    Embeddings are mean-pooled over non-special, non-padding residue tokens.
    """
    checksum = sequence_checksum(sequences)
    if not force and embedding_cache_valid(
        embedding_path,
        metadata_path,
        expected_model_name=config.model_name,
        expected_n_sequences=len(sequences),
        expected_sequence_checksum=checksum,
    ):
        LOGGER.info("Using cached embeddings from %s", embedding_path)
        return np.load(embedding_path)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Generating protein embeddings requires torch and transformers. On Windows, use a "
            "Python 3.9-3.12 environment for PyTorch. Recommended: conda env create -f "
            "environment-embeddings.yml"
        ) from exc

    device = resolve_device(config.device)
    LOGGER.info("Loading %s on %s", config.model_name, device)
    cache_dir = config.cache_dir
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(config.model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    total = len(sequences)
    max_length = config.max_sequence_length + 2
    with torch.no_grad():
        for start in range(0, total, config.batch_size):
            batch = [str(sequence).strip().upper() for sequence in sequences[start : start + config.batch_size]]
            LOGGER.info("Embedding batch %s-%s of %s", start + 1, min(start + len(batch), total), total)
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model(**tokenized)
            pooled = mean_pool_residue_tokens(outputs.last_hidden_state, tokenized["attention_mask"])
            all_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    output_path = Path(embedding_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    write_embedding_metadata(
        metadata_path,
        model_name=config.model_name,
        n_sequences=len(sequences),
        sequence_checksum_value=checksum,
        embedding_dim=int(embeddings.shape[1]),
        device=device,
        batch_size=config.batch_size,
        max_sequence_length=config.max_sequence_length,
    )
    LOGGER.info("Wrote embeddings to %s with shape %s", output_path, embeddings.shape)
    return embeddings


def mean_pool_residue_tokens(hidden_states: Any, attention_mask: Any) -> Any:
    """Mean-pool residue token embeddings, excluding first and final non-pad tokens."""
    import torch

    mask = attention_mask.clone().to(hidden_states.dtype)
    if mask.shape[1] > 0:
        mask[:, 0] = 0
    lengths = attention_mask.sum(dim=1).to(torch.long)
    row_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    final_token_indices = torch.clamp(lengths - 1, min=0)
    mask[row_indices, final_token_indices] = 0
    denominator = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denominator
