"""Generate cached frozen protein-language-model embeddings."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.embeddings import EmbeddingConfig, generate_esm_embeddings
from src.utils import configure_logging, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--force", action="store_true", help="Regenerate embeddings even if cache is valid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)

    paths = config["paths"]
    dataset_config = config["dataset"]
    embedding_config = config["embeddings"]
    cleaned_path = Path(paths["cleaned_dataset"])
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run scripts/prepare_data.py first."
        )

    frame = pd.read_csv(cleaned_path)
    sequence_column = dataset_config["sequence_column"]
    if sequence_column not in frame.columns:
        raise ValueError(f"Missing sequence column in cleaned dataset: {sequence_column}")

    sequences = frame[sequence_column].astype(str).tolist()
    LOGGER.info("Generating embeddings for %s sequences", len(sequences))
    try:
        generate_esm_embeddings(
            sequences=sequences,
            config=EmbeddingConfig(
                model_name=embedding_config["model_name"],
                batch_size=int(embedding_config["batch_size"]),
                max_sequence_length=int(embedding_config["max_sequence_length"]),
                device=embedding_config.get("device", "auto"),
                cache_dir=paths.get("hf_cache_dir"),
            ),
            embedding_path=paths["embedding_file"],
            metadata_path=paths["embedding_metadata"],
            force=args.force,
        )
    except ImportError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
