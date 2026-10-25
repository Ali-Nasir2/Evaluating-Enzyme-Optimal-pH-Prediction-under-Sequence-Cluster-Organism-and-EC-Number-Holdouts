"""Generate handcrafted sequence features from the cleaned dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.sequence_features import build_feature_table
from src.utils import configure_logging, ensure_directory, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--include-dipeptides",
        action="store_true",
        help="Also compute 400 dipeptide-composition features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)

    dataset_config = config["dataset"]
    paths = config["paths"]
    cleaned_path = Path(paths["cleaned_dataset"])
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run scripts/prepare_data.py first."
        )

    frame = pd.read_csv(cleaned_path)
    metadata_columns = [
        dataset_config["id_column"],
        dataset_config["target_column"],
        dataset_config.get("organism_column"),
        dataset_config.get("taxonomy_column"),
        dataset_config.get("ec_column"),
        dataset_config.get("split_column"),
    ]
    metadata_columns = [column for column in metadata_columns if column]

    LOGGER.info("Generating handcrafted features for %s sequences", len(frame))
    features = build_feature_table(
        frame=frame,
        sequence_column=dataset_config["sequence_column"],
        metadata_columns=metadata_columns,
        include_dipeptides=args.include_dipeptides,
    )

    output_path = Path(paths["handcrafted_features"])
    ensure_directory(output_path.parent)
    features.to_csv(output_path, index=False)
    LOGGER.info("Wrote handcrafted features to %s with shape %s", output_path, features.shape)


if __name__ == "__main__":
    main()
