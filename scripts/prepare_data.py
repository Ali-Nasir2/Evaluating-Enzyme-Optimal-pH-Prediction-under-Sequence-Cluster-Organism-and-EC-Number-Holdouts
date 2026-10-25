"""Load, validate, and clean the raw enzyme optimal-pH dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_dataset, resolve_dataset_path
from src.data_validation import (
    validate_dataset,
    validation_config_from_project_config,
    write_quality_report,
)
from src.utils import configure_logging, ensure_directory, file_sha256, load_config, runtime_environment

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)

    raw_data_dir = config["paths"]["raw_data_dir"]
    dataset_file = config.get("dataset", {}).get("file")
    dataset_path = resolve_dataset_path(raw_data_dir, dataset_file)

    LOGGER.info("Loading dataset from %s", dataset_path)
    loaded = load_dataset(dataset_path)
    LOGGER.info(
        "Loaded %s records with %s columns from %s",
        len(loaded.frame),
        len(loaded.frame.columns),
        loaded.source_format,
    )

    validation_config = validation_config_from_project_config(config)
    cleaned, report = validate_dataset(loaded.frame, validation_config)

    output_path = Path(config["paths"]["cleaned_dataset"])
    ensure_directory(output_path.parent)
    cleaned.to_csv(output_path, index=False)
    write_quality_report(report, config["paths"]["data_quality_report"])

    LOGGER.info("Dataset checksum: %s", file_sha256(dataset_path))
    LOGGER.info("Runtime environment: %s", runtime_environment())
    LOGGER.info("Wrote cleaned dataset to %s", output_path)
    LOGGER.info("Wrote data-quality report to %s", config["paths"]["data_quality_report"])

    if cleaned.empty:
        raise ValueError(
            "No records remain after validation. Inspect reports/data_quality_report.md "
            "and adjust cleaning thresholds only if scientifically justified."
        )


if __name__ == "__main__":
    main()
