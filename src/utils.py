"""Shared utility helpers for configuration, logging, and reproducibility."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent command-line logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return config


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_reproducible_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


def file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment() -> dict[str, str]:
    """Return basic runtime information for reproducibility reports."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = "false"
        info["cuda_device"] = "none"
    info["cwd"] = os.getcwd()
    return info

