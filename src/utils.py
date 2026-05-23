"""
utils.py
--------
Utility helpers shared across the SML pipeline.

Author : Shreya Mishra (IED/10032/23)
Course : ED317 - Statistical Machine Learning I
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducibility across numpy, random, torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the project root (sml_project/)."""
    return Path(__file__).resolve().parent.parent


def get_paths() -> dict:
    """Return a dictionary of standard project paths."""
    root = get_project_root()
    paths = {
        "root": root,
        "data": root / "data",
        "outputs": root / "outputs",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_logger(name: str = "sml") -> logging.Logger:
    """Return a configured logger that prints to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


if __name__ == "__main__":
    set_global_seed()
    log = get_logger()
    log.info("utils.py loaded. Project paths: %s", get_paths())
