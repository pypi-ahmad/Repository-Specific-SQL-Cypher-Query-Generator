"""Utility helpers for reproducibility, storage, and logging."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def configure_logging() -> None:
    """Configure project logger once."""

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for python, numpy, and torch if available.

    Args:
        seed: Global random seed.

    Example:
        >>> set_global_seed(42)
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        logger.debug("Torch not available while seeding; continuing.")


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    """Save JSON payload with stable formatting."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load JSON payload."""

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def utc_now_iso() -> str:
    """Return current UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    """Normalize whitespace for robust string comparison."""

    return " ".join(text.strip().split())
