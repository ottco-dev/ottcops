"""Logging helpers for the OpenCore service."""
from __future__ import annotations

import logging
from .config import LOG_LEVEL


def build_logger() -> logging.Logger:
    """Initialize and return the shared logger."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)
    else:
        logging.getLogger().setLevel(level)
    return logging.getLogger("ottcouture.app")
