"""Logging configuration for the AgriBot runtime.

Two sinks: a terse console stream for the operator standing in the arena, and
a verbose rotating file for post-run analysis. Both are configured once from
``telemetry`` in the master config.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

__all__ = ["setup_logging", "get_logger"]

_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(funcName)s:%(lineno)d %(message)s"
_DATE_FMT = "%H:%M:%S"

_configured = False


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    filename: str = "agribot.log",
    max_bytes: int = 8 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure the root ``agribot`` logger. Idempotent unless ``force``."""
    global _configured
    root = logging.getLogger("agribot")
    if _configured and not force:
        return root

    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(_level(console_level))
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_level(file_level))
        file_handler.setFormatter(logging.Formatter(_FILE_FMT))
        root.addHandler(file_handler)

    _configured = True
    return root


def _level(name: str) -> int:
    resolved = logging.getLevelName(str(name).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Get a child of the ``agribot`` logger, e.g. ``get_logger("vision.line")``."""
    return logging.getLogger(f"agribot.{name}" if not name.startswith("agribot") else name)
