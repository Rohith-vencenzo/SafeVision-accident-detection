"""Logging helpers - one console handler, optional rotating file handler."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False

__all__ = ["configure_logging", "get_logger"]

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(
    level: str | int = "INFO",
    log_file: str | os.PathLike | None = None,
    force: bool = False,
) -> None:
    """Configure the ``safevision`` logger tree exactly once."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        numeric = _LEVELS.get(level.strip().upper(), logging.INFO)
    else:
        numeric = int(level)

    root = logging.getLogger("safevision")
    root.setLevel(numeric)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-34s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(numeric)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(numeric)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # never let logging break detection
            root.warning("Could not open log file %s (%s); continuing on console only", path, exc)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, **_: Any) -> logging.Logger:
    """Return a child logger of the ``safevision`` tree."""
    if name == "__main__":
        name = "safevision.main"
    if not name.startswith("safevision"):
        name = f"safevision.{name}"
    return logging.getLogger(name)
