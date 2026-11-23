"""Logging setup and small shared path and formatting helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logger() -> logging.Logger:
    """
    Configure root and Falcon logging handlers for CLI execution.

    Attaches a debug-level stderr handler if the root logger has no
    handlers yet, enables warning capture, and suppresses verbose debug
    logs from ``numba`` and ``numexpr``.

    Returns
    -------
    logging.Logger
        The ``"falcon"`` application logger.
    """
    logger = logging.getLogger("falcon")
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "{asctime} {levelname} [{name}/{processName}] "
                "{module}.{funcName} : {message}",
                style="{",
            )
        )
        root.addHandler(handler)
    # Silence verbose third-party debug/info logs.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return logger


def resolve_path(path: Path) -> Path:
    """
    Resolve ``path`` strictly, falling back when the target is absent.

    Using ``strict=True`` first surfaces symlink loops and permission
    errors while still allowing non-existent destination paths.

    Parameters
    ----------
    path : Path
        Filesystem path to canonicalize.

    Returns
    -------
    Path
        Canonicalized path.
    """
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.resolve(strict=False)


def format_hms(seconds: float) -> str:
    """
    Format a non-negative duration in seconds as ``H:MM:SS``.

    Parameters
    ----------
    seconds : float
        Elapsed or estimated duration in seconds.

    Returns
    -------
    str
        Human-readable ``H:MM:SS`` string.
    """
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"
