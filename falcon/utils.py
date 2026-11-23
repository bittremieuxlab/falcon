"""Logging setup and small shared path, glob, and worker helpers."""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "configure_logger",
    "effective_n_workers",
    "expand_input_globs",
    "format_hms",
    "resolve_path",
]


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


def expand_input_globs(
    patterns: Sequence[str],
    *,
    keep_unmatched_literals: bool = False,
) -> list[Path]:
    """
    Expand file paths and glob patterns in deterministic sorted order.

    Matches within each pattern are sorted lexicographically so results
    never depend on filesystem iteration order. Repeated matches across
    patterns are preserved.

    Parameters
    ----------
    patterns : Sequence[str]
        Literal paths or glob patterns.
    keep_unmatched_literals : bool, default=False
        When ``True``, preserve unmatched patterns containing no glob
        metacharacters as literal ``Path`` entries (useful for checking
        path aliasing before files exist).

    Returns
    -------
    list[Path]
        Expanded file paths.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            resolved.extend(Path(match) for match in matches)
        elif keep_unmatched_literals and not glob.has_magic(pattern):
            resolved.append(Path(pattern))
    return resolved


def format_hms(seconds: float) -> str:
    """
    Format a duration in seconds as ``H:MM:SS``, clamping below zero.

    Parameters
    ----------
    seconds : float
        Elapsed or estimated duration in seconds. Negative inputs are
        clamped to ``0:00:00``.

    Returns
    -------
    str
        Human-readable ``H:MM:SS`` string.
    """
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def effective_n_workers() -> int:
    """
    Return the effective worker count for ``joblib`` parallel pools.

    Returns
    -------
    int
        Positive worker count derived from ``joblib.effective_n_jobs``.
    """
    import joblib

    return max(1, int(joblib.effective_n_jobs(-1)))
