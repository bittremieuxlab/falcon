"""Logging setup and small shared path, glob, and worker helpers."""

from __future__ import annotations

import contextlib
import glob
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import joblib

_CGROUP_MEMORY_PATHS: tuple[str, ...] = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

__all__ = [
    "atomic_write_text",
    "configure_logger",
    "effective_memory_bytes",
    "effective_n_workers",
    "format_hms",
    "loky_unordered_pool",
    "resolve_input_files",
]


def configure_logger() -> logging.Logger:
    """
    Configure root and Falcon logging handlers for CLI execution.

    Attaches a debug-level stderr handler if the root logger has no
    handlers yet, enables warning capture, and suppresses verbose debug
    logs from ``numba``.
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
    logging.getLogger("numba").setLevel(logging.WARNING)
    return logger


def resolve_input_files(patterns: Sequence[str]) -> list[Path]:
    """
    Expand input peak-file glob patterns in deterministic sorted order.

    Matches within each pattern are sorted lexicographically so staging
    order never depends on filesystem iteration order. Existing literal
    paths containing glob metacharacters (e.g. ``sample_[1].mgf``) are
    also accepted.

    Raises
    ------
    FileNotFoundError
        If ``patterns`` is empty, a pattern matches no files, or a
        matched path does not refer to an existing file.
    IsADirectoryError
        If any path or glob match refers to a directory.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        matches = [Path(m) for m in sorted(glob.glob(pattern))]
        if not matches:
            literal = Path(pattern)
            if literal.exists() or literal.is_symlink():
                matches = [literal]
            else:
                raise FileNotFoundError(
                    f"Input peak path/pattern matched no files: {pattern!r}"
                )
        for path in matches:
            if path.is_dir():
                raise IsADirectoryError(
                    f"Input peak path is a directory, not a file: {path}"
                )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Input peak path is not an existing regular file: {path}"
                )
        resolved.extend(matches)
    if not resolved:
        raise FileNotFoundError("No input peak files supplied")
    return resolved


@contextlib.contextmanager
def atomic_write_text(
    path: Path,
    *,
    prefix: str = ".falcon_",
    encoding: str = "utf-8",
) -> Generator[IO[str], None, None]:
    """Atomically write text to ``path`` via a sibling temp file."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=prefix, suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f_out:
            yield f_out
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def format_hms(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS``, clamping at zero."""
    if not math.isfinite(seconds) or seconds <= 0:
        return "0:00:00"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def effective_n_workers() -> int:
    """Return the effective worker count for ``joblib`` pools."""
    import joblib

    return max(1, int(joblib.effective_n_jobs(-1)))


def loky_unordered_pool(n_jobs: int) -> joblib.Parallel:
    """Create a ``loky`` parallel pool streaming unordered results."""
    import joblib

    return joblib.Parallel(
        n_jobs=max(1, n_jobs),
        backend="loky",
        verbose=0,
        return_as="generator_unordered",
    )


def effective_memory_bytes() -> int | None:
    """
    Return effective available physical RAM in bytes, or ``None``.

    Combines POSIX ``sysconf`` physical memory with any stricter Linux
    cgroup memory limit when readable.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    total_ram = int(page_size) * int(phys_pages)
    for cg_path in _CGROUP_MEMORY_PATHS:
        try:
            raw = Path(cg_path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) > 0:
            total_ram = min(total_ram, int(raw))
    return total_ram
