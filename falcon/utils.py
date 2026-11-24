"""Logging setup and small shared path, glob, and worker helpers."""

from __future__ import annotations

import glob
import logging
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import joblib

_PROC_CGROUP_PATH = "/proc/self/cgroup"
_CGROUP_V2_ROOT = "/sys/fs/cgroup"
_CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"
_CGROUP_MEMORY_PATHS: tuple[str, ...] = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

__all__ = [
    "configure_logger",
    "effective_memory_bytes",
    "effective_n_workers",
    "expand_input_globs",
    "find_path_alias",
    "format_hms",
    "loky_unordered_pool",
    "paths_alias",
    "resolve_path",
]


def configure_logger() -> logging.Logger:
    """
    Configure root and Falcon logging handlers for CLI execution.

    Attaches a debug-level stderr handler if the root logger has no
    handlers yet, enables warning capture, and suppresses verbose debug
    logs from ``numba``.

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


def paths_alias(
    target: Path,
    candidate: Path,
    *,
    resolved_target: Path | None = None,
    target_exists: bool | None = None,
) -> bool:
    """
    Return whether ``target`` and ``candidate`` refer to the same file.

    Checks both canonical resolved paths (covering non-existent paths
    and symlinks) and ``os.path.samefile`` (covering hard links).

    Parameters
    ----------
    target : Path
        Destination or output path to test.
    candidate : Path
        Existing or literal input path to compare against.
    resolved_target : Path | None, default=None
        Optional pre-resolved canonical form of ``target``.
    target_exists : bool | None, default=None
        Optional precomputed ``target.exists()`` flag.

    Returns
    -------
    bool
        ``True`` if both paths alias the same filesystem location.
    """
    if resolved_target is None:
        try:
            resolved_target = resolve_path(target)
        except (OSError, RuntimeError, ValueError):
            return False
    try:
        if resolved_target == resolve_path(candidate):
            return True
    except (OSError, RuntimeError, ValueError):
        return False
    if target_exists is False:
        return False
    try:
        return os.path.samefile(target, candidate)
    except (OSError, ValueError):
        return False


def find_path_alias(
    target: Path,
    candidates: Iterable[Path],
    *,
    resolved_target: Path | None = None,
    strict: bool = False,
) -> Path | None:
    """
    Return the first candidate in ``candidates`` aliasing ``target``.

    Pre-resolves ``target`` and checks its existence once before
    testing each candidate via :func:`paths_alias`.

    Parameters
    ----------
    target : Path
        Destination or output path to check.
    candidates : Iterable[Path]
        Existing or literal paths to test against ``target``.
    resolved_target : Path | None, default=None
        Optional pre-resolved canonical form of ``target``.
    strict : bool, default=False
        When ``True``, propagate resolution errors on ``target``
        instead of returning ``None``.

    Returns
    -------
    Path | None
        First aliasing candidate path, or ``None`` if no candidate
        aliases ``target``.
    """
    if resolved_target is None:
        try:
            resolved_target = resolve_path(target)
        except (OSError, RuntimeError, ValueError):
            if strict:
                raise
            return None
    try:
        target_exists = target.exists()
    except (OSError, ValueError):
        target_exists = False
    for candidate in candidates:
        if paths_alias(
            target,
            candidate,
            resolved_target=resolved_target,
            target_exists=target_exists,
        ):
            return candidate
    return None


def expand_input_globs(
    patterns: Sequence[str],
    *,
    keep_unmatched_literals: bool = False,
) -> list[Path]:
    """
    Expand file paths and glob patterns in deterministic sorted order.

    Matches within each pattern are sorted lexicographically so results
    never depend on filesystem iteration order. Repeated matches across
    patterns are preserved. Existing literal paths containing glob
    metacharacters (such as ``[1]``) are retained even when ``glob``
    finds no wildcard expansion.

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
            continue
        literal = Path(pattern)
        if (
            literal.exists()
            or literal.is_symlink()
            or (keep_unmatched_literals and not glob.has_magic(pattern))
        ):
            resolved.append(literal)
    return resolved


def format_hms(seconds: float) -> str:
    """
    Format a duration in seconds as ``H:MM:SS``, clamping below zero.

    Parameters
    ----------
    seconds : float
        Elapsed or estimated duration in seconds. Non-finite or
        negative inputs are clamped to ``0:00:00``.

    Returns
    -------
    str
        Human-readable ``H:MM:SS`` string.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        return "0:00:00"
    total = int(seconds)
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


def loky_unordered_pool(n_jobs: int) -> joblib.Parallel:
    """
    Create a ``loky`` parallel pool streaming unordered results.

    Parameters
    ----------
    n_jobs : int
        Requested worker count (clamped to at least 1).

    Returns
    -------
    joblib.Parallel
        Configured ``loky`` pool with
        ``return_as="generator_unordered"``.
    """
    import joblib

    return joblib.Parallel(
        n_jobs=max(1, n_jobs),
        backend="loky",
        verbose=0,
        return_as="generator_unordered",
    )


def _read_positive_int(path: Path) -> int | None:
    """Read a positive decimal integer from ``path``, or ``None``."""
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if raw.isdigit():
        value = int(raw)
        if value > 0:
            return value
    return None


def _collect_limit(
    candidate: Path, seen: set[Path], limits: list[int]
) -> None:
    """Read a positive cgroup limit from ``candidate`` once."""
    if candidate not in seen:
        seen.add(candidate)
        limit = _read_positive_int(candidate)
        if limit is not None:
            limits.append(limit)


def _walk_cgroup_limits(
    root: Path,
    rel_path: str,
    filename: str,
    seen: set[Path],
    limits: list[int],
) -> None:
    """Collect finite limits from ``root / rel_path`` up to ``root``."""
    parts = [p for p in rel_path.split("/") if p and p not in (".", "..")]
    current = root.joinpath(*parts)
    while True:
        _collect_limit(current / filename, seen, limits)
        if current == root:
            break
        current = current.parent


def _cgroup_memory_limit() -> int | None:
    """Return minimum finite Linux cgroup memory limit in bytes."""
    limits: list[int] = []
    seen: set[Path] = set()
    try:
        proc_text = Path(_PROC_CGROUP_PATH).read_text(
            encoding="utf-8", errors="surrogateescape"
        )
    except (OSError, UnicodeError):
        proc_text = ""
    for line in proc_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, rel_path = fields
        if hierarchy == "0" and controllers == "":
            _walk_cgroup_limits(
                Path(_CGROUP_V2_ROOT), rel_path, "memory.max", seen, limits
            )
        elif "memory" in controllers.split(","):
            _walk_cgroup_limits(
                Path(_CGROUP_V1_MEMORY_ROOT),
                rel_path,
                "memory.limit_in_bytes",
                seen,
                limits,
            )
    for fallback in _CGROUP_MEMORY_PATHS:
        _collect_limit(Path(fallback), seen, limits)
    return min(limits) if limits else None


def effective_memory_bytes() -> int | None:
    """
    Return effective available physical RAM in bytes, or ``None``.

    Combines POSIX ``sysconf`` physical memory with any stricter Linux
    cgroup v1/v2 memory limits along the current process hierarchy.

    Returns
    -------
    int | None
        Effective memory limit in bytes, or ``None`` if unavailable.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    total_ram = int(page_size) * int(phys_pages)
    cgroup_limit = _cgroup_memory_limit()
    if cgroup_limit is not None:
        total_ram = min(total_ram, cgroup_limit)
    return total_ram
