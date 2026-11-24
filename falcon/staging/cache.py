"""Staging cache provenance manifests, validation, and lifecycle."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..dataset import (
    bucket_dataset_name,
    is_bucket_dataset_name,
    spectra_dir,
)
from ..types import OTHER_BUCKET, charge_sort_key, parse_charge_bucket_list
from ..utils import resolve_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import spectrum_utils.spectrum as sus

    from ..config import FalconConfig
    from ..types import ChargeBucket, ChargeValue, ProcessedSpectrum

logger = logging.getLogger(__name__)

STAGING_CACHE_SCHEMA_VERSION = 1
STAGING_CACHE_FILENAME = "staging_cache.json"

_CACHE_HINT = "rerun with --overwrite or a different --work-dir."

__all__ = [
    "STAGING_CACHE_FILENAME",
    "STAGING_CACHE_SCHEMA_VERSION",
    "build_staging_manifest",
    "deserialize_cached_buckets",
    "diff_staging_manifests",
    "load_or_stage_buckets",
    "serialize_cached_buckets",
]


def _json_equal(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without ``bool == int/float``."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _json_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(x, y) for x, y in zip(left, right, strict=True)
        )
    return bool(left == right)


def serialize_cached_buckets(
    buckets: Sequence[ChargeBucket] | None,
) -> list[list[ChargeValue] | str] | None:
    """
    Serialize charge bucket keys for ``staging_cache.json``.

    Parameters
    ----------
    buckets : Sequence[ChargeBucket] | None
        Charge bucket keys, or ``None`` when unset.

    Returns
    -------
    list[list[ChargeValue] | str] | None
        Deterministic JSON-serializable representation, or ``None``.
    """
    if buckets is None:
        return None
    serialized: list[list[ChargeValue] | str] = []
    for bucket in buckets:
        if bucket == OTHER_BUCKET:
            serialized.append(OTHER_BUCKET)
        else:
            serialized.append(sorted(bucket, key=charge_sort_key))
    return serialized


def deserialize_cached_buckets(raw_buckets: Any) -> list[ChargeBucket]:
    """
    Validate and deserialize bucket keys from ``staging_cache.json``.

    Parameters
    ----------
    raw_buckets : Any
        JSON-decoded ``buckets`` field from a staging cache manifest.

    Returns
    -------
    list[ChargeBucket]
        Validated charge bucket keys.

    Raises
    ------
    ValueError
        If ``raw_buckets`` is malformed or contains invalid bucket keys.
    """
    if not isinstance(raw_buckets, list):
        raise ValueError("Cached staging manifest has invalid 'buckets' field")
    if len(raw_buckets) == 0:
        return []
    return list(parse_charge_bucket_list(raw_buckets))


def build_staging_manifest(
    config: FalconConfig,
    resolved_inputs: Sequence[Path],
    buckets: Sequence[ChargeBucket] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable staging cache provenance manifest.

    Captures every parameter that affects staged Lance datasets:
    resolved input files (canonical path, byte size, mtime_ns),
    ``preprocessing``, ``clustering.fragment_tol``, and
    ``clustering.precursor_charge_buckets``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    resolved_inputs : Sequence[Path]
        Expanded input peak file paths in deterministic staging order.
    buckets : Sequence[ChargeBucket] | None, default=None
        Staged charge bucket keys written to disk, if already computed.

    Returns
    -------
    dict[str, Any]
        JSON-serializable manifest dictionary.
    """
    input_entries: list[dict[str, Any]] = []
    for input_path in resolved_inputs:
        resolved = resolve_path(input_path)
        st = os.stat(resolved)
        input_entries.append(
            {
                "path": str(resolved),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )

    return {
        "schema_version": STAGING_CACHE_SCHEMA_VERSION,
        "falcon_version": str(__version__),
        "inputs": input_entries,
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "fragment_tol": config.clustering.fragment_tol,
        "precursor_charge_buckets": serialize_cached_buckets(
            config.clustering.precursor_charge_buckets
        ),
        "buckets": serialize_cached_buckets(buckets),
    }


def diff_staging_manifests(
    cached: Any,
    expected: Mapping[str, Any],
) -> list[str]:
    """
    Return staging manifest section names that differ from ``expected``.

    Parameters
    ----------
    cached : Any
        Parsed JSON payload loaded from ``staging_cache.json``.
    expected : Mapping[str, Any]
        Expected staging manifest for the current run.

    Returns
    -------
    list[str]
        Names of mismatched sections (empty list when the cache is
        valid and reusable).
    """
    if not isinstance(cached, Mapping):
        return ["schema_version"]
    if not _json_equal(
        cached.get("schema_version"), expected.get("schema_version")
    ):
        return ["schema_version"]

    return [
        key
        for key in (
            "inputs",
            "preprocessing",
            "fragment_tol",
            "precursor_charge_buckets",
        )
        if key not in cached or not _json_equal(cached[key], expected[key])
    ]


def _cache_error(message: str) -> ValueError:
    """Format a staging cache error with the standard recovery hint."""
    return ValueError(f"{message}; {_CACHE_HINT}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata to ``path`` via rename."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".staging_cache_", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            f_out = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with f_out:
            json.dump(payload, f_out, indent=2, allow_nan=False)
            f_out.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _has_unverified_staging_files(target_dir: Path) -> bool:
    """Return True if ``target_dir`` has unmanifested cache files."""
    return any(
        entry.name == "charges.joblib" or is_bucket_dataset_name(entry.name)
        for entry in target_dir.iterdir()
    )


def _staged_dataset_names(target_dir: Path) -> set[str] | None:
    """Return staged ``spectra_charge_*.lance`` dir names, or None."""
    names: set[str] = set()
    for entry in target_dir.iterdir():
        if is_bucket_dataset_name(entry.name):
            if not entry.is_dir():
                return None
            names.add(entry.name)
    return names


def _clear_directory(target_dir: Path) -> None:
    """Remove all files and subdirectories inside ``target_dir``."""
    for entry in target_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _load_validated_cache(
    work_dir: Path,
    target_dir: Path,
    cache_path: Path,
    expected_manifest: Mapping[str, Any],
) -> list[ChargeBucket]:
    """Read and validate an existing ``staging_cache.json`` manifest."""
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _cache_error(
            f"Corrupted staging cache manifest at {cache_path}"
        ) from exc

    mismatches = diff_staging_manifests(cached, expected_manifest)
    if mismatches:
        diff_str = ", ".join(mismatches)
        raise _cache_error(
            f"Cached staging data in {work_dir} does not match the current "
            f"configuration (mismatched: {diff_str})"
        )

    try:
        cached_buckets = deserialize_cached_buckets(cached.get("buckets"))
    except ValueError as exc:
        raise _cache_error(
            f"Invalid staging cache buckets in {cache_path}: {exc}"
        ) from exc

    expected_datasets = {
        bucket_dataset_name(bucket) for bucket in cached_buckets
    }
    if _staged_dataset_names(target_dir) != expected_datasets:
        raise _cache_error(
            f"Invalid staging cache buckets in {cache_path}: staged datasets "
            "on disk do not match cached buckets"
        )
    logger.info(
        "Reusing validated staging cache in %s (%d bucket(s))",
        work_dir,
        len(cached_buckets),
    )
    return cached_buckets


def load_or_stage_buckets(
    config: FalconConfig,
    work_dir: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> list[ChargeBucket]:
    """
    Return staged charge buckets, reusing validated cache when safe.

    Under ``config.io.overwrite``, the staging directory is cleared and
    spectra are re-read and re-partitioned. Otherwise, an existing
    ``staging_cache.json`` manifest is reused only when all staging
    inputs, preprocessing parameters, ``fragment_tol``, charge buckets,
    and on-disk Lance directories match.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory containing the ``spectra`` folder.
    process_spectrum : Callable
        Configured single-spectrum preprocessing callable.

    Returns
    -------
    list[ChargeBucket]
        Charge bucket keys with staged Lance datasets.
    """
    from .ingest import resolve_input_files, stage_spectra

    resolved_inputs = resolve_input_files(config.io.input_filenames)
    expected_manifest = build_staging_manifest(config, resolved_inputs)
    target_dir = spectra_dir(work_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / STAGING_CACHE_FILENAME

    if config.io.overwrite:
        _clear_directory(target_dir)
    elif cache_path.exists() or cache_path.is_symlink():
        return _load_validated_cache(
            work_dir, target_dir, cache_path, expected_manifest
        )
    elif _has_unverified_staging_files(target_dir):
        raise _cache_error(
            f"Working directory {work_dir} contains an unverified or legacy "
            "staging cache without staging_cache.json"
        )

    charge_buckets = stage_spectra(
        resolved_inputs,
        work_dir,
        process_spectrum,
        config.clustering.precursor_charge_buckets,
    )
    manifest = {
        **expected_manifest,
        "buckets": serialize_cached_buckets(charge_buckets),
    }
    _write_json_atomic(cache_path, manifest)
    return charge_buckets
