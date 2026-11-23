"""Input discovery, parallel Lance staging, and cache provenance.

This module is the single owner of Falcon's staged Lance layout:
schema, canonical sort order, bucket dataset naming/labels, concurrent
Lance writes, and ``staging_cache.json`` manifest validation.
"""

from __future__ import annotations

import collections
import glob
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import lance
import pyarrow as pa
import spectrum_utils.spectrum as sus

from . import __version__, ms_io
from .config.models import FalconConfig
from .types import (
    ChargeBucket,
    ChargeValue,
    ProcessedSpectrum,
    charge_sort_key,
    parse_charge_buckets,
)

logger = logging.getLogger("falcon")

STAGING_CACHE_SCHEMA_VERSION = 1
STAGING_CACHE_FILENAME = "staging_cache.json"
SPECTRA_SUBDIR = "spectra"

_BUCKET_PREFIX = "spectra_charge_"
_BUCKET_SUFFIX = ".lance"
_CACHE_HINT = "rerun with --overwrite or a different --work-dir."

# Canonical ordering shared across staging, clustering, and export so
# positional cluster labels always bind to the exact same spectra.
CANONICAL_SORT_COLUMNS: tuple[str, ...] = (
    "precursor_mz",
    "retention_time",
    "identifier",
)
METADATA_COLUMNS: tuple[str, ...] = (
    "identifier",
    "precursor_charge",
    "precursor_mz",
    "retention_time",
)
INTERVAL_COLUMNS: tuple[str, ...] = (
    "identifier",
    "precursor_mz",
    "precursor_charge",
    "retention_time",
    "mz",
    "intensity",
)
SPECTRUM_SCHEMA = pa.schema(
    [
        pa.field("identifier", pa.string()),
        pa.field("precursor_mz", pa.float32()),
        pa.field("precursor_charge", pa.int8()),
        pa.field("mz", pa.list_(pa.float32())),
        pa.field("intensity", pa.list_(pa.float32())),
        pa.field("retention_time", pa.float32()),
    ]
)

# Flush a bucket's buffered spectra to its Lance dataset once this many
# rows accumulate, bounding per-bucket memory during staging.
_LANCE_WRITE_BATCH_SIZE = 10_000
# Maximum spectra buffered between file reader processes and Lance
# writer threads, bounding peak memory during preprocessing.
_MAX_SPECTRA_IN_MEMORY = 1_000_000


def spectra_dir(work_dir: Path) -> Path:
    """Return the ``spectra`` staging directory under ``work_dir``."""
    return work_dir / SPECTRA_SUBDIR


def bucket_key_to_str(
    bucket_key: ChargeBucket | set[ChargeValue] | str,
) -> str:
    """
    Convert a charge bucket key to a deterministic filename suffix.

    Parameters
    ----------
    bucket_key : ChargeBucket | set[ChargeValue] | str
        Charge bucket key as a tuple/set of charges or sentinel string.

    Returns
    -------
    str
        Filename suffix starting with ``"_"``.

    Examples
    --------
    >>> bucket_key_to_str((1, 2))
    '_1_2'
    >>> bucket_key_to_str((3, "unknown"))
    '_3_unknown'
    >>> bucket_key_to_str("other")
    '_other'
    """
    if isinstance(bucket_key, set):
        bucket_key = tuple(sorted(bucket_key, key=charge_sort_key))
    if isinstance(bucket_key, tuple):
        return "_" + "_".join(str(part) for part in bucket_key)
    if isinstance(bucket_key, str):
        return "_" + bucket_key
    raise TypeError(f"Unsupported bucket key type: {type(bucket_key)}")


def bucket_dataset_name(bucket: ChargeBucket) -> str:
    """Return the on-disk Lance directory name for ``bucket``."""
    return f"spectra_charge{bucket_key_to_str(bucket)}{_BUCKET_SUFFIX}"


def bucket_dataset_path(work_dir: Path, bucket: ChargeBucket) -> Path:
    """Return the full staged Lance dataset path for ``bucket``."""
    return spectra_dir(work_dir) / bucket_dataset_name(bucket)


def format_bucket_label(bucket: ChargeBucket) -> str:
    """
    Format a charge bucket key as a readable label for log messages.

    For example, ``(2,)`` becomes ``"2"``, ``(1, 2)`` becomes
    ``"1, 2"``, ``("unknown",)`` becomes ``"unknown"``, and ``"other"``
    becomes ``"other"``.
    """
    if isinstance(bucket, str):
        return bucket
    return ", ".join(str(part) for part in bucket)


def resolve_input_files(patterns: Sequence[str]) -> list[Path]:
    """
    Expand input peak-file glob patterns in deterministic sorted order.

    Matches within each pattern are sorted lexicographically so staging
    order and cache provenance do not depend on filesystem iteration
    order. Repeated matches across patterns are preserved.

    Parameters
    ----------
    patterns : Sequence[str]
        File paths or glob patterns from ``config.io.input_filenames``.

    Returns
    -------
    list[Path]
        Expanded peak file paths.

    Raises
    ------
    FileNotFoundError
        If no files match any supplied pattern.
    """
    resolved = [
        Path(fn) for pattern in patterns for fn in sorted(glob.glob(pattern))
    ]
    if not resolved:
        raise FileNotFoundError(
            f"No input peak files matched: {list(patterns)}"
        )
    return resolved


# ----------------------------------------------------------------------
# Staging cache manifest serialization and validation
# ----------------------------------------------------------------------


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
        if bucket == "other":
            serialized.append("other")
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
    parsed = parse_charge_buckets(raw_buckets)
    if parsed is None:
        raise ValueError("Cached staging manifest has invalid 'buckets' field")
    return list(parsed)


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
        resolved = input_path.resolve()
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
    if not isinstance(cached, Mapping) or cached.get(
        "schema_version"
    ) != expected.get("schema_version"):
        return ["schema_version"]

    return [
        key
        for key in (
            "inputs",
            "preprocessing",
            "fragment_tol",
            "precursor_charge_buckets",
        )
        if cached.get(key) != expected.get(key)
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
        with os.fdopen(fd, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, indent=2)
            f_out.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _has_unverified_staging_files(target_dir: Path) -> bool:
    """Return True if ``target_dir`` has unmanifested cache files."""
    return any(
        entry.name == "charges.joblib" or entry.name.endswith(_BUCKET_SUFFIX)
        for entry in target_dir.iterdir()
    )


def _staged_dataset_names(target_dir: Path) -> set[str] | None:
    """Return staged ``spectra_charge_*.lance`` dir names, or None."""
    names: set[str] = set()
    for entry in target_dir.iterdir():
        if entry.name.startswith(_BUCKET_PREFIX) and entry.name.endswith(
            _BUCKET_SUFFIX
        ):
            if not entry.is_dir():
                return None
            names.add(entry.name)
    return names


def _clear_directory(target_dir: Path) -> None:
    """Remove all files and subdirectories inside ``target_dir``."""
    for entry in target_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _load_validated_cache(
    cache_path: Path,
    expected_manifest: Mapping[str, Any],
    target_dir: Path,
    work_dir: Path,
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
    resolved_inputs = resolve_input_files(config.io.input_filenames)
    expected_manifest = build_staging_manifest(config, resolved_inputs)
    target_dir = spectra_dir(work_dir)
    cache_path = target_dir / STAGING_CACHE_FILENAME

    if config.io.overwrite:
        _clear_directory(target_dir)
    elif cache_path.is_file():
        return _load_validated_cache(
            cache_path, expected_manifest, target_dir, work_dir
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


# ----------------------------------------------------------------------
# Parallel spectrum reading and concurrent Lance dataset staging
# ----------------------------------------------------------------------


class _PerChargeLockRegistry:
    """Lazily created per-bucket locks for concurrent Lance writes."""

    def __init__(self) -> None:
        self._locks: dict[object, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: object) -> threading.Lock:
        """Return the lock for ``key``, creating it lazily if needed."""
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock


def _build_charge_routing(
    charge_buckets: Sequence[Any] | None,
) -> tuple[
    bool,
    tuple[ChargeBucket, ...] | None,
    dict[ChargeValue, tuple[ChargeValue, ...]],
    bool,
]:
    """Validate charge buckets and build the per-charge routing map."""
    if charge_buckets is None:
        return True, None, {}, False
    canonical = parse_charge_buckets(list(charge_buckets))
    assert canonical is not None  # noqa: S101
    charge_to_bucket: dict[ChargeValue, tuple[ChargeValue, ...]] = {}
    catch_other_charges = False
    for bucket in canonical:
        if bucket == "other":
            catch_other_charges = True
            continue
        for charge in bucket:
            charge_to_bucket[charge] = bucket
    return False, canonical, charge_to_bucket, catch_other_charges


def stage_spectra(
    input_filenames: Sequence[str | os.PathLike[str]],
    work_dir: str | os.PathLike[str],
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
    charge_buckets: Sequence[Any] | None,
) -> list[ChargeBucket]:
    """
    Read, preprocess, and stage spectra in per-charge Lance datasets.

    Parameters
    ----------
    input_filenames : Sequence[str | os.PathLike[str]]
        Resolved peak file paths to read.
    work_dir : str | os.PathLike[str]
        Working directory containing the ``spectra`` staging subfolder.
    process_spectrum : Callable
        Preprocessing function returning a ``ProcessedSpectrum`` or
        ``None`` if the spectrum fails quality gates.
    charge_buckets : Sequence[Any] | None
        Explicit charge groupings or ``None`` for auto mode (one bucket
        per observed charge).

    Returns
    -------
    list[ChargeBucket]
        Charge buckets for which a Lance dataset was created.
    """
    resolved_files = [Path(fn) for fn in input_filenames]
    if not resolved_files:
        raise FileNotFoundError("No input peak files supplied")

    work_path = Path(work_dir)
    logger.info("Reading spectra from %d peak file(s)...", len(resolved_files))
    effective_cpus = max(1, int(joblib.effective_n_jobs(-1)))
    max_file_workers = min(len(resolved_files), effective_cpus)

    auto_bucket, canonical_buckets, charge_to_bucket, catch_other_charges = (
        _build_charge_routing(charge_buckets)
    )

    spectra_queue: queue.Queue[ProcessedSpectrum | None] = queue.Queue(
        maxsize=_MAX_SPECTRA_IN_MEMORY
    )
    lance_locks = _PerChargeLockRegistry()
    stop_event = threading.Event()
    writer_errors: list[BaseException] = []
    error_lock = threading.Lock()

    def _writer_loop() -> None:
        try:
            _write_spectra_lance(
                spectra_queue,
                lance_locks,
                charge_to_bucket,
                catch_other_charges,
                auto_bucket,
                work_path,
                stop_event,
            )
        except BaseException as exc:
            with error_lock:
                writer_errors.append(exc)
            stop_event.set()

    writers = [
        threading.Thread(target=_writer_loop, daemon=True)
        for _ in range(max_file_workers)
    ]
    for writer in writers:
        writer.start()

    low_quality_counter = 0
    try:
        for file_spectra, lqc in joblib.Parallel(
            n_jobs=max_file_workers, return_as="generator_unordered"
        )(
            joblib.delayed(_read_spectra)(file, process_spectrum)
            for file in resolved_files
        ):
            low_quality_counter += lqc
            for spec in file_spectra:
                while not stop_event.is_set():
                    try:
                        spectra_queue.put(spec, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop_event.is_set():
                    break
            if stop_event.is_set():
                break
    except BaseException:
        stop_event.set()
        raise
    finally:
        if not stop_event.is_set():
            for _ in writers:
                spectra_queue.put(None)
        for writer in writers:
            writer.join()

    if writer_errors:
        raise writer_errors[0]

    target_dir = spectra_dir(work_path)
    candidate_buckets: Sequence[ChargeBucket] = (
        _discover_auto_buckets(target_dir)
        if canonical_buckets is None
        else canonical_buckets
    )
    n_spectra = 0
    valid_buckets: list[ChargeBucket] = []
    for bucket in candidate_buckets:
        dataset_path = target_dir / bucket_dataset_name(bucket)
        try:
            dataset = lance.dataset(str(dataset_path))
            n_spectra += dataset.count_rows()
            valid_buckets.append(bucket)
        except (ValueError, FileNotFoundError):
            logger.debug("No spectra for bucket %s, skipping", bucket)
    logger.info(
        "Read %d spectra from %d peak file(s)",
        n_spectra,
        len(resolved_files),
    )
    logger.info("Skipped %d low-quality spectra", low_quality_counter)
    return valid_buckets


def _discover_auto_buckets(
    lance_dir: str | os.PathLike[str],
) -> list[tuple[ChargeValue, ...]]:
    """
    Discover singleton per-charge buckets created in auto mode.

    Returns buckets sorted deterministically (numeric charges ascending,
    then ``"unknown"``).
    """
    buckets: list[tuple[ChargeValue, ...]] = []
    for entry in Path(lance_dir).iterdir():
        name = entry.name
        if not (
            name.startswith(_BUCKET_PREFIX) and name.endswith(_BUCKET_SUFFIX)
        ):
            continue
        key = name[len(_BUCKET_PREFIX) : -len(_BUCKET_SUFFIX)]
        bucket: tuple[ChargeValue, ...] = (
            ("unknown",) if key == "unknown" else (int(key),)
        )
        buckets.append(bucket)
    return sorted(buckets, key=lambda b: charge_sort_key(b[0]))


def _read_spectra(
    filename: str | os.PathLike[str],
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> tuple[list[ProcessedSpectrum], int]:
    """Read and preprocess all spectra from a single peak file."""
    low_quality_counter = 0
    spectra: list[ProcessedSpectrum] = []
    abs_path = Path(filename).resolve()
    for spec in ms_io.get_spectra(abs_path):
        processed = process_spectrum(spec)
        if processed is None:
            low_quality_counter += 1
        else:
            spectra.append(processed)
    return spectra, low_quality_counter


def _write_spectra_lance(
    spectra_queue: queue.Queue[ProcessedSpectrum | None],
    lance_locks: _PerChargeLockRegistry,
    charge_to_bucket: Mapping[ChargeValue, tuple[ChargeValue, ...]],
    catch_other_charges: bool,
    auto_bucket: bool,
    work_dir: Path,
    stop_event: threading.Event | None = None,
) -> None:
    """Drain preprocessed spectra from ``spectra_queue`` into Lance."""
    spec_to_write: dict[ChargeBucket, list[ProcessedSpectrum]] = (
        collections.defaultdict(list)
    )

    def _flush(key: ChargeBucket) -> None:
        batch = spec_to_write[key]
        if not batch:
            return
        _write_to_dataset(batch, key, lance_locks.get(key), work_dir)
        batch.clear()

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            spec = spectra_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if spec is None:
            for key in list(spec_to_write):
                _flush(key)
            return

        raw_charge = spec["precursor_charge"]
        charge: ChargeValue = "unknown" if raw_charge is None else raw_charge

        bucket_key: ChargeBucket | None
        if auto_bucket:
            bucket_key = (charge,)
        else:
            bucket_key = charge_to_bucket.get(charge) or (
                "other" if catch_other_charges else None
            )
        if bucket_key is not None:
            spec_to_write[bucket_key].append(spec)
            if len(spec_to_write[bucket_key]) >= _LANCE_WRITE_BATCH_SIZE:
                _flush(bucket_key)


def _write_to_dataset(
    spec_to_write: Sequence[Mapping[str, Any]],
    charge_bucket: ChargeBucket,
    lock: threading.Lock,
    work_dir: str | os.PathLike[str],
) -> int:
    """Create or append a batch of preprocessed spectra to Lance."""
    new_rows = pa.Table.from_pylist(spec_to_write, SPECTRUM_SCHEMA)
    path = bucket_dataset_path(Path(work_dir), charge_bucket)
    with lock:
        if not path.exists():
            logger.debug("Creating lance dataset at %s", path)
            lance.write_dataset(
                new_rows,
                str(path),
                mode="create",
                data_storage_version="stable",
            )
        else:
            lance.write_dataset(new_rows, str(path), mode="append")
    return len(new_rows)
