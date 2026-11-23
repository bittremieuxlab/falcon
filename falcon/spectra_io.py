"""Read peak files, stage Lance datasets, and write CSV assignments.

This module implements the spectrum and Lance IO layer orchestrated by
``falcon.py``: deterministic input glob expansion, parallel peak-file
reading, charge-bucket routing, per-charge Lance dataset staging, and
cluster-assignment CSV export.
"""

from __future__ import annotations

import collections
import glob
import logging
import multiprocessing
import multiprocessing.pool
import os
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import lance
import pandas as pd
import pyarrow as pa
import spectrum_utils.spectrum as sus

from .cluster.spectrum import ProcessedSpectrum
from .config import FalconConfig
from .ms_io import ms_io

logger = logging.getLogger("falcon")

BucketKey = set[int | str] | tuple[int | str, ...] | str

# Flush a bucket's buffered spectra to its Lance dataset once this many
# rows accumulate, bounding per-bucket memory during staging.
_LANCE_WRITE_BATCH_SIZE = 10_000
# Maximum spectra buffered between file reader processes and Lance
# writer threads, bounding peak memory during preprocessing.
_MAX_SPECTRA_IN_MEMORY = 1_000_000


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


def _prepare_spectra(
    input_filenames: Sequence[str | os.PathLike[str]],
    work_dir: str | os.PathLike[str],
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
    charge_buckets: Sequence[BucketKey] | None,
) -> list[BucketKey]:
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
    charge_buckets : Sequence[BucketKey] | None
        Explicit charge groupings (tuples/sets of charges/``"unknown"``
        or ``"other"``), or ``None`` for auto mode (one bucket per
        observed charge).

    Returns
    -------
    list[BucketKey]
        Charge buckets for which a non-missing Lance dataset was
        created.
    """
    resolved_files = [os.fspath(fn) for fn in input_filenames]
    if not resolved_files:
        raise FileNotFoundError("No input peak files supplied")
    logger.info("Reading spectra from %d peak file(s)...", len(resolved_files))
    # Use at most one worker per input file up to the CPU count.
    max_file_workers = min(len(resolved_files), multiprocessing.cpu_count())
    # Bound buffered spectra to avoid excessive staging memory usage.
    spectra_queue: queue.Queue[ProcessedSpectrum | None] = queue.Queue(
        maxsize=_MAX_SPECTRA_IN_MEMORY
    )
    # Per-bucket locks serialize writes within each Lance dataset while
    # allowing different charge buckets to flush concurrently.
    lance_locks = _PerChargeLockRegistry()
    schema = pa.schema(
        [
            pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int8()),
            pa.field("mz", pa.list_(pa.float32())),
            pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
        ]
    )
    # In auto mode (charge_buckets is None), every observed charge
    # (including missing charge) gets its own singleton bucket.
    auto_bucket = charge_buckets is None
    charge_to_bucket: dict[int | str, tuple[int | str, ...]] = {}
    catch_other_charges = False
    if charge_buckets is not None:
        for bucket in charge_buckets:
            if bucket == "other":
                if catch_other_charges:
                    raise ValueError(
                        "Charge bucket 'other' appears more than once"
                    )
                catch_other_charges = True
                continue  # Unmatched charges are routed later.
            for charge in bucket:
                if charge in charge_to_bucket:
                    raise ValueError(
                        f"Charge {charge} appears in more than one bucket"
                    )
                bucket_tuple = tuple(
                    sorted(bucket, key=lambda x: (isinstance(x, str), x))
                )
                charge_to_bucket[charge] = bucket_tuple

    work_dir_str = os.fspath(work_dir)
    lance_writers = multiprocessing.pool.ThreadPool(
        max_file_workers,
        _write_spectra_lance,
        (
            spectra_queue,
            lance_locks,
            schema,
            charge_to_bucket,
            catch_other_charges,
            auto_bucket,
            work_dir_str,
        ),
    )
    # Stream processed file results into the queue as workers finish
    # (`generator_unordered`) so `maxsize` bounds peak memory.
    low_quality_counter = 0
    for file_spectra, lqc in joblib.Parallel(
        n_jobs=max_file_workers, return_as="generator_unordered"
    )(
        joblib.delayed(_read_spectra)(file, process_spectrum)
        for file in resolved_files
    ):
        low_quality_counter += lqc
        for spec in file_spectra:
            spectra_queue.put(spec)
    # Push one sentinel per writer thread to signal clean shutdown.
    for _ in range(max_file_workers):
        spectra_queue.put(None)
    lance_writers.close()
    lance_writers.join()

    # Count surviving spectra across all created Lance datasets.
    lance_dir = os.path.join(work_dir_str, "spectra")
    # In auto mode, buckets are discovered from the written datasets.
    candidate_buckets: Sequence[BucketKey] = (
        _discover_auto_buckets(lance_dir)
        if charge_buckets is None
        else charge_buckets
    )
    n_spectra = 0
    valid_buckets: list[BucketKey] = []
    for bucket in candidate_buckets:
        dataset_path = os.path.join(
            lance_dir, f"spectra_charge{bucket_key_to_str(bucket)}.lance"
        )
        try:
            dataset = lance.dataset(dataset_path)
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
) -> list[tuple[int | str, ...]]:
    """
    Discover per-charge buckets created in auto mode.

    In auto mode every distinct charge is written to its own singleton
    bucket, so the bucket list is reconstructed from the
    ``spectra_charge<key>.lance`` directories written to disk.

    Parameters
    ----------
    lance_dir : str | os.PathLike[str]
        Directory containing the per-charge Lance datasets.

    Returns
    -------
    list[tuple[int | str, ...]]
        Discovered singleton buckets sorted deterministically (numeric
        charges ascending, then ``"unknown"``).
    """
    prefix, suffix = "spectra_charge_", ".lance"
    buckets: list[tuple[int | str, ...]] = []
    for name in os.listdir(lance_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        key = name[len(prefix) : -len(suffix)]
        bucket: tuple[int | str, ...] = (
            (key,) if key == "unknown" else (int(key),)
        )
        buckets.append(bucket)
    return sorted(buckets, key=lambda b: (isinstance(b[0], str), b[0]))


def _create_lance_dataset(
    charge_bucket: BucketKey,
    schema: pa.Schema,
    work_dir: str | os.PathLike[str],
) -> lance.LanceDataset:
    """
    Create an empty Lance dataset for a charge bucket.

    Parameters
    ----------
    charge_bucket : BucketKey
        Charge bucket key identifying the dataset path.
    schema : pa.Schema
        PyArrow schema of the Lance table.
    work_dir : str | os.PathLike[str]
        Working directory containing the ``spectra`` staging folder.

    Returns
    -------
    lance.LanceDataset
        The newly created empty Lance dataset.
    """
    lance_path = os.path.join(
        os.fspath(work_dir),
        "spectra",
        f"spectra_charge{bucket_key_to_str(charge_bucket)}.lance",
    )
    # Use mode="create" (only called when the path does not yet exist)
    # to avoid a redundant Lance warning on overwrite/append.
    dataset = lance.write_dataset(
        pa.Table.from_pylist([], schema),
        lance_path,
        mode="create",
        data_storage_version="stable",
    )
    logger.debug("Creating lance dataset at %s", lance_path)
    return dataset


def _read_spectra(
    filename: str,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> tuple[list[ProcessedSpectrum], int]:
    """
    Read and preprocess all spectra from a single peak file.

    Parameters
    ----------
    filename : str
        Path to the input peak file.
    process_spectrum : Callable
        Function to preprocess each ``MsmsSpectrum`` into a
        ``ProcessedSpectrum`` or ``None``.

    Returns
    -------
    tuple[list[ProcessedSpectrum], int]
        Accepted preprocessed spectrum dictionaries and the count of
        rejected low-quality spectra.
    """
    low_quality_counter = 0
    spectra: list[ProcessedSpectrum] = []
    filename = os.path.abspath(filename)
    for spec in ms_io.get_spectra(filename):
        processed = process_spectrum(spec)
        if processed is None:
            low_quality_counter += 1
        else:
            spectra.append(processed)
    return spectra, low_quality_counter


class _PerChargeLockRegistry:
    """
    Lazily created per-bucket locks for concurrent Lance writes.

    Serializes writes targeting the same charge bucket while allowing
    independent buckets to flush in parallel.
    """

    def __init__(self) -> None:
        """Initialize an empty thread-safe lock registry."""
        self._locks: dict[object, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: object) -> threading.Lock:
        """
        Return the lock associated with ``key``, creating it if needed.

        Parameters
        ----------
        key : object
            Hashable bucket identifier.

        Returns
        -------
        threading.Lock
            Lock guarding writes to the bucket's Lance dataset.
        """
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock


def _write_spectra_lance(
    spectra_queue: queue.Queue[ProcessedSpectrum | None],
    lance_locks: _PerChargeLockRegistry,
    schema: pa.Schema,
    charge_to_bucket: dict[int | str, tuple[int | str, ...]],
    catch_other_charges: bool,
    auto_bucket: bool,
    work_dir: str | os.PathLike[str],
) -> None:
    """
    Consume preprocessed spectra from a queue and write Lance datasets.

    Parameters
    ----------
    spectra_queue : queue.Queue[ProcessedSpectrum | None]
        Queue yielding spectrum dictionaries and ``None`` sentinels.
    lance_locks : _PerChargeLockRegistry
        Per-bucket locks synchronizing writes within each dataset.
    schema : pa.Schema
        PyArrow schema for the staged Lance tables.
    charge_to_bucket : dict[int | str, tuple[int | str, ...]]
        Mapping from precursor charge (or ``"unknown"``) to the sorted
        bucket tuple sharing a dataset.
    catch_other_charges : bool
        Whether unmapped charges route to the ``"other"`` bucket.
    auto_bucket : bool
        Whether every distinct charge gets its own singleton bucket.
        Takes precedence over ``charge_to_bucket`` and
        ``catch_other_charges``.
    work_dir : str | os.PathLike[str]
        Working directory containing the ``spectra`` staging folder.
    """
    work_dir_str = os.fspath(work_dir)
    spec_to_write: dict[BucketKey, list[ProcessedSpectrum]] = (
        collections.defaultdict(list)
    )
    while True:
        spec = spectra_queue.get()
        if spec is None:
            # Flush all remaining buffered rows before exiting.
            for bucket_key in spec_to_write:
                if len(spec_to_write[bucket_key]) == 0:
                    continue
                _write_to_dataset(
                    spec_to_write[bucket_key],
                    bucket_key,
                    lance_locks.get(bucket_key),
                    schema,
                    work_dir_str,
                )
                spec_to_write[bucket_key].clear()
            return
        raw_charge = spec["precursor_charge"]
        charge: int | str = "unknown" if raw_charge is None else raw_charge

        # Route the spectrum to its target charge bucket.
        bucket_key: BucketKey | None
        if auto_bucket:
            # Cluster every observed charge in its own singleton bucket.
            bucket_key = (charge,)
        else:
            bucket_key = charge_to_bucket.get(charge) or (
                "other" if catch_other_charges else None
            )
        if bucket_key is not None:
            spec_to_write[bucket_key].append(spec)

            if len(spec_to_write[bucket_key]) >= _LANCE_WRITE_BATCH_SIZE:
                _write_to_dataset(
                    spec_to_write[bucket_key],
                    bucket_key,
                    lance_locks.get(bucket_key),
                    schema,
                    work_dir_str,
                )
                spec_to_write[bucket_key].clear()


def _write_to_dataset(
    spec_to_write: Sequence[Mapping[str, Any]],
    charge_bucket: BucketKey,
    lock: threading.Lock,
    schema: pa.Schema,
    work_dir: str | os.PathLike[str],
) -> int:
    """
    Append a batch of preprocessed spectra to a Lance dataset.

    Parameters
    ----------
    spec_to_write : Sequence[Mapping[str, Any]]
        Preprocessed spectrum rows to append.
    charge_bucket : BucketKey
        Bucket key identifying the destination Lance dataset.
    lock : threading.Lock
        Lock synchronizing dataset creation and appends for this bucket.
    schema : pa.Schema
        PyArrow schema of the Lance table.
    work_dir : str | os.PathLike[str]
        Working directory containing the ``spectra`` staging folder.

    Returns
    -------
    int
        Number of spectrum rows written.
    """
    new_rows = pa.Table.from_pylist(spec_to_write, schema)
    bucket_str = bucket_key_to_str(charge_bucket)
    work_dir_str = os.fspath(work_dir)
    path = os.path.join(
        work_dir_str, "spectra", f"spectra_charge{bucket_str}.lance"
    )
    with lock:
        if not os.path.exists(path):
            _create_lance_dataset(charge_bucket, schema, work_dir_str)
        lance.write_dataset(new_rows, path, mode="append")
    return len(new_rows)


def bucket_key_to_str(bucket_key: BucketKey) -> str:
    """
    Convert a charge bucket key to a deterministic filename suffix.

    Parameters
    ----------
    bucket_key : BucketKey
        Charge bucket key as a set, tuple, or sentinel string.

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
        # Sort sets deterministically (numeric charges before strings).
        bucket_key = tuple(
            sorted(bucket_key, key=lambda x: (isinstance(x, str), x))
        )
    if isinstance(bucket_key, tuple):
        parts = [str(p) for p in bucket_key]
        return "_" + "_".join(parts)
    elif isinstance(bucket_key, str):
        return "_" + bucket_key
    else:
        raise TypeError(f"Unsupported bucket key type: {type(bucket_key)}")


def _write_cluster_info(
    clusters: pd.DataFrame,
    config: FalconConfig,
    work_dir: Path,
) -> None:
    """
    Write configuration header comments and cluster assignments to CSV.

    Parameters
    ----------
    clusters : pd.DataFrame
        Cluster assignment table with columns ``identifier``,
        ``precursor_charge``, ``precursor_mz``, ``retention_time``, and
        ``cluster``.
    config : FalconConfig
        Validated run configuration whose metadata is written as ``#``
        header lines.
    work_dir : Path
        Effective working directory (including temporary directory when
        ``config.io.work_dir`` is ``None``).
    """
    csv_path = Path(f"{config.io.output_filename}.csv")
    with csv_path.open("a", encoding="utf-8") as f_out:
        for line in config.to_metadata_lines(work_dir=work_dir):
            f_out.write(f"# {line}\n")
        f_out.write("#\n")
        clusters.to_csv(f_out, index=False, chunksize=1000000)
