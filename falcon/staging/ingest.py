"""Input discovery, charge routing, and batch-routed Lance ingestion."""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib
import lance
import numpy as np
import pyarrow as pa

from ..dataset import (
    SPECTRUM_SCHEMA,
    bucket_dataset_path,
    spectra_dir,
)
from ..types import INT32_MAX, charge_sort_key, parse_charge_bucket_list
from ..utils import effective_n_workers, expand_input_globs, resolve_path

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

    import spectrum_utils.spectrum as sus

    from ..types import ChargeBucket, ChargeValue, ProcessedSpectrum

logger = logging.getLogger(__name__)

# Flush a bucket's buffered spectra to its Lance dataset once this many
# rows accumulate, bounding per-bucket memory during staging.
_LANCE_WRITE_BATCH_SIZE = 10_000

__all__ = [
    "ChargeRouter",
    "resolve_input_files",
    "stage_spectra",
]


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
        If ``patterns`` is empty, any supplied pattern matches no
        files, or any matched path is not an existing regular file
        (e.g. a broken symlink).
    IsADirectoryError
        If any literal path or glob match refers to a directory.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        matches = expand_input_globs([pattern])
        if not matches:
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


class ChargeRouter:
    """
    Route raw spectrum charges to canonical ``ChargeBucket`` keys.

    Supports automatic per-charge bucketing (``charge_buckets=None``)
    and explicit non-overlapping charge groupings with optional
    ``"other"`` fallback.

    Parameters
    ----------
    charge_buckets : Sequence[Any] | None
        Explicit charge groupings or ``None`` for auto mode.

    Raises
    ------
    ValueError
        If ``charge_buckets`` is non-``None`` and invalid.
    """

    def __init__(self, charge_buckets: Sequence[Any] | None) -> None:
        if charge_buckets is None:
            self._canonical_buckets: tuple[ChargeBucket, ...] | None = None
            self._charge_to_bucket: dict[
                ChargeValue, tuple[ChargeValue, ...]
            ] = {}
            self._catch_other = False
            return

        canonical = parse_charge_bucket_list(list(charge_buckets))
        charge_to_bucket: dict[ChargeValue, tuple[ChargeValue, ...]] = {}
        catch_other = False
        for bucket in canonical:
            if bucket == "other":
                catch_other = True
                continue
            for charge in bucket:
                charge_to_bucket[charge] = bucket
        self._canonical_buckets = canonical
        self._charge_to_bucket = charge_to_bucket
        self._catch_other = catch_other

    @property
    def auto(self) -> bool:
        """Return ``True`` when routing in automatic per-charge mode."""
        return self._canonical_buckets is None

    @property
    def canonical_buckets(self) -> tuple[ChargeBucket, ...] | None:
        """Return explicit buckets, or ``None`` in auto mode."""
        return self._canonical_buckets

    def route(
        self, raw_charge: int | np.integer[Any] | None
    ) -> ChargeBucket | None:
        """
        Map a raw precursor charge to its target ``ChargeBucket``.

        Parameters
        ----------
        raw_charge : int | np.integer | None
            Precursor charge state, or ``None`` when unknown.

        Returns
        -------
        ChargeBucket | None
            Target charge bucket key, or ``None`` if the spectrum should
            be dropped.
        """
        charge: ChargeValue = (
            "unknown" if raw_charge is None else int(raw_charge)
        )
        if self._canonical_buckets is None:
            return (charge,)
        bucket = self._charge_to_bucket.get(charge)
        if bucket is not None:
            return bucket
        return "other" if self._catch_other else None


def _auto_bucket_sort_key(bucket: ChargeBucket) -> tuple[bool, int | str]:
    """Sort auto-mode singleton charge buckets deterministically."""
    return charge_sort_key(cast("tuple[ChargeValue, ...]", bucket)[0])


def _collect_valid_buckets(
    router: ChargeRouter,
    written_counts: Mapping[ChargeBucket, int],
) -> list[ChargeBucket]:
    """Return non-empty staged buckets in deterministic order."""
    if router.canonical_buckets is None:
        return sorted(
            (b for b, count in written_counts.items() if count > 0),
            key=_auto_bucket_sort_key,
        )
    valid: list[ChargeBucket] = []
    for bucket in router.canonical_buckets:
        if written_counts.get(bucket, 0) > 0:
            valid.append(bucket)
        else:
            logger.debug("No spectra for bucket %s, skipping", bucket)
    return valid


def _build_list_array(arrays: Sequence[np.ndarray]) -> pa.ListArray:
    """Pack a sequence of 1D float32 arrays into a PyArrow ListArray."""
    n = len(arrays)
    offsets_i64 = np.empty(n + 1, dtype=np.int64)
    offsets_i64[0] = 0
    if n == 0:
        flat = np.empty(0, dtype=np.float32)
    else:
        lengths = np.fromiter(
            (len(a) for a in arrays), dtype=np.int64, count=n
        )
        np.cumsum(lengths, dtype=np.int64, out=offsets_i64[1:])
        total_values = int(offsets_i64[-1])
        if total_values > INT32_MAX:
            raise OverflowError(
                f"Total peak count ({total_values}) exceeds PyArrow "
                f"int32 ListArray offset limit ({INT32_MAX})"
            )
        flat = np.concatenate(arrays)
    offsets = offsets_i64.astype(np.int32, copy=False)
    return pa.ListArray.from_arrays(offsets, pa.array(flat, type=pa.float32()))


def _spectra_to_arrow_table(
    spec_to_write: Sequence[Mapping[str, Any]],
) -> pa.Table:
    """Build a columnar PyArrow table matching ``SPECTRUM_SCHEMA``."""
    n = len(spec_to_write)
    mz_arrays = [np.asarray(s["mz"], dtype=np.float32) for s in spec_to_write]
    int_arrays = [
        np.asarray(s["intensity"], dtype=np.float32) for s in spec_to_write
    ]
    precursor_mz = np.fromiter(
        (s["precursor_mz"] for s in spec_to_write), dtype=np.float32, count=n
    )
    return pa.table(
        {
            "identifier": pa.array(
                [s["identifier"] for s in spec_to_write], type=pa.string()
            ),
            "precursor_mz": pa.array(precursor_mz, type=pa.float32()),
            "precursor_charge": pa.array(
                [s["precursor_charge"] for s in spec_to_write], type=pa.int8()
            ),
            "mz": _build_list_array(mz_arrays),
            "intensity": _build_list_array(int_arrays),
            "retention_time": pa.array(
                [s["retention_time"] for s in spec_to_write],
                type=pa.float32(),
            ),
        },
        schema=SPECTRUM_SCHEMA,
    )


def _write_to_dataset(
    spec_to_write: Sequence[Mapping[str, Any]],
    charge_bucket: ChargeBucket,
    lock: threading.Lock,
    work_dir: Path,
    *,
    mode: Literal["overwrite", "append"] = "append",
) -> int:
    """Create, overwrite, or append a batch of spectra to Lance."""
    new_rows = _spectra_to_arrow_table(spec_to_write)
    path = bucket_dataset_path(work_dir, charge_bucket)
    with lock:
        if mode == "overwrite":
            effective_mode = "overwrite" if path.exists() else "create"
            logger.debug(
                "Writing lance dataset at %s (mode=%s)", path, effective_mode
            )
            lance.write_dataset(
                new_rows,
                str(path),
                mode=effective_mode,
                data_storage_version="stable",
            )
        else:
            lance.write_dataset(new_rows, str(path), mode="append")
    return len(new_rows)


class _LanceBatchWriter:
    """
    Route preprocessed spectra into per-bucket batches for Lance.

    Maintains a single pending buffer per active charge bucket on the
    main thread and dispatches full batches to a bounded thread pool.
    Each bucket's first batch writes in ``"overwrite"`` mode and
    completes before any ``"append"`` batch for that bucket is
    submitted, preventing duplicate rows on repeated staging runs and
    eliminating write-order races.
    """

    def __init__(
        self,
        router: ChargeRouter,
        work_dir: Path,
        max_workers: int,
    ) -> None:
        self._router = router
        self._work_dir = work_dir
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._inflight_sem = threading.BoundedSemaphore(
            max(2, max_workers * 2)
        )
        self._locks: dict[ChargeBucket, threading.Lock] = (
            collections.defaultdict(threading.Lock)
        )
        self._pending: dict[ChargeBucket, list[ProcessedSpectrum]] = (
            collections.defaultdict(list)
        )
        self._initialized: set[ChargeBucket] = set()
        self._init_futures: dict[
            ChargeBucket, concurrent.futures.Future[int]
        ] = {}
        self._inflight: dict[concurrent.futures.Future[int], ChargeBucket] = {}
        self._written_counts: dict[ChargeBucket, int] = (
            collections.defaultdict(int)
        )
        self._excluded_count = 0

    @property
    def excluded_count(self) -> int:
        """Return count of spectra excluded by charge buckets."""
        return self._excluded_count

    @property
    def written_counts(self) -> dict[ChargeBucket, int]:
        """Return per-bucket counts of spectra committed to Lance."""
        return dict(self._written_counts)

    def __enter__(self) -> _LanceBatchWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            return
        try:
            self._flush_and_wait()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._executor.shutdown(wait=True)

    def _consume_future(self, fut: concurrent.futures.Future[int]) -> None:
        """Collect one completed future count or re-raise its error."""
        bucket = self._inflight.pop(fut)
        self._written_counts[bucket] += fut.result()

    def _reap_completed(self) -> None:
        """Collect finished futures promptly to surface errors early."""
        for fut in [f for f in self._inflight if f.done()]:
            self._consume_future(fut)

    def _submit(
        self, bucket: ChargeBucket, batch: list[ProcessedSpectrum]
    ) -> None:
        """Dispatch one bucket batch to the bounded writer pool."""
        self._reap_completed()
        if bucket in self._initialized:
            init_fut = self._init_futures.pop(bucket, None)
            if init_fut is not None and init_fut in self._inflight:
                self._consume_future(init_fut)
            mode: Literal["overwrite", "append"] = "append"
        else:
            mode = "overwrite"

        self._inflight_sem.acquire()
        lock = self._locks[bucket]

        def _task() -> int:
            try:
                return _write_to_dataset(
                    batch, bucket, lock, self._work_dir, mode=mode
                )
            finally:
                self._inflight_sem.release()

        try:
            fut = self._executor.submit(_task)
        except BaseException:
            self._inflight_sem.release()
            raise
        self._inflight[fut] = bucket
        if mode == "overwrite":
            self._initialized.add(bucket)
            self._init_futures[bucket] = fut

    def add_spectrum(self, spec: ProcessedSpectrum) -> None:
        """Route one spectrum into its buffer, flushing when full."""
        bucket = self._router.route(spec["precursor_charge"])
        if bucket is None:
            self._excluded_count += 1
            return
        buf = self._pending[bucket]
        buf.append(spec)
        if len(buf) >= _LANCE_WRITE_BATCH_SIZE:
            self._pending[bucket] = []
            self._submit(bucket, buf)

    def add_spectra(self, spectra: Sequence[ProcessedSpectrum]) -> None:
        """Route ``spectra`` into buffers, flushing full batches."""
        self._reap_completed()
        for spec in spectra:
            self.add_spectrum(spec)

    def _flush_and_wait(self) -> None:
        """Flush remaining buffered batches and join inflight writes."""
        for bucket, buf in list(self._pending.items()):
            if buf:
                self._pending[bucket] = []
                self._submit(bucket, buf)
        for fut in concurrent.futures.as_completed(list(self._inflight)):
            self._consume_future(fut)


def _read_spectra(
    filename: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> tuple[list[ProcessedSpectrum], int]:
    """Read and preprocess all spectra from a single peak file."""
    from .. import ms_io

    low_quality_counter = 0
    spectra: list[ProcessedSpectrum] = []
    abs_path = resolve_path(filename)
    for spec in ms_io.get_spectra(abs_path):
        processed = process_spectrum(spec)
        if processed is None:
            low_quality_counter += 1
        else:
            spectra.append(processed)
    return spectra, low_quality_counter


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
    # Resolve in the parent process so warm loky workers do not look up
    # relative paths against a stale working directory after chdir().
    resolved_files = [resolve_path(Path(fn)) for fn in input_filenames]
    if not resolved_files:
        raise FileNotFoundError("No input peak files supplied")

    router = ChargeRouter(charge_buckets)
    work_path = Path(work_dir)
    spectra_dir(work_path).mkdir(parents=True, exist_ok=True)
    logger.info("Reading spectra from %d peak file(s)...", len(resolved_files))
    max_file_workers = min(len(resolved_files), effective_n_workers())

    low_quality_counter = 0
    with _LanceBatchWriter(router, work_path, max_file_workers) as writer:
        if len(resolved_files) == 1:
            from .. import ms_io

            for spec in ms_io.get_spectra(str(resolved_files[0])):
                processed = process_spectrum(spec)
                if processed is None:
                    low_quality_counter += 1
                else:
                    writer.add_spectrum(processed)
        else:
            parallel_iter = joblib.Parallel(
                n_jobs=max_file_workers,
                backend="loky",
                return_as="generator_unordered",
            )(
                joblib.delayed(_read_spectra)(file, process_spectrum)
                for file in resolved_files
            )
            with contextlib.closing(parallel_iter) as file_results:
                for file_spectra, lqc in file_results:
                    low_quality_counter += lqc
                    writer.add_spectra(file_spectra)

    written_counts = writer.written_counts
    n_spectra = sum(written_counts.values())
    valid_buckets = _collect_valid_buckets(router, written_counts)
    logger.info(
        "Staged %d spectra from %d peak file(s)",
        n_spectra,
        len(resolved_files),
    )
    logger.info("Skipped %d low-quality spectra", low_quality_counter)
    if writer.excluded_count > 0:
        logger.info(
            "Excluded %d spectra outside configured charge buckets",
            writer.excluded_count,
        )
    return valid_buckets
