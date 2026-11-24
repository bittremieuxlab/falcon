"""Spectrum preprocessing, charge routing, Lance staging, and caching."""

from __future__ import annotations

import collections
import contextlib
import fcntl
import glob
import json
import logging
import math
import os
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import joblib
import lance
import numpy as np
import pyarrow as pa

from . import __version__, ms_io
from .config import (
    INT8_MAX,
    INT8_MIN,
    OTHER_BUCKET,
    UNKNOWN_CHARGE,
    ChargeBucket,
    ChargeValue,
    FalconConfig,
    PathLike,
    PreprocessingConfig,
    charge_sort_key,
    mz_bin_bounds,
    parse_charge_buckets,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import IO

    import pandas as pd
    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

# Match spectrum_utils==0.3.5 proton mass constant (Da) for
# charge-reduced precursor peak removal.
_PROTON_MASS = 1.0072766
_LANCE_WRITE_BATCH_SIZE = 10_000

SPECTRA_SUBDIR = "spectra"
BUCKET_PREFIX = "spectra_charge_"
BUCKET_SUFFIX = ".lance"
STAGING_CACHE_SCHEMA_VERSION = 1
STAGING_CACHE_FILENAME = "staging_cache.json"
STAGING_LOCK_FILENAME = ".staging.lock"

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
SORTED_INTERVAL_COLUMNS: tuple[str, ...] = (
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

_CACHE_HINT = "rerun with --overwrite or a different --work-dir."


class ProcessedSpectrum(TypedDict):
    """Preprocessed spectrum record staged into a Lance dataset."""

    identifier: str
    precursor_mz: float
    precursor_charge: int | None
    mz: np.ndarray
    intensity: np.ndarray
    retention_time: float | None


# ----------------------------------------------------------------------
# Input file resolution and spectrum preprocessing
# ----------------------------------------------------------------------


def resolve_input_files(patterns: Sequence[str]) -> list[Path]:
    """
    Expand input peak-file glob patterns in deterministic sorted order.

    Accepts existing literal paths containing glob metacharacters
    (e.g. ``sample_[1].mgf``) when glob expansion finds no matches.
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


def _is_valid_spectrum(
    mz: np.ndarray, min_peaks: int, min_mz_range: float
) -> bool:
    return (
        len(mz) >= min_peaks
        and len(mz) > 0
        and float(mz[-1]) - float(mz[0]) >= min_mz_range
    )


def _restrict_mz_range(
    spectrum: sus.MsmsSpectrum, min_mz: float, max_mz: float
) -> None:
    mz_f64 = spectrum.mz.astype(np.float64, copy=False)
    lo = int(np.searchsorted(mz_f64, min_mz, "left"))
    hi = int(np.searchsorted(mz_f64, max_mz, "right"))
    if lo != 0 or hi != len(mz_f64):
        spectrum.mz = spectrum.mz[lo:hi]
        spectrum.intensity = spectrum.intensity[lo:hi]


def _remove_precursor_peak(
    spectrum: sus.MsmsSpectrum, remove_precursor_tol: float
) -> None:
    charge = spectrum.precursor_charge
    precursor_mz_f64 = float(spectrum.precursor_mz)
    mz_f64 = spectrum.mz.astype(np.float64, copy=False)
    abs_charge = 0 if charge is None else abs(int(charge))

    if abs_charge <= 1:
        targets: tuple[float, ...] = (precursor_mz_f64,)
    else:
        sign = 1.0 if charge is not None and charge > 0 else -1.0
        neutral_mass = (precursor_mz_f64 - sign * _PROTON_MASS) * abs_charge
        targets = (
            precursor_mz_f64,
            *(
                neutral_mass / sub_charge + sign * _PROTON_MASS
                for sub_charge in range(abs_charge - 1, 0, -1)
            ),
        )

    keep = np.ones(len(spectrum.mz), dtype=bool)
    for target in targets:
        keep &= np.abs(mz_f64 - target) > remove_precursor_tol
    if not keep.all():
        spectrum.mz = spectrum.mz[keep]
        spectrum.intensity = spectrum.intensity[keep]


def _has_valid_header(
    precursor_mz: float, rt: float | None, charge: Any
) -> bool:
    if not (math.isfinite(precursor_mz) and precursor_mz > 0.0):
        return False
    if rt is not None and math.isinf(rt):
        return False
    return charge is None or (
        isinstance(charge, (int, np.integer))
        and INT8_MIN <= charge <= INT8_MAX
    )


def process_spectrum(
    spectrum: sus.MsmsSpectrum,
    prep: PreprocessingConfig,
    *,
    min_mz: float | None = None,
    max_mz: float | None = None,
) -> ProcessedSpectrum | None:
    """
    Filter, scale, and L2-normalize a single MS/MS spectrum.

    Mutates ``spectrum`` in place via ``spectrum_utils`` and returns
    ``None`` if quality or numeric validity gates fail.
    """
    precursor_mz = float(spectrum.precursor_mz)
    rt = spectrum.retention_time
    charge = spectrum.precursor_charge
    if (
        not _has_valid_header(precursor_mz, rt, charge)
        or not _is_valid_spectrum(
            spectrum.mz, prep.min_peaks, prep.min_mz_range
        )
        or not np.isfinite(spectrum.mz).all()
    ):
        return None

    _restrict_mz_range(
        spectrum,
        prep.min_mz if min_mz is None else min_mz,
        prep.max_mz if max_mz is None else max_mz,
    )
    if not _is_valid_spectrum(spectrum.mz, prep.min_peaks, prep.min_mz_range):
        return None

    if prep.remove_precursor_tol is not None:
        _remove_precursor_peak(spectrum, prep.remove_precursor_tol)
        if not _is_valid_spectrum(
            spectrum.mz, prep.min_peaks, prep.min_mz_range
        ):
            return None

    min_i = float(spectrum.intensity.min())
    max_i = float(spectrum.intensity.max())
    if not (min_i >= 0.0 and max_i > 0.0 and math.isfinite(max_i)):
        return None

    spectrum = spectrum.filter_intensity(
        prep.min_intensity, prep.max_peaks_used
    )
    if not _is_valid_spectrum(spectrum.mz, prep.min_peaks, prep.min_mz_range):
        return None

    if prep.scaling != "off":
        # spectrum_utils 0.3.5 rejects max_rank=None for rank scaling.
        if prep.max_peaks_used is not None:
            spectrum = spectrum.scale_intensity(
                prep.scaling, max_rank=prep.max_peaks_used
            )
        else:
            spectrum = spectrum.scale_intensity(prep.scaling)

    intensity_f64 = spectrum.intensity.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(intensity_f64))
    intensity = (intensity_f64 / norm).astype(np.float32)

    return {
        "identifier": spectrum.identifier,
        "precursor_mz": precursor_mz,
        "precursor_charge": None if charge is None else int(charge),
        "mz": spectrum.mz,
        "intensity": intensity,
        "retention_time": None if rt is None else float(rt),
    }


# ----------------------------------------------------------------------
# Lance dataset naming, canonical metadata, and batch ingestion
# ----------------------------------------------------------------------


def _is_bucket_dataset_name(name: str) -> bool:
    return (
        len(name) > len(BUCKET_PREFIX) + len(BUCKET_SUFFIX)
        and name.startswith(BUCKET_PREFIX)
        and name.endswith(BUCKET_SUFFIX)
    )


def _join_bucket(bucket: ChargeBucket, sep: str) -> str:
    return (
        OTHER_BUCKET
        if bucket == OTHER_BUCKET
        else sep.join(str(part) for part in bucket)
    )


def bucket_dataset_name(bucket: ChargeBucket) -> str:
    """Return the on-disk Lance directory name for ``bucket``."""
    return f"{BUCKET_PREFIX}{_join_bucket(bucket, '_')}{BUCKET_SUFFIX}"


def bucket_dataset_path(work_dir: Path, bucket: ChargeBucket) -> Path:
    """Return the full staged Lance dataset path for ``bucket``."""
    return work_dir / SPECTRA_SUBDIR / bucket_dataset_name(bucket)


def format_bucket_label(bucket: ChargeBucket) -> str:
    """Format a charge bucket key as a readable label for logs."""
    return _join_bucket(bucket, ", ")


def read_canonical_metadata(
    dataset: lance.LanceDataset,
    columns: Sequence[str],
    *,
    with_row_index: bool = False,
) -> pd.DataFrame:
    """Read ``columns`` from ``dataset`` sorted in canonical order."""
    df = dataset.to_table(columns=list(columns)).to_pandas()
    if with_row_index:
        df = df.reset_index()
    return df.sort_values(list(CANONICAL_SORT_COLUMNS), kind="stable")


def _spectra_to_arrow_table(spectra: Sequence[ProcessedSpectrum]) -> pa.Table:
    n = len(spectra)
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    mz_arrays: list[np.ndarray] = []
    int_arrays: list[np.ndarray] = []
    for i, spec in enumerate(spectra):
        mz = np.asarray(spec["mz"], dtype=np.float32)
        intensity = np.asarray(spec["intensity"], dtype=np.float32)
        offsets[i + 1] = offsets[i] + len(mz)
        mz_arrays.append(mz)
        int_arrays.append(intensity)

    flat_mz = (
        np.concatenate(mz_arrays) if mz_arrays else np.empty(0, np.float32)
    )
    flat_int = (
        np.concatenate(int_arrays) if int_arrays else np.empty(0, np.float32)
    )
    return pa.table(
        {
            "identifier": pa.array(
                [s["identifier"] for s in spectra], type=pa.string()
            ),
            "precursor_mz": pa.array(
                [s["precursor_mz"] for s in spectra], type=pa.float32()
            ),
            "precursor_charge": pa.array(
                [s["precursor_charge"] for s in spectra], type=pa.int8()
            ),
            "mz": pa.ListArray.from_arrays(
                offsets, pa.array(flat_mz, type=pa.float32())
            ),
            "intensity": pa.ListArray.from_arrays(
                offsets, pa.array(flat_int, type=pa.float32())
            ),
            "retention_time": pa.array(
                [s["retention_time"] for s in spectra], type=pa.float32()
            ),
        },
        schema=SPECTRUM_SCHEMA,
    )


class _LanceBatchWriter:
    """Buffer preprocessed spectra per charge bucket and flush to Lance."""

    def __init__(
        self,
        charge_buckets: tuple[ChargeBucket, ...] | None,
        work_dir: Path,
    ) -> None:
        self._work_dir = work_dir
        self._explicit = charge_buckets is not None
        self._charge_map: dict[ChargeValue, tuple[ChargeValue, ...]] = {}
        self._catch_other = False
        if charge_buckets is not None:
            for bucket in charge_buckets:
                if bucket == OTHER_BUCKET:
                    self._catch_other = True
                else:
                    for charge in bucket:
                        self._charge_map[charge] = bucket
        self._pending: dict[ChargeBucket, list[ProcessedSpectrum]] = (
            collections.defaultdict(list)
        )
        self._initialized: set[ChargeBucket] = set()
        self.written_counts: dict[ChargeBucket, int] = collections.defaultdict(
            int
        )
        self.excluded_count = 0

    def _route(self, raw_charge: int | None) -> ChargeBucket | None:
        charge: ChargeValue = (
            UNKNOWN_CHARGE if raw_charge is None else raw_charge
        )
        if not self._explicit:
            return (charge,)
        bucket = self._charge_map.get(charge)
        if bucket is not None:
            return bucket
        return OTHER_BUCKET if self._catch_other else None

    def _flush_bucket(
        self, bucket: ChargeBucket, batch: list[ProcessedSpectrum]
    ) -> None:
        table = _spectra_to_arrow_table(batch)
        path = bucket_dataset_path(self._work_dir, bucket)
        if bucket in self._initialized:
            lance.write_dataset(table, str(path), mode="append")
        else:
            logger.debug("Writing lance dataset at %s (mode=overwrite)", path)
            lance.write_dataset(
                table,
                str(path),
                mode="overwrite",
                data_storage_version="stable",
            )
            self._initialized.add(bucket)
        self.written_counts[bucket] += len(table)

    def add(self, spec: ProcessedSpectrum) -> None:
        bucket = self._route(spec["precursor_charge"])
        if bucket is None:
            self.excluded_count += 1
            return
        buf = self._pending[bucket]
        buf.append(spec)
        if len(buf) >= _LANCE_WRITE_BATCH_SIZE:
            self._pending[bucket] = []
            self._flush_bucket(bucket, buf)

    def flush(self) -> None:
        for bucket, buf in self._pending.items():
            if buf:
                self._flush_bucket(bucket, buf)
        self._pending.clear()


def _read_and_process_file(
    path: Path, prep: PreprocessingConfig, min_mz: float, max_mz: float
) -> tuple[list[ProcessedSpectrum], int]:
    spectra: list[ProcessedSpectrum] = []
    low_quality = 0
    for spec in ms_io.get_spectra(path):
        processed = process_spectrum(spec, prep, min_mz=min_mz, max_mz=max_mz)
        if processed is None:
            low_quality += 1
        else:
            spectra.append(processed)
    return spectra, low_quality


class _WorkDirLockEntry:
    """Reference-counted per-directory reentrant staging lock state."""

    __slots__ = ("depth", "refcount", "rlock")

    def __init__(self) -> None:
        self.rlock = threading.RLock()
        self.refcount = 0
        self.depth = 0


_WORK_DIR_LOCKS_GUARD = threading.Lock()
_WORK_DIR_LOCKS: dict[tuple[int, int], _WorkDirLockEntry] = {}


@contextlib.contextmanager
def _staging_lock(work_dir: Path) -> Generator[None, None, None]:
    """
    Serialize staging operations that share ``work_dir``.

    Combines a per-directory ``RLock`` (keyed by ``(st_dev, st_ino)``)
    with ``fcntl.flock`` on ``work_dir / STAGING_LOCK_FILENAME`` so
    both threads and processes serialize without breaking reentrant
    calls on the same thread. The lock file lives outside ``spectra/``
    so ``--overwrite`` cannot remove it.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    st = os.stat(work_dir)
    key = (st.st_dev, st.st_ino)
    with _WORK_DIR_LOCKS_GUARD:
        entry = _WORK_DIR_LOCKS.get(key)
        if entry is None:
            entry = _WorkDirLockEntry()
            _WORK_DIR_LOCKS[key] = entry
        entry.refcount += 1

    try:
        entry.rlock.acquire()
        try:
            lock_file: IO[bytes] | None = None
            if entry.depth == 0:
                lock_path = work_dir / STAGING_LOCK_FILENAME
                lock_file = lock_path.open("a+b")
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                except BaseException:
                    lock_file.close()
                    raise
            entry.depth += 1
            try:
                yield
            finally:
                entry.depth -= 1
                if lock_file is not None:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    finally:
                        lock_file.close()
        finally:
            entry.rlock.release()
    finally:
        with _WORK_DIR_LOCKS_GUARD:
            entry.refcount -= 1
            if entry.refcount == 0:
                _WORK_DIR_LOCKS.pop(key, None)


def stage_spectra(
    input_filenames: Sequence[PathLike],
    work_dir: PathLike,
    config: FalconConfig,
) -> list[ChargeBucket]:
    """
    Read, preprocess, and stage spectra into per-charge Lance datasets.

    Streams directly on the main thread for single-file or single-worker
    runs, and parses multiple files in parallel via ``joblib`` workers.
    """
    # Absolutize without dereferencing symlinks so warm loky workers
    # stay unaffected by os.chdir() while preserving symlink stems.
    resolved_files = [Path(os.path.abspath(fn)) for fn in input_filenames]
    work_path = Path(work_dir)
    prep = config.preprocessing
    _, min_mz, max_mz = mz_bin_bounds(
        prep.min_mz, prep.max_mz, config.clustering.fragment_tol
    )
    charge_buckets = config.clustering.precursor_charge_buckets
    writer = _LanceBatchWriter(charge_buckets, work_path)
    n_workers = max(1, int(joblib.effective_n_jobs(-1)))
    max_workers = min(len(resolved_files), n_workers)
    low_quality = 0

    with _staging_lock(work_path):
        (work_path / SPECTRA_SUBDIR).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Reading spectra from %d peak file(s)...", len(resolved_files)
        )
        if max_workers == 1:
            for path in resolved_files:
                for spec in ms_io.get_spectra(path):
                    processed = process_spectrum(
                        spec, prep, min_mz=min_mz, max_mz=max_mz
                    )
                    if processed is None:
                        low_quality += 1
                    else:
                        writer.add(processed)
        else:
            parallel_iter = joblib.Parallel(
                n_jobs=max_workers,
                backend="loky",
                return_as="generator_unordered",
            )(
                joblib.delayed(_read_and_process_file)(
                    path, prep, min_mz, max_mz
                )
                for path in resolved_files
            )
            with contextlib.closing(parallel_iter) as file_results:
                for file_spectra, lqc in file_results:
                    low_quality += lqc
                    for spec in file_spectra:
                        writer.add(spec)
        writer.flush()

    counts = writer.written_counts
    if charge_buckets is None:
        valid_buckets = sorted(
            (b for b, count in counts.items() if count > 0),
            key=lambda b: charge_sort_key(
                cast("tuple[ChargeValue, ...]", b)[0]
            ),
        )
    else:
        valid_buckets = []
        for bucket in charge_buckets:
            if counts.get(bucket, 0) > 0:
                valid_buckets.append(bucket)
            else:
                logger.debug(
                    "No spectra for bucket %s, skipping",
                    format_bucket_label(bucket),
                )

    logger.info(
        "Staged %d spectra from %d peak file(s)",
        sum(counts.values()),
        len(resolved_files),
    )
    logger.info("Skipped %d low-quality spectra", low_quality)
    if writer.excluded_count > 0:
        logger.info(
            "Excluded %d spectra outside configured charge buckets",
            writer.excluded_count,
        )
    return valid_buckets


# ----------------------------------------------------------------------
# Staging cache manifest and validation
# ----------------------------------------------------------------------


def _serialize_buckets(
    buckets: Sequence[ChargeBucket] | None,
) -> list[list[ChargeValue] | str] | None:
    if buckets is None:
        return None
    return [OTHER_BUCKET if b == OTHER_BUCKET else list(b) for b in buckets]


def _build_manifest(
    config: FalconConfig,
    resolved_inputs: Sequence[Path],
    buckets: Sequence[ChargeBucket] | None = None,
) -> dict[str, Any]:
    input_entries = []
    for input_path in resolved_inputs:
        st = os.stat(input_path)
        input_entries.append(
            {
                "path": str(Path(os.path.abspath(input_path))),
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
        "precursor_charge_buckets": _serialize_buckets(
            config.clustering.precursor_charge_buckets
        ),
        "buckets": _serialize_buckets(buckets),
    }


def _validate_cached_manifest(
    cache_path: Path,
    expected: Mapping[str, Any],
    target_dir: Path,
    work_dir: Path,
) -> list[ChargeBucket]:
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Corrupted staging cache manifest at {cache_path}; {_CACHE_HINT}"
        ) from exc

    if not isinstance(cached, Mapping):
        raise ValueError(
            f"Cached staging data in {work_dir} does not match the current "
            f"configuration (mismatched: schema_version); {_CACHE_HINT}"
        )

    mismatches = [
        key
        for key in (
            "schema_version",
            "inputs",
            "preprocessing",
            "fragment_tol",
            "precursor_charge_buckets",
        )
        if key not in cached or cached[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            f"Cached staging data in {work_dir} does not match the current "
            f"configuration (mismatched: {', '.join(mismatches)}); "
            f"{_CACHE_HINT}"
        )

    raw_buckets = cached.get("buckets")
    if not isinstance(raw_buckets, list):
        raise ValueError(
            f"Invalid staging cache buckets in {cache_path}; {_CACHE_HINT}"
        )
    cached_buckets = (
        list(parse_charge_buckets(raw_buckets)) if raw_buckets else []
    )
    expected_dirs = {bucket_dataset_name(b) for b in cached_buckets}
    disk_dirs = {
        e.name
        for e in target_dir.iterdir()
        if e.is_dir() and _is_bucket_dataset_name(e.name)
    }
    if disk_dirs != expected_dirs:
        raise ValueError(
            f"Invalid staging cache buckets in {cache_path}: staged datasets "
            f"on disk do not match cached buckets; {_CACHE_HINT}"
        )
    return cached_buckets


def load_or_stage_buckets(
    config: FalconConfig, work_dir: Path
) -> list[ChargeBucket]:
    """Return staged charge buckets, reusing a validated cache when safe."""
    resolved_inputs = resolve_input_files(config.io.input_filenames)
    expected_manifest = _build_manifest(config, resolved_inputs)
    target_dir = work_dir / SPECTRA_SUBDIR
    cache_path = target_dir / STAGING_CACHE_FILENAME

    with _staging_lock(work_dir):
        if config.io.overwrite:
            shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            if cache_path.is_file():
                cached_buckets = _validate_cached_manifest(
                    cache_path, expected_manifest, target_dir, work_dir
                )
                logger.info(
                    "Reusing validated staging cache in %s (%d bucket(s))",
                    work_dir,
                    len(cached_buckets),
                )
                return cached_buckets
            if any(
                _is_bucket_dataset_name(e.name) for e in target_dir.iterdir()
            ):
                raise ValueError(
                    f"Working directory {work_dir} contains unverified "
                    f"staged datasets without {STAGING_CACHE_FILENAME}; "
                    f"{_CACHE_HINT}"
                )

        charge_buckets = stage_spectra(resolved_inputs, work_dir, config)
        manifest = {
            **expected_manifest,
            "buckets": _serialize_buckets(charge_buckets),
        }
        with ms_io.atomic_write_text(
            cache_path, prefix=".staging_cache_"
        ) as out:
            json.dump(manifest, out, indent=2, allow_nan=False)
            out.write("\n")
        return charge_buckets
