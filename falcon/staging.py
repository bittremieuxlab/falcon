"""Lance schema, charge routing, ingestion, and cache validation."""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import joblib
import lance
import numpy as np
import pyarrow as pa

from . import __version__, ms_io
from .types import (
    INT8_MAX,
    INT8_MIN,
    OTHER_BUCKET,
    UNKNOWN_CHARGE,
    charge_sort_key,
    parse_charge_bucket_list,
)
from .utils import (
    atomic_write_text,
    effective_n_workers,
    loky_unordered_pool,
    resolve_input_files,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import pandas as pd
    import spectrum_utils.spectrum as sus

    from .config import FalconConfig
    from .types import (
        ChargeBucket,
        ChargeValue,
        PathLike,
        ProcessedSpectrum,
    )

logger = logging.getLogger(__name__)

SPECTRA_SUBDIR = "spectra"
BUCKET_PREFIX = "spectra_charge_"
BUCKET_SUFFIX = ".lance"
STAGING_CACHE_SCHEMA_VERSION = 1
STAGING_CACHE_FILENAME = "staging_cache.json"

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

_LANCE_WRITE_BATCH_SIZE = 10_000
_CACHE_HINT = "rerun with --overwrite or a different --work-dir."

__all__ = [
    "BUCKET_PREFIX",
    "BUCKET_SUFFIX",
    "CANONICAL_SORT_COLUMNS",
    "METADATA_COLUMNS",
    "SORTED_INTERVAL_COLUMNS",
    "SPECTRA_SUBDIR",
    "SPECTRUM_SCHEMA",
    "STAGING_CACHE_FILENAME",
    "STAGING_CACHE_SCHEMA_VERSION",
    "ChargeRouter",
    "bucket_dataset_name",
    "bucket_dataset_path",
    "bucket_label_from_uri",
    "build_staging_manifest",
    "deserialize_cached_buckets",
    "diff_staging_manifests",
    "format_bucket_label",
    "is_bucket_dataset_name",
    "load_or_stage_buckets",
    "read_canonical_metadata",
    "serialize_cached_buckets",
    "spectra_dir",
    "stage_spectra",
]


# ----------------------------------------------------------------------
# Lance dataset naming and canonical metadata helpers
# ----------------------------------------------------------------------


def spectra_dir(work_dir: Path) -> Path:
    """Return the ``spectra`` staging directory under ``work_dir``."""
    return work_dir / SPECTRA_SUBDIR


def is_bucket_dataset_name(name: str) -> bool:
    """Return whether ``name`` matches a staged charge dataset."""
    return (
        len(name) > len(BUCKET_PREFIX) + len(BUCKET_SUFFIX)
        and name.startswith(BUCKET_PREFIX)
        and name.endswith(BUCKET_SUFFIX)
    )


def _join_bucket(bucket: ChargeBucket, sep: str) -> str:
    if bucket == OTHER_BUCKET:
        return OTHER_BUCKET
    return sep.join(str(part) for part in bucket)


def bucket_dataset_name(bucket: ChargeBucket) -> str:
    """Return the on-disk Lance directory name for ``bucket``."""
    return f"{BUCKET_PREFIX}{_join_bucket(bucket, '_')}{BUCKET_SUFFIX}"


def bucket_dataset_path(work_dir: Path, bucket: ChargeBucket) -> Path:
    """Return the full staged Lance dataset path for ``bucket``."""
    return spectra_dir(work_dir) / bucket_dataset_name(bucket)


def format_bucket_label(bucket: ChargeBucket) -> str:
    """Format a charge bucket key as a readable label for logs."""
    return _join_bucket(bucket, ", ")


def bucket_label_from_uri(uri: str, fallback: str) -> str:
    """Derive a human-readable charge label from a Lance dataset URI."""
    name = Path(uri).name
    if is_bucket_dataset_name(name):
        return name[len(BUCKET_PREFIX) : -len(BUCKET_SUFFIX)].replace(
            "_", ", "
        )
    return fallback


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


# ----------------------------------------------------------------------
# Charge routing and Lance batch ingestion
# ----------------------------------------------------------------------


class ChargeRouter:
    """
    Route raw precursor charges to canonical ``ChargeBucket`` keys.

    Supports automatic per-charge bucketing (``charge_buckets=None``)
    and explicit non-overlapping charge groupings with optional
    ``"other"`` fallback.
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
            if bucket == OTHER_BUCKET:
                catch_other = True
            else:
                for charge in bucket:
                    charge_to_bucket[charge] = bucket
        self._canonical_buckets = canonical
        self._charge_to_bucket = charge_to_bucket
        self._catch_other = catch_other

    @property
    def canonical_buckets(self) -> tuple[ChargeBucket, ...] | None:
        """Return explicit buckets, or ``None`` in auto mode."""
        return self._canonical_buckets

    def route(
        self, raw_charge: int | np.integer[Any] | None
    ) -> ChargeBucket | None:
        """Map a precursor charge to its bucket, or ``None``."""
        if raw_charge is None:
            charge: ChargeValue = UNKNOWN_CHARGE
        else:
            charge_int = int(raw_charge)
            if not (INT8_MIN <= charge_int <= INT8_MAX):
                raise ValueError(
                    f"Precursor charge {charge_int} is outside int8 range "
                    f"[{INT8_MIN}, {INT8_MAX}]"
                )
            charge = charge_int
        if self._canonical_buckets is None:
            return (charge,)
        bucket = self._charge_to_bucket.get(charge)
        if bucket is not None:
            return bucket
        return OTHER_BUCKET if self._catch_other else None


def _spectra_to_arrow_table(
    spectra: Sequence[Mapping[str, Any]],
) -> pa.Table:
    """Build a columnar PyArrow table matching ``SPECTRUM_SCHEMA``."""
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
    """Buffer preprocessed spectra per bucket and flush to Lance."""

    def __init__(self, router: ChargeRouter, work_dir: Path) -> None:
        self._router = router
        self._work_dir = work_dir
        self._pending: dict[ChargeBucket, list[ProcessedSpectrum]] = (
            collections.defaultdict(list)
        )
        self._initialized: set[ChargeBucket] = set()
        self.written_counts: dict[ChargeBucket, int] = collections.defaultdict(
            int
        )
        self.excluded_count = 0

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
        bucket = self._router.route(spec["precursor_charge"])
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


def _iter_processed_spectra(
    filename: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> Iterator[ProcessedSpectrum | None]:
    abs_path = (
        filename if filename.is_absolute() else Path(os.path.abspath(filename))
    )
    for spec in ms_io.get_spectra(abs_path):
        yield process_spectrum(spec)


def _read_spectra(
    filename: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> tuple[list[ProcessedSpectrum], int]:
    low_quality = 0
    spectra: list[ProcessedSpectrum] = []
    for processed in _iter_processed_spectra(filename, process_spectrum):
        if processed is None:
            low_quality += 1
        else:
            spectra.append(processed)
    return spectra, low_quality


def stage_spectra(
    input_filenames: Sequence[PathLike],
    work_dir: PathLike,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
    charge_buckets: Sequence[Any] | None,
) -> list[ChargeBucket]:
    """
    Read, preprocess, and stage spectra into per-charge Lance datasets.

    Streams directly on the main thread for single-file or single-worker
    runs, and parses multiple files in parallel via ``joblib`` workers.
    """
    # Absolutize without dereferencing symlinks so warm loky workers
    # remain unaffected by os.chdir() while preserving symlink stems.
    resolved_files = [Path(os.path.abspath(fn)) for fn in input_filenames]
    if not resolved_files:
        raise FileNotFoundError("No input peak files supplied")

    router = ChargeRouter(charge_buckets)
    work_path = Path(work_dir)
    spectra_dir(work_path).mkdir(parents=True, exist_ok=True)
    logger.info("Reading spectra from %d peak file(s)...", len(resolved_files))

    writer = _LanceBatchWriter(router, work_path)
    max_workers = min(len(resolved_files), effective_n_workers())
    low_quality = 0

    if max_workers == 1:
        for file in resolved_files:
            for processed in _iter_processed_spectra(file, process_spectrum):
                if processed is None:
                    low_quality += 1
                else:
                    writer.add(processed)
    else:
        parallel_iter = loky_unordered_pool(max_workers)(
            joblib.delayed(_read_spectra)(file, process_spectrum)
            for file in resolved_files
        )
        with contextlib.closing(parallel_iter) as file_results:
            for file_spectra, lqc in file_results:
                low_quality += lqc
                for spec in file_spectra:
                    writer.add(spec)
    writer.flush()

    counts = writer.written_counts
    if router.canonical_buckets is None:
        valid_buckets = sorted(
            (b for b, count in counts.items() if count > 0),
            key=lambda b: charge_sort_key(
                cast("tuple[ChargeValue, ...]", b)[0]
            ),
        )
    else:
        valid_buckets = []
        for bucket in router.canonical_buckets:
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


def serialize_cached_buckets(
    buckets: Sequence[ChargeBucket] | None,
) -> list[list[ChargeValue] | str] | None:
    """Serialize charge bucket keys for ``staging_cache.json``."""
    if buckets is None:
        return None
    return [
        OTHER_BUCKET
        if bucket == OTHER_BUCKET
        else sorted(bucket, key=charge_sort_key)
        for bucket in buckets
    ]


def deserialize_cached_buckets(raw_buckets: Any) -> list[ChargeBucket]:
    """Validate and parse bucket keys from ``staging_cache.json``."""
    if not isinstance(raw_buckets, list):
        raise ValueError("Cached staging manifest has invalid 'buckets' field")
    return (
        [] if not raw_buckets else list(parse_charge_bucket_list(raw_buckets))
    )


def build_staging_manifest(
    config: FalconConfig,
    resolved_inputs: Sequence[Path],
    buckets: Sequence[ChargeBucket] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable staging cache provenance manifest."""
    input_entries: list[dict[str, Any]] = []
    for input_path in resolved_inputs:
        logical = Path(os.path.abspath(input_path))
        resolved = input_path.resolve(strict=True)
        st = os.stat(resolved)
        entry: dict[str, Any] = {
            "path": str(logical),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
        if resolved != logical:
            entry["target"] = str(resolved)
        input_entries.append(entry)

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
    """Return staging manifest keys that differ from ``expected``."""
    if not isinstance(cached, Mapping) or (
        cached.get("schema_version") != expected.get("schema_version")
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
        if cached.get(key) != expected[key] or key not in cached
    ]


def _clear_directory(target_dir: Path) -> None:
    for entry in target_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def load_or_stage_buckets(
    config: FalconConfig,
    work_dir: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> list[ChargeBucket]:
    """
    Return staged charge buckets, reusing validated cache when safe.

    Under ``config.io.overwrite``, clears the staging directory and
    re-stages all spectra. Otherwise validates an existing
    ``staging_cache.json`` manifest and on-disk Lance directories before
    reusing cached buckets.
    """
    resolved_inputs = resolve_input_files(config.io.input_filenames)
    expected_manifest = build_staging_manifest(config, resolved_inputs)
    target_dir = spectra_dir(work_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / STAGING_CACHE_FILENAME

    if config.io.overwrite:
        _clear_directory(target_dir)
    elif cache_path.exists() or cache_path.is_symlink():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Corrupted staging cache manifest at {cache_path}; "
                f"{_CACHE_HINT}"
            ) from exc

        mismatches = diff_staging_manifests(cached, expected_manifest)
        if mismatches:
            diff_str = ", ".join(mismatches)
            raise ValueError(
                f"Cached staging data in {work_dir} does not match the "
                f"current configuration (mismatched: {diff_str}); "
                f"{_CACHE_HINT}"
            )

        try:
            cached_buckets = deserialize_cached_buckets(cached.get("buckets"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid staging cache buckets in {cache_path}: {exc}; "
                f"{_CACHE_HINT}"
            ) from exc

        expected_datasets = {bucket_dataset_name(b) for b in cached_buckets}
        disk_entries = [
            e for e in target_dir.iterdir() if is_bucket_dataset_name(e.name)
        ]
        if (
            any(not e.is_dir() for e in disk_entries)
            or {e.name for e in disk_entries} != expected_datasets
        ):
            raise ValueError(
                f"Invalid staging cache buckets in {cache_path}: staged "
                f"datasets on disk do not match cached buckets; {_CACHE_HINT}"
            )
        logger.info(
            "Reusing validated staging cache in %s (%d bucket(s))",
            work_dir,
            len(cached_buckets),
        )
        return cached_buckets
    elif any(
        e.name == "charges.joblib" or is_bucket_dataset_name(e.name)
        for e in target_dir.iterdir()
    ):
        raise ValueError(
            f"Working directory {work_dir} contains an unverified or legacy "
            f"staging cache without staging_cache.json; {_CACHE_HINT}"
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
    with atomic_write_text(cache_path, prefix=".staging_cache_") as f_out:
        json.dump(manifest, f_out, indent=2, allow_nan=False)
        f_out.write("\n")
    return charge_buckets
