"""Shared Lance dataset schema, bucket naming, and metadata reading.

This lightweight storage-contract module is imported by staging,
clustering, export, and pipeline orchestration so clustering and export
never pull in peak-file readers or parallel staging machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance
    import pandas as pd

    from .types import ChargeBucket

SPECTRA_SUBDIR = "spectra"
BUCKET_PREFIX = "spectra_charge_"
BUCKET_SUFFIX = ".lance"

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

__all__ = [
    "BUCKET_PREFIX",
    "BUCKET_SUFFIX",
    "CANONICAL_SORT_COLUMNS",
    "INTERVAL_COLUMNS",
    "METADATA_COLUMNS",
    "SORTED_INTERVAL_COLUMNS",
    "SPECTRA_SUBDIR",
    "SPECTRUM_SCHEMA",
    "bucket_dataset_name",
    "bucket_dataset_path",
    "bucket_key_to_str",
    "bucket_label_from_uri",
    "format_bucket_label",
    "is_bucket_dataset_name",
    "read_canonical_metadata",
    "spectra_dir",
]


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


def bucket_key_to_str(bucket: ChargeBucket) -> str:
    """
    Convert a canonical charge bucket key to a filename component.

    Parameters
    ----------
    bucket : ChargeBucket
        Canonical charge bucket tuple (e.g. ``(1, 2)``,
        ``("unknown",)``) or ``"other"``.

    Returns
    -------
    str
        Filename component without leading underscore (e.g. ``"1_2"``,
        ``"3_unknown"``, ``"other"``).
    """
    if bucket == "other":
        return "other"
    if isinstance(bucket, tuple):
        return "_".join(str(part) for part in bucket)
    raise TypeError(f"Unsupported bucket key type: {type(bucket).__name__}")


def bucket_dataset_name(bucket: ChargeBucket) -> str:
    """Return the on-disk Lance directory name for ``bucket``."""
    return f"{BUCKET_PREFIX}{bucket_key_to_str(bucket)}{BUCKET_SUFFIX}"


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
    if bucket == "other":
        return "other"
    if isinstance(bucket, tuple):
        return ", ".join(str(part) for part in bucket)
    raise TypeError(f"Unsupported bucket key type: {type(bucket).__name__}")


def bucket_label_from_uri(uri: str, fallback: str) -> str:
    """
    Derive a human-readable charge label from a Lance dataset URI.

    Parameters
    ----------
    uri : str
        Dataset URI or directory path.
    fallback : str
        Label returned when ``uri`` does not match the staged bucket
        naming convention.

    Returns
    -------
    str
        Formatted charge label (e.g. ``"1, 2"`` or ``"unknown"``).
    """
    name = Path(uri.rstrip("/")).name
    if is_bucket_dataset_name(name):
        key = name[len(BUCKET_PREFIX) : -len(BUCKET_SUFFIX)]
        return key.replace("_", ", ")
    return fallback


def read_canonical_metadata(
    dataset: lance.LanceDataset,
    columns: Sequence[str],
    *,
    with_row_index: bool = False,
) -> pd.DataFrame:
    """
    Read ``columns`` from ``dataset`` sorted in canonical order.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge Lance dataset.
    columns : Sequence[str]
        Column names to project (must include
        ``CANONICAL_SORT_COLUMNS``).
    with_row_index : bool, default=False
        When ``True``, preserve the original Lance row position in an
        ``"index"`` column before sorting (for subsequent ``take``).

    Returns
    -------
    pd.DataFrame
        Metadata table sorted by ``CANONICAL_SORT_COLUMNS``.

    Raises
    ------
    ValueError
        If any required column in ``CANONICAL_SORT_COLUMNS`` is omitted
        from ``columns``.
    """
    missing = [col for col in CANONICAL_SORT_COLUMNS if col not in columns]
    if missing:
        raise ValueError(
            "columns must include CANONICAL_SORT_COLUMNS; missing: "
            + ", ".join(missing)
        )
    df = dataset.to_table(columns=list(columns)).to_pandas()
    if with_row_index:
        df = df.reset_index()
    return df.sort_values(list(CANONICAL_SORT_COLUMNS), kind="stable")
