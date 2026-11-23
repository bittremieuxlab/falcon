"""Shared domain data types, charge-bucket parsing, and m/z bin math.

This leaf module depends only on stdlib and NumPy so lightweight entry
points (such as ``falcon.config`` and ``falcon.cli``) and peak readers
(``falcon.ms_io``) never pull in heavy scientific dependencies like
Numba, SciPy, Lance, or ``spectrum_utils``.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple, TypedDict

import numpy as np

INT8_MIN = -128
INT8_MAX = 127
INT32_MAX = 2**31 - 1

ChargeValue = int | Literal["unknown"]
ChargeBucket = tuple[ChargeValue, ...] | Literal["other"]


class ProcessedSpectrum(TypedDict):
    """Preprocessed spectrum record staged into a Lance dataset."""

    identifier: str
    precursor_mz: float
    precursor_charge: int | None
    mz: np.ndarray
    intensity: np.ndarray
    retention_time: float | None


class SpectrumTuple(NamedTuple):
    """Compact spectrum record for similarity and clustering code."""

    precursor_mz: float
    precursor_charge: Any
    mz: np.ndarray
    intensity: np.ndarray


class ConsensusTuple(NamedTuple):
    """Representative spectrum produced for one cluster."""

    precursor_mz: np.float32
    precursor_charge: np.int32 | float
    mz: np.ndarray
    intensity: np.ndarray
    retention_time: np.float32
    cluster_id: np.int32


def make_consensus_tuple(
    precursor_mz: float,
    precursor_charge: Any,
    mz: np.ndarray,
    intensity: np.ndarray,
    retention_time: float,
    cluster_id: int,
) -> ConsensusTuple:
    """
    Construct a ``ConsensusTuple`` with canonical float32/int32 dtypes.

    Missing or NaN precursor charges are normalized to ``np.nan``.

    Parameters
    ----------
    precursor_mz : float
        Representative precursor m/z value.
    precursor_charge : Any
        Precursor charge state, or ``None``/``NaN`` when unknown.
    mz : np.ndarray
        Fragment peak m/z array.
    intensity : np.ndarray
        Fragment peak intensity array.
    retention_time : float
        Representative retention time (or ``NaN`` when missing).
    cluster_id : int
        Cluster label assigned to this representative.

    Returns
    -------
    ConsensusTuple
        Typed representative spectrum record.
    """
    if precursor_charge is None or math.isnan(float(precursor_charge)):
        norm_charge: np.int32 | float = np.nan
    else:
        norm_charge = np.int32(precursor_charge)
    return ConsensusTuple(
        precursor_mz=np.float32(precursor_mz),
        precursor_charge=norm_charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        retention_time=np.float32(retention_time),
        cluster_id=np.int32(cluster_id),
    )


def charge_sort_key(value: ChargeValue) -> tuple[bool, int | str]:
    """
    Sort numeric precursor charges ascending before ``"unknown"``.

    Parameters
    ----------
    value : ChargeValue
        Integer charge state or ``"unknown"``.

    Returns
    -------
    tuple[bool, int | str]
        Deterministic sort key ordering integers before strings.
    """
    return (isinstance(value, str), value)


def _parse_single_charge_bucket(
    raw_bucket: Any,
    seen_values: set[ChargeValue],
) -> tuple[ChargeValue, ...]:
    """Validate and canonicalize one non-``"other"`` charge bucket."""
    if not isinstance(raw_bucket, (list, tuple, set, frozenset)):
        raise ValueError(
            "Each charge bucket must be a sequence of charges or 'other'"
        )
    if len(raw_bucket) == 0:
        raise ValueError("Charge bucket must not be empty")

    bucket_seen: set[ChargeValue] = set()
    for elem in raw_bucket:
        if isinstance(elem, bool):
            raise ValueError(
                "Charge bucket entries must be integers or 'unknown', "
                "not booleans"
            )
        if elem == "unknown":
            parsed: ChargeValue = "unknown"
        elif isinstance(elem, int):
            if not (INT8_MIN <= elem <= INT8_MAX):
                raise ValueError(
                    f"Precursor charge {elem} is outside int8 range "
                    f"[{INT8_MIN}, {INT8_MAX}]"
                )
            parsed = elem
        else:
            raise ValueError(
                f"Invalid charge bucket entry {elem!r}: expected int8 "
                "charge or 'unknown'"
            )

        if parsed in bucket_seen:
            raise ValueError(
                f"Duplicate charge value {parsed!r} within the same bucket"
            )
        if parsed in seen_values:
            raise ValueError(
                f"Charge value {parsed!r} appears in more than one bucket"
            )
        bucket_seen.add(parsed)

    seen_values |= bucket_seen
    return tuple(sorted(bucket_seen, key=charge_sort_key))


def parse_charge_buckets(
    value: Any,
) -> tuple[ChargeBucket, ...] | None:
    """
    Validate and canonicalize ``precursor_charge_buckets``.

    Accepts ``None`` or ``"auto"`` (returning ``None``), or a non-empty
    sequence of non-overlapping int8/``"unknown"`` buckets and at most
    one ``"other"`` sentinel.

    Parameters
    ----------
    value : Any
        Raw charge bucket specification from Python, TOML, CLI, or a
        cached staging manifest.

    Returns
    -------
    tuple[ChargeBucket, ...] | None
        Canonical immutable charge bucket tuple, or ``None`` for auto
        mode.

    Raises
    ------
    ValueError
        If the bucket specification is malformed, empty, out of int8
        bounds, or contains duplicate or overlapping charges.
    """
    if value is None or value == "auto":
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "precursor_charge_buckets must be 'auto' or a non-empty list "
            "of buckets"
        )
    if len(value) == 0:
        raise ValueError("precursor_charge_buckets must not be empty")

    buckets: list[ChargeBucket] = []
    seen_values: set[ChargeValue] = set()
    seen_other = False
    for raw_bucket in value:
        if raw_bucket == "other":
            if seen_other:
                raise ValueError(
                    "Charge bucket 'other' appears more than once"
                )
            seen_other = True
            buckets.append("other")
            continue
        buckets.append(_parse_single_charge_bucket(raw_bucket, seen_values))

    return tuple(buckets)


def mz_bin_bounds(
    min_mz: float, max_mz: float, bin_size: float
) -> tuple[int, float, float]:
    """
    Compute aligned m/z bin boundaries and exact bin count.

    Guarantees ``start_dim <= min_mz``, ``end_dim > max_mz``, and an
    exact integer bin count ``n_bins >= 1`` in float64 precision.

    Parameters
    ----------
    min_mz : float
        Lower bound of the mass range (inclusive).
    max_mz : float
        Upper bound of the mass range (inclusive).
    bin_size : float
        Positive bin width in Da.

    Returns
    -------
    tuple[int, float, float]
        ``(n_bins, start_dim, end_dim)`` where ``start_dim`` is the
        largest grid multiple at or below ``min_mz`` and ``end_dim`` is
        the smallest grid multiple strictly above ``max_mz``.

    Raises
    ------
    ValueError
        If inputs are non-finite, ``bin_size <= 0``,
        ``min_mz > max_mz``, or the grid overflows float precision.
    """
    if not (
        math.isfinite(min_mz)
        and math.isfinite(max_mz)
        and math.isfinite(bin_size)
    ):
        raise ValueError("min_mz, max_mz, and bin_size must be finite")
    if bin_size <= 0.0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    if min_mz > max_mz:
        raise ValueError(
            f"min_mz ({min_mz}) must not exceed max_mz ({max_mz})"
        )

    start_ratio = min_mz / bin_size
    end_ratio = max_mz / bin_size
    if not (math.isfinite(start_ratio) and math.isfinite(end_ratio)):
        raise ValueError(
            f"bin_size ({bin_size}) causes non-finite m/z bin count over "
            f"[{min_mz}, {max_mz}]"
        )

    start_idx = math.floor(start_ratio)
    if start_idx * bin_size > min_mz:
        start_idx -= 1
    end_idx = math.floor(end_ratio) + 1
    if end_idx * bin_size <= max_mz:
        end_idx += 1

    start_dim = start_idx * bin_size
    end_dim = end_idx * bin_size
    if not (math.isfinite(start_dim) and math.isfinite(end_dim)):
        raise ValueError(
            f"bin_size ({bin_size}) causes non-finite m/z bounds over "
            f"[{min_mz}, {max_mz}]"
        )
    if start_dim > min_mz or end_dim <= max_mz:
        raise ValueError(
            f"bin_size ({bin_size}) is below float64 precision at "
            f"[{min_mz}, {max_mz}]"
        )
    return end_idx - start_idx, start_dim, end_dim
