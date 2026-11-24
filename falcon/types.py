"""Shared domain data types, charge-bucket parsing, and m/z bin math.

This leaf module imports only stdlib at module load time so lightweight
entry points (``falcon.config`` and ``falcon.cli``) never pull in NumPy,
Numba, SciPy, Lance, or ``spectrum_utils``. Array helpers import NumPy
lazily when called.
"""

from __future__ import annotations

import functools
import math
import os
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

if TYPE_CHECKING:
    import numpy as np

INT8_MIN = -128
INT8_MAX = 127
INT32_MAX = 2**31 - 1

DEFAULT_MIN_MZ: float = 101.0
DEFAULT_MAX_MZ: float = 1500.0
DEFAULT_OUTLIER_CUTOFF_LOWER: float = 1.5
DEFAULT_OUTLIER_CUTOFF_UPPER: float = 1.5

NONE_SENTINEL = "none"
AUTO_SENTINEL = "auto"
UNKNOWN_CHARGE: Literal["unknown"] = "unknown"
OTHER_BUCKET: Literal["other"] = "other"

PrecursorTolMode = Literal["ppm", "Da"]
ScalingMethod = Literal["off", "root", "log", "rank"]
LinkageMethod = Literal["single", "complete", "average"]
ConsensusMethod = Literal["medoid", "average"]

PathLike = str | os.PathLike[str]
ChargeValue = int | Literal["unknown"]
ChargeBucket = tuple[ChargeValue, ...] | Literal["other"]

__all__ = [
    "AUTO_SENTINEL",
    "DEFAULT_MAX_MZ",
    "DEFAULT_MIN_MZ",
    "DEFAULT_OUTLIER_CUTOFF_LOWER",
    "DEFAULT_OUTLIER_CUTOFF_UPPER",
    "INT8_MAX",
    "INT8_MIN",
    "INT32_MAX",
    "NONE_SENTINEL",
    "OTHER_BUCKET",
    "UNKNOWN_CHARGE",
    "ChargeBucket",
    "ChargeValue",
    "ClusterResult",
    "ConsensusMethod",
    "ConsensusTuple",
    "LinkageMethod",
    "PathLike",
    "PrecursorTolMode",
    "ProcessedSpectrum",
    "ScalingMethod",
    "SpectrumTuple",
    "charge_sort_key",
    "make_consensus_tuple",
    "mz_bin_bounds",
    "offset_cluster_result",
    "parse_charge_bucket_list",
    "parse_optional_charge_buckets",
]


@functools.lru_cache(maxsize=1)
def _numpy() -> Any:
    """Lazily import and cache ``numpy`` on first array-helper call."""
    import numpy as np

    return np


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

    precursor_mz: float | np.floating[Any]
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

    @classmethod
    def from_spectrum(
        cls,
        spec: SpectrumTuple,
        retention_time: float | np.floating[Any],
        cluster_id: int,
    ) -> ConsensusTuple:
        """
        Build a ``ConsensusTuple`` directly from a ``SpectrumTuple``.

        Parameters
        ----------
        spec : SpectrumTuple
            Source spectrum whose precursor and fragment peaks become
            the representative.
        retention_time : float
            Retention time for the representative (or ``NaN``).
        cluster_id : int
            Cluster label assigned to this representative.

        Returns
        -------
        ConsensusTuple
            Representative spectrum record with canonical dtypes.
        """
        return make_consensus_tuple(
            precursor_mz=spec.precursor_mz,
            precursor_charge=spec.precursor_charge,
            mz=spec.mz,
            intensity=spec.intensity,
            retention_time=retention_time,
            cluster_id=cluster_id,
        )


class ClusterResult(NamedTuple):
    """Cluster label assignments and representative spectra."""

    labels: np.ndarray
    representatives: list[ConsensusTuple]


def make_consensus_tuple(
    precursor_mz: float | np.floating[Any],
    precursor_charge: Any,
    mz: np.ndarray,
    intensity: np.ndarray,
    retention_time: float | np.floating[Any],
    cluster_id: int,
) -> ConsensusTuple:
    """
    Construct a ``ConsensusTuple`` with canonical float32/int32 dtypes.

    Missing or non-finite precursor charges normalize to ``np.nan``.

    Parameters
    ----------
    precursor_mz : float | np.floating
        Representative precursor m/z value.
    precursor_charge : Any
        Precursor charge state, or ``None``/``NaN`` when unknown.
    mz : np.ndarray
        Fragment peak m/z array.
    intensity : np.ndarray
        Fragment peak intensity array.
    retention_time : float | np.floating
        Representative retention time (or ``NaN`` when missing).
    cluster_id : int
        Cluster label assigned to this representative.

    Returns
    -------
    ConsensusTuple
        Typed representative spectrum record.
    """
    np = _numpy()

    if precursor_charge is None or not math.isfinite(float(precursor_charge)):
        norm_charge: np.int32 | float = np.nan
    else:
        norm_charge = np.int32(precursor_charge)
    return ConsensusTuple(
        precursor_mz=np.float32(precursor_mz),
        precursor_charge=norm_charge,
        mz=np.array(mz, dtype=np.float32, copy=True),
        intensity=np.array(intensity, dtype=np.float32, copy=True),
        retention_time=np.float32(retention_time),
        cluster_id=np.int32(cluster_id),
    )


def offset_cluster_result(
    result: ClusterResult,
    offset: int,
    *,
    include_representatives: bool = True,
) -> tuple[ClusterResult, int]:
    """
    Shift cluster labels and representative IDs by ``offset``.

    Never mutates ``result.labels`` in place.

    Parameters
    ----------
    result : ClusterResult
        Source cluster labels and representative spectra.
    offset : int
        Non-negative integer offset added to every label and cluster ID.
    include_representatives : bool, default=True
        When ``False``, skip offsetting representative spectra and
        return an empty list.

    Returns
    -------
    tuple[ClusterResult, int]
        Shifted ``ClusterResult`` and the next available global label
        offset (``max(shifted_labels) + 1``, or ``offset`` if empty).
    """
    np = _numpy()

    if len(result.labels) == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), []), offset

    offset_labels = np.asarray(result.labels, dtype=np.int32) + np.int32(
        offset
    )
    if include_representatives and result.representatives:
        offset_reps = [
            rep._replace(cluster_id=np.int32(int(rep.cluster_id) + offset))
            for rep in result.representatives
        ]
    else:
        offset_reps = []
    next_offset = int(np.max(offset_labels)) + 1
    return ClusterResult(offset_labels, offset_reps), next_offset


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
        if elem == UNKNOWN_CHARGE:
            parsed: ChargeValue = UNKNOWN_CHARGE
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


def parse_charge_bucket_list(value: Any) -> tuple[ChargeBucket, ...]:
    """
    Validate and canonicalize a non-empty charge-bucket specification.

    Accepts a non-empty list or tuple of non-overlapping
    int8/``"unknown"`` buckets and at most one ``"other"`` sentinel.

    Parameters
    ----------
    value : Any
        Raw sequence of charge buckets.

    Returns
    -------
    tuple[ChargeBucket, ...]
        Canonical immutable charge bucket tuple.

    Raises
    ------
    ValueError
        If ``value`` is not a non-empty sequence of valid,
        non-overlapping charge buckets.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "Charge buckets must be a non-empty list or tuple of buckets"
        )
    if len(value) == 0:
        raise ValueError("Charge buckets must not be empty")

    buckets: list[ChargeBucket] = []
    seen_values: set[ChargeValue] = set()
    seen_other = False
    for raw_bucket in value:
        if raw_bucket == OTHER_BUCKET:
            if seen_other:
                raise ValueError(
                    "Charge bucket 'other' appears more than once"
                )
            seen_other = True
            buckets.append(OTHER_BUCKET)
            continue
        buckets.append(_parse_single_charge_bucket(raw_bucket, seen_values))

    return tuple(buckets)


def parse_optional_charge_buckets(
    value: Any,
) -> tuple[ChargeBucket, ...] | None:
    """
    Validate ``precursor_charge_buckets``, allowing ``None``/``"auto"``.

    Parameters
    ----------
    value : Any
        Raw charge bucket specification from Python, TOML, or CLI.

    Returns
    -------
    tuple[ChargeBucket, ...] | None
        Canonical immutable charge bucket tuple, or ``None`` for auto
        mode.
    """
    if value is None or value == AUTO_SENTINEL:
        return None
    return parse_charge_bucket_list(value)


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
        ``(n_bins, start_dim, end_dim)`` on the ``bin_size`` grid with
        outward-safe float64 bounds ``start_dim <= min_mz`` and
        ``end_dim > max_mz``.

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
