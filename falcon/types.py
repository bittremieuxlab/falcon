"""Shared domain data types, charge-bucket parsing, and m/z bin math.

Imports only the standard library at module load time so CLI inspection
commands never pull in NumPy, Numba, SciPy, Lance, or spectrum_utils.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

if TYPE_CHECKING:
    import numpy as np

INT8_MIN = -128
INT8_MAX = 127

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
        """Build a ``ConsensusTuple`` from a ``SpectrumTuple``."""
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
    """
    import numpy as np

    if precursor_charge is None or not math.isfinite(float(precursor_charge)):
        norm_charge: np.int32 | float = np.nan
    else:
        norm_charge = np.int32(int(precursor_charge))
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

    Does not mutate ``result.labels`` in place. Returns the shifted
    ``ClusterResult`` and the next available global label offset.
    """
    import numpy as np

    labels_arr = np.asarray(result.labels, dtype=np.int32)
    if labels_arr.size == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), []), offset

    offset_labels = labels_arr + np.int32(offset)
    offset_reps = (
        [
            rep._replace(cluster_id=np.int32(int(rep.cluster_id) + offset))
            for rep in result.representatives
        ]
        if include_representatives and result.representatives
        else []
    )
    next_offset = int(np.max(offset_labels)) + 1
    return ClusterResult(offset_labels, offset_reps), next_offset


def charge_sort_key(value: ChargeValue) -> tuple[bool, int | str]:
    """Sort numeric precursor charges ascending before ``"unknown"``."""
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
    if not raw_bucket:
        raise ValueError("Charge bucket must not be empty")

    bucket_seen: set[ChargeValue] = set()
    for elem in raw_bucket:
        if elem == UNKNOWN_CHARGE:
            parsed: ChargeValue = UNKNOWN_CHARGE
        elif isinstance(elem, int) and not isinstance(elem, bool):
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
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "Charge buckets must be a non-empty list or tuple of buckets"
        )
    if not value:
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
        else:
            buckets.append(
                _parse_single_charge_bucket(raw_bucket, seen_values)
            )
    return tuple(buckets)


def parse_optional_charge_buckets(
    value: Any,
) -> tuple[ChargeBucket, ...] | None:
    """Validate ``precursor_charge_buckets`` (or ``None``/``auto``)."""
    if value is None or value == AUTO_SENTINEL:
        return None
    return parse_charge_bucket_list(value)


def mz_bin_bounds(
    min_mz: float, max_mz: float, bin_size: float
) -> tuple[int, float, float]:
    """
    Compute outward-aligned m/z bin boundaries and exact bin count.

    Guarantees ``start_dim <= min_mz``, ``end_dim > max_mz``, and
    ``n_bins >= 1`` in float64 precision.

    Returns
    -------
    tuple[int, float, float]
        ``(n_bins, start_dim, end_dim)`` on the ``bin_size`` grid.
    """
    if bin_size <= 0.0 or min_mz > max_mz:
        raise ValueError(
            f"Invalid m/z bin bounds: min_mz={min_mz}, max_mz={max_mz}, "
            f"bin_size={bin_size}"
        )
    start_idx = math.floor(min_mz / bin_size)
    if start_idx * bin_size > min_mz:
        start_idx -= 1
    end_idx = math.floor(max_mz / bin_size) + 1
    if end_idx * bin_size <= max_mz:
        end_idx += 1
    return end_idx - start_idx, start_idx * bin_size, end_idx * bin_size
