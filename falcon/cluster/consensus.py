"""Representative spectrum selection via medoids or peak averaging."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numba as nb
import numpy as np

from ..types import (
    DEFAULT_OUTLIER_CUTOFF_LOWER,
    DEFAULT_OUTLIER_CUTOFF_UPPER,
    ConsensusTuple,
    make_consensus_tuple,
    mz_bin_bounds,
)
from .distance import condensed_index, pack_spectra_csr
from .refine import iter_cluster_groups

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..config import AverageConsensusConfig, ConsensusConfig
    from ..types import SpectrumTuple

# Minimum cluster size required to select a medoid; 1- and 2-member
# clusters fall back to their first spectrum as representative.
_MIN_MEDOID_CLUSTER_SIZE = 3
# Minimum fraction of cluster spectra that must contain a peak in a bin
# for that bin to be retained in an averaged consensus spectrum.
_MIN_BIN_OCCUPANCY_FRACTION = 0.7

__all__ = [
    "bin_cluster_peaks",
    "get_cluster_averages",
    "get_cluster_medoids",
    "get_representative_spectra",
    "sigma_clip",
]


def get_representative_spectra(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    consensus: ConsensusConfig,
    *,
    pdist: np.ndarray | None = None,
    min_mz: float | None = None,
    max_mz: float | None = None,
    fragment_tol: float | None = None,
) -> list[ConsensusTuple]:
    """
    Compute one representative spectrum per cluster in an interval.

    Dispatches to :func:`get_cluster_medoids` or
    :func:`get_cluster_averages` based on ``consensus``.

    Parameters
    ----------
    spectra : Sequence[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels for the interval.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Index mapping from positions in ``labels`` back to ``spectra``
        and ``pdist``.
    consensus : ConsensusConfig
        Validated consensus configuration (medoid or average).
    pdist : np.ndarray | None, default=None
        Condensed pairwise distance matrix (required for medoid).
    min_mz, max_mz : float | None, default=None
        Configured lower and upper fragment m/z bounds (required for
        average).
    fragment_tol : float | None, default=None
        Fragment tolerance in Da (required for average; bin width is
        ``2 * fragment_tol``).

    Returns
    -------
    list[ConsensusTuple]
        Representative spectra ordered by cluster label.

    Raises
    ------
    ValueError
        If ``consensus`` is unsupported or required mode-specific
        arguments are omitted.
    """
    method = getattr(consensus, "method", None)
    if method == "medoid":
        if pdist is None:
            raise ValueError("pdist is required for medoid consensus")
        return get_cluster_medoids(spectra, labels, rts, order_map, pdist)
    if method == "average":
        if min_mz is None or max_mz is None or fragment_tol is None:
            raise ValueError(
                "min_mz, max_mz, and fragment_tol are required for "
                "average consensus"
            )
        avg_cfg = cast("AverageConsensusConfig", consensus)
        return get_cluster_averages(
            spectra,
            labels,
            rts,
            order_map,
            min_mz=min_mz,
            max_mz=max_mz,
            fragment_tol=fragment_tol,
            outlier_cutoff_lower=avg_cfg.outlier_cutoff_lower,
            outlier_cutoff_upper=avg_cfg.outlier_cutoff_upper,
        )
    raise ValueError(f"Unsupported consensus config: {consensus!r}")


@nb.njit(cache=True, boundscheck=False)
def _select_medoid_positions(
    labels: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
    n_spectra: int,
) -> np.ndarray:
    """
    Return the position in ``labels`` of each cluster's medoid.

    Clusters with at least ``_MIN_MEDOID_CLUSTER_SIZE`` members choose
    the member minimizing summed float64 within-cluster distance;
    smaller clusters choose their first member.
    """
    n = labels.shape[0]
    positions = np.empty(n, dtype=np.int64)
    group_idx = 0
    pos = 0
    # Keep group scanning inline: calling a cached @njit generator from
    # another cached @njit kernel triggers Numba cache KeyError bugs.
    while pos < n:
        label = labels[pos]
        start_i = pos
        pos += 1
        if label != -1:
            while pos < n and labels[pos] == label:
                pos += 1
        size = pos - start_i
        medoid_offset = 0
        if size >= _MIN_MEDOID_CLUSTER_SIZE:
            row_sum = np.zeros(size, np.float64)
            for row in range(size):
                i_spec = order_map[start_i + row]
                for col in range(row + 1, size):
                    j_spec = order_map[start_i + col]
                    if i_spec == j_spec:
                        continue
                    dist = float(
                        pdist[condensed_index(i_spec, j_spec, n_spectra)]
                    )
                    row_sum[row] += dist
                    row_sum[col] += dist
            medoid_offset = int(np.argmin(row_sum))
        positions[group_idx] = start_i + medoid_offset
        group_idx += 1
    return positions[:group_idx]


def _validate_consensus_inputs(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    *,
    pdist: np.ndarray | None = None,
) -> int:
    """Validate 1D consensus inputs and return ``len(labels)``."""
    if (
        labels.ndim != 1
        or order_map.ndim != 1
        or rts.ndim != 1
        or (pdist is not None and pdist.ndim != 1)
    ):
        raise ValueError("Consensus input arrays must all be 1D arrays")
    if not np.issubdtype(order_map.dtype, np.integer):
        raise ValueError(
            f"order_map must have integer dtype, got {order_map.dtype}"
        )
    n_labels = labels.shape[0]
    if order_map.shape[0] != n_labels or rts.shape[0] != n_labels:
        raise ValueError(
            "labels, order_map, and rts must have matching lengths, "
            f"got {n_labels}, {order_map.shape[0]}, and {rts.shape[0]}"
        )
    n_spectra = len(spectra)
    if n_labels > 0 and (
        int(order_map.min()) < 0 or int(order_map.max()) >= n_spectra
    ):
        raise ValueError(f"order_map indices must lie in [0, {n_spectra})")
    return n_labels


def get_cluster_medoids(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> list[ConsensusTuple]:
    """
    Select the medoid representative spectrum for each cluster.

    Parameters
    ----------
    spectra : Sequence[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels for the interval.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Index mapping from positions in ``labels`` back to ``spectra``
        and ``pdist``.
    pdist : np.ndarray
        Condensed pairwise distance matrix for ``spectra``.

    Returns
    -------
    list[ConsensusTuple]
        Medoid representative spectra ordered by cluster label.

    Raises
    ------
    ValueError
        If input arrays are not 1D, lengths disagree, ``order_map``
        contains out-of-bounds indices, or ``pdist`` is undersized when
        a cluster has at least ``_MIN_MEDOID_CLUSTER_SIZE`` members.
    """
    n_labels = _validate_consensus_inputs(
        spectra, labels, rts, order_map, pdist=pdist
    )
    if n_labels == 0:
        return []
    n_spectra = len(spectra)
    required = n_spectra * (n_spectra - 1) // 2
    if pdist.shape[0] < required and any(
        stop_i - start_i >= _MIN_MEDOID_CLUSTER_SIZE
        for start_i, stop_i in iter_cluster_groups(labels)
    ):
        raise ValueError(
            f"pdist length ({pdist.shape[0]}) must be at least "
            f"{required} for {n_spectra} spectra"
        )
    positions = _select_medoid_positions(labels, order_map, pdist, n_spectra)
    return [
        ConsensusTuple.from_spectrum(
            spectra[int(order_map[pos])],
            float(rts[pos]),
            int(labels[pos]),
        )
        for pos in positions
    ]


def bin_cluster_peaks(
    spectra: Sequence[SpectrumTuple],
    start_dim: float,
    n_bins: int,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparsely bin cluster fragment peaks relative to ``start_dim``.

    Retains only bins occupied by at least
    ``_MIN_BIN_OCCUPANCY_FRACTION`` of the cluster's spectra without
    allocating dense per-bin containers over empty m/z space.

    Parameters
    ----------
    spectra : Sequence[SpectrumTuple]
        Cluster member spectra to bin jointly.
    start_dim : float
        Grid-aligned lower m/z boundary from ``mz_bin_bounds``.
    n_bins : int
        Total number of grid bins from ``mz_bin_bounds``.
    bin_size : float
        Width of each m/z bin in Da.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(sorted_mz, sorted_intensity, bin_starts, bin_stops)`` where
        surviving bin ``k`` occupies ``bin_starts[k]:bin_stops[k]``.
    """
    n_spectra = len(spectra)
    empty_f32 = np.empty(0, dtype=np.float32)
    empty_i64 = np.empty(0, dtype=np.int64)
    if n_spectra == 0:
        return empty_f32, empty_f32, empty_i64, empty_i64

    all_mz, all_int, offsets = pack_spectra_csr(spectra)
    if len(all_mz) == 0:
        return empty_f32, empty_f32, empty_i64, empty_i64

    lengths = np.diff(offsets)
    spec_ids = np.repeat(np.arange(n_spectra, dtype=np.int32), lengths)

    bin_idx = np.floor((all_mz.astype(np.float64) - start_dim) / bin_size)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return empty_f32, empty_f32, empty_i64, empty_i64

    bin_idx_i64 = bin_idx[valid].astype(np.int64)
    all_mz = all_mz[valid]
    all_int = all_int[valid]
    spec_ids = spec_ids[valid]

    # Stable sort preserves per-spectrum ordering inside each bin so
    # spec_ids is non-decreasing within every bin slice.
    order = np.argsort(bin_idx_i64, kind="stable")
    sorted_bins = bin_idx_i64[order]
    sorted_mz = all_mz[order]
    sorted_int = all_int[order]
    sorted_spec_ids = spec_ids[order]

    change = np.flatnonzero(sorted_bins[1:] != sorted_bins[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change))
    stops = np.concatenate(
        (change, np.array([len(sorted_bins)], dtype=np.int64))
    )

    is_new_spec = np.empty(len(sorted_bins), dtype=np.int64)
    is_new_spec[0] = 1
    is_new_spec[1:] = (sorted_bins[1:] != sorted_bins[:-1]) | (
        sorted_spec_ids[1:] != sorted_spec_ids[:-1]
    )
    distinct_spectra = np.add.reduceat(is_new_spec, starts)
    keep = distinct_spectra >= _MIN_BIN_OCCUPANCY_FRACTION * n_spectra
    if not np.any(keep):
        return empty_f32, empty_f32, empty_i64, empty_i64
    return sorted_mz, sorted_int, starts[keep], stops[keep]


@nb.njit(cache=True)
def _std_float64(values: np.ndarray) -> float:
    """Compute population standard deviation in float64 via Welford."""
    n = len(values)
    mean = 0.0
    m2 = 0.0
    for i in range(n):
        x = float(values[i])
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        m2 += delta * delta2
    return math.sqrt(max(0.0, m2 / n))


@nb.njit(cache=True)
def sigma_clip(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove intensity outliers iteratively from a single m/z bin.

    Computes median and population standard deviation in float64 to
    prevent float32 accumulation drift on large clusters while keeping
    the returned arrays in their input dtypes. Stops once at most two
    observations remain, the standard deviation reaches zero, or an
    iteration removes no values. Callers must pass 1D arrays of equal
    length.

    Parameters
    ----------
    intensities : np.ndarray
        Peak intensities observed in the bin.
    mzs : np.ndarray
        Peak m/z values aligned with ``intensities``.
    outlier_cutoff_lower : float
        Lower bound in standard deviations below the median.
    outlier_cutoff_upper : float
        Upper bound in standard deviations above the median.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Retained intensities and their corresponding m/z values.
    """
    while len(intensities) > 2:
        med = float(np.median(intensities.astype(np.float64)))
        std = _std_float64(intensities)
        if std == 0.0:
            break
        lower_bound = med - outlier_cutoff_lower * std
        upper_bound = med + outlier_cutoff_upper * std
        mask = (intensities >= lower_bound) & (intensities <= upper_bound)
        if np.sum(mask) == len(intensities):
            break
        intensities = intensities[mask]
        mzs = mzs[mask]
    return intensities, mzs


@nb.njit(cache=True)
def _clip_and_average_bins(
    sorted_mz: np.ndarray,
    sorted_int: np.ndarray,
    bin_starts: np.ndarray,
    bin_stops: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma-clip retained bins and return float32 (mz, intensity)."""
    n_bins = len(bin_starts)
    out_mz = np.empty(n_bins, dtype=np.float32)
    out_int = np.empty(n_bins, dtype=np.float32)
    written = 0

    for k in range(n_bins):
        lo = bin_starts[k]
        hi = bin_stops[k]
        intensities = sorted_int[lo:hi]
        mzs = sorted_mz[lo:hi]
        if len(intensities) > 2:
            intensities, mzs = sigma_clip(
                intensities,
                mzs,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            if len(intensities) == 0:
                continue
        count = len(intensities)
        mz_sum = 0.0
        int_sum = 0.0
        for idx in range(count):
            mz_sum += float(mzs[idx])
            int_sum += float(intensities[idx])
        out_mz[written] = np.float32(mz_sum / count)
        out_int[written] = np.float32(int_sum / count)
        written += 1

    return out_mz[:written], out_int[:written]


def get_cluster_averages(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    *,
    min_mz: float,
    max_mz: float,
    fragment_tol: float,
    outlier_cutoff_lower: float = DEFAULT_OUTLIER_CUTOFF_LOWER,
    outlier_cutoff_upper: float = DEFAULT_OUTLIER_CUTOFF_UPPER,
) -> list[ConsensusTuple]:
    """
    Compute intensity-averaged consensus spectra for each cluster.

    Adapted from Carr et al., Proteomics (2024),
    doi:10.1002/pmic.202300234.

    Parameters
    ----------
    spectra : Sequence[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels for the interval.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Index mapping from positions in ``labels`` back to ``spectra``.
    min_mz, max_mz : float
        Configured lower and upper fragment m/z bounds.
    fragment_tol : float
        Fragment tolerance in Da (bin width is ``2 * fragment_tol``).
    outlier_cutoff_lower : float, default=1.5
        Lower sigma-clipping cutoff in standard deviations.
    outlier_cutoff_upper : float, default=1.5
        Upper sigma-clipping cutoff in standard deviations.

    Returns
    -------
    list[ConsensusTuple]
        Averaged consensus spectra ordered by cluster label.

    Raises
    ------
    ValueError
        If input arrays are not 1D, lengths disagree, or ``order_map``
        contains out-of-bounds indices.
    """
    n_labels = _validate_consensus_inputs(spectra, labels, rts, order_map)
    bin_size = 2.0 * fragment_tol
    n_bins, start_dim, _ = mz_bin_bounds(min_mz, max_mz, bin_size)
    if n_labels == 0:
        return []
    reps: list[ConsensusTuple] = []

    for start_i, stop_i in iter_cluster_groups(labels):
        if stop_i - start_i > 1:
            members = [
                spectra[int(order_map[i])] for i in range(start_i, stop_i)
            ]
            avg_precursor_mz = float(
                np.mean(
                    [spec.precursor_mz for spec in members], dtype=np.float64
                )
            )
            charge = members[0].precursor_charge
            avg_rt = float(np.mean(rts[start_i:stop_i], dtype=np.float64))

            sorted_mz, sorted_int, bin_starts, bin_stops = bin_cluster_peaks(
                members, start_dim, n_bins, bin_size
            )
            avg_mz, avg_int = _clip_and_average_bins(
                sorted_mz,
                sorted_int,
                bin_starts,
                bin_stops,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            reps.append(
                make_consensus_tuple(
                    precursor_mz=avg_precursor_mz,
                    precursor_charge=charge,
                    mz=avg_mz,
                    intensity=avg_int,
                    retention_time=avg_rt,
                    cluster_id=int(labels[start_i]),
                )
            )
        else:
            reps.append(
                ConsensusTuple.from_spectrum(
                    spectra[int(order_map[start_i])],
                    float(rts[start_i]),
                    int(labels[start_i]),
                )
            )
    return reps
