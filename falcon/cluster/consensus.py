"""Representative spectrum selection via medoids or peak averaging."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numba as nb
import numpy as np

from ..config.models import (
    AverageConsensusConfig,
    ConsensusConfig,
    MedoidConsensusConfig,
)
from ..types import (
    ConsensusTuple,
    SpectrumTuple,
    make_consensus_tuple,
    mz_bin_bounds,
)
from .distance import condensed_index_unchecked

# Minimum cluster size required to select a medoid; 1- and 2-member
# clusters fall back to their first spectrum as representative.
_MIN_MEDOID_CLUSTER_SIZE = 3
# Minimum fraction of cluster spectra that must contain a peak in a bin
# for that bin to be retained in an averaged consensus spectrum.
_MIN_BIN_OCCUPANCY_FRACTION = 0.7


@nb.njit(cache=True)
def iter_cluster_groups(clusters: np.ndarray) -> Iterator[tuple[int, int]]:
    """
    Yield ``(start, stop)`` index slices for contiguous cluster labels.

    Assumes ``clusters`` is sorted by label with any ``-1`` noise labels
    first; each ``-1`` entry is yielded as a length-1 slice.

    Parameters
    ----------
    clusters : np.ndarray
        Sorted 1D cluster label array (``-1`` denotes noise points).

    Yields
    ------
    tuple[int, int]
        Half-open ``(start_i, stop_i)`` slice bounds for each group.
    """
    n = clusters.shape[0]
    start_i = 0
    while start_i < n and clusters[start_i] == -1:
        yield start_i, start_i + 1
        start_i += 1
    stop_i = start_i
    while stop_i < n:
        label = clusters[stop_i]
        start_i = stop_i
        while stop_i < n and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


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
        and the condensed distance matrix.
    consensus : ConsensusConfig
        Validated consensus configuration (medoid or average).
    pdist : np.ndarray | None, default=None
        Condensed pairwise distance matrix (required for medoid).
    min_mz : float | None, default=None
        Configured lower fragment m/z bound (required for average).
    max_mz : float | None, default=None
        Configured upper fragment m/z bound (required for average).
    fragment_tol : float | None, default=None
        Fragment tolerance in Da (required for average; bin width is
        ``2 * fragment_tol``).

    Returns
    -------
    list[ConsensusTuple]
        Representative spectra ordered by cluster label.
    """
    if isinstance(consensus, MedoidConsensusConfig):
        if pdist is None:
            raise ValueError("pdist is required for medoid consensus")
        return _get_cluster_medoids(spectra, labels, rts, order_map, pdist)

    if isinstance(consensus, AverageConsensusConfig):
        if min_mz is None or max_mz is None or fragment_tol is None:
            raise ValueError(
                "min_mz, max_mz, and fragment_tol are required for "
                "average consensus"
            )
        return _get_cluster_averages(
            spectra,
            labels,
            rts,
            order_map,
            min_mz=min_mz,
            max_mz=max_mz,
            bin_size=2.0 * fragment_tol,
            outlier_cutoff_lower=consensus.outlier_cutoff_lower,
            outlier_cutoff_upper=consensus.outlier_cutoff_upper,
        )

    raise ValueError(f"Unsupported consensus config: {consensus!r}")


@nb.njit(cache=True, fastmath=True, boundscheck=False)
def _select_medoid_positions(
    labels: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
    n_spectra: int,
) -> np.ndarray:
    """
    Return the position in ``labels`` of each cluster's medoid.

    Clusters with at least ``_MIN_MEDOID_CLUSTER_SIZE`` members choose
    the member minimizing summed float32 within-cluster distance;
    smaller clusters choose their first member.
    """
    n_groups = 0
    for _ in iter_cluster_groups(labels):
        n_groups += 1

    positions = np.empty(n_groups, dtype=np.int64)
    for group_idx, (start_i, stop_i) in enumerate(iter_cluster_groups(labels)):
        size = stop_i - start_i
        medoid_offset = 0
        if size >= _MIN_MEDOID_CLUSTER_SIZE:
            row_sum = np.zeros(size, np.float32)
            for row in range(size):
                i_spec = order_map[start_i + row]
                for col in range(row + 1, size):
                    j_spec = order_map[start_i + col]
                    dist = pdist[
                        condensed_index_unchecked(i_spec, j_spec, n_spectra)
                    ]
                    row_sum[row] += dist
                    row_sum[col] += dist
            medoid_offset = int(np.argmin(row_sum))
        positions[group_idx] = start_i + medoid_offset
    return positions


def _get_cluster_medoids(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> list[ConsensusTuple]:
    """Select the medoid representative spectrum for each cluster."""
    positions = _select_medoid_positions(
        labels, order_map, pdist, len(spectra)
    )
    reps: list[ConsensusTuple] = []
    for pos in positions:
        spec = spectra[int(order_map[pos])]
        reps.append(
            make_consensus_tuple(
                precursor_mz=spec.precursor_mz,
                precursor_charge=spec.precursor_charge,
                mz=spec.mz,
                intensity=spec.intensity,
                retention_time=float(rts[pos]),
                cluster_id=int(labels[pos]),
            )
        )
    return reps


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

    lengths = np.fromiter(
        (len(spec.mz) for spec in spectra), dtype=np.int64, count=n_spectra
    )
    if int(lengths.sum()) == 0:
        return empty_f32, empty_f32, empty_i64, empty_i64

    all_mz = np.concatenate(
        [np.asarray(spec.mz, dtype=np.float32) for spec in spectra]
    )
    all_int = np.concatenate(
        [np.asarray(spec.intensity, dtype=np.float32) for spec in spectra]
    )
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
    order = np.argsort(bin_idx_i64, kind="mergesort")
    sorted_bins = bin_idx_i64[order]
    sorted_mz = np.ascontiguousarray(all_mz[order], dtype=np.float32)
    sorted_int = np.ascontiguousarray(all_int[order], dtype=np.float32)
    sorted_spec_ids = spec_ids[order]

    change = np.flatnonzero(sorted_bins[1:] != sorted_bins[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(sorted_bins)]))

    min_spectra = _MIN_BIN_OCCUPANCY_FRACTION * n_spectra
    keep_starts: list[int] = []
    keep_stops: list[int] = []
    for lo, hi in zip(starts, stops, strict=True):
        s_slice = sorted_spec_ids[lo:hi]
        distinct_spectra = 1 + int(
            np.count_nonzero(s_slice[1:] != s_slice[:-1])
        )
        if distinct_spectra >= min_spectra:
            keep_starts.append(int(lo))
            keep_stops.append(int(hi))

    if not keep_starts:
        return empty_f32, empty_f32, empty_i64, empty_i64
    return (
        sorted_mz,
        sorted_int,
        np.asarray(keep_starts, dtype=np.int64),
        np.asarray(keep_stops, dtype=np.int64),
    )


@nb.njit(cache=True)
def sigma_clip(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove intensity outliers iteratively from a single m/z bin.

    Stops once at most two observations remain, the standard deviation
    reaches zero, or an iteration removes no values.

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
        med = np.median(intensities)
        std = np.std(intensities)
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
        out_mz[written] = np.mean(mzs)
        out_int[written] = np.mean(intensities)
        written += 1

    return out_mz[:written], out_int[:written]


def _get_cluster_averages(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    *,
    min_mz: float,
    max_mz: float,
    bin_size: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> list[ConsensusTuple]:
    """
    Compute intensity-averaged consensus spectra for each cluster.

    Adapted from Carr et al., Proteomics (2024):
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.202300234.
    """
    n_bins, start_dim, _ = mz_bin_bounds(min_mz, max_mz, bin_size)
    reps: list[ConsensusTuple] = []

    for start_i, stop_i in iter_cluster_groups(labels):
        if stop_i - start_i > 1:
            members = [
                spectra[int(order_map[i])] for i in range(start_i, stop_i)
            ]
            avg_precursor_mz = float(
                np.mean([spec.precursor_mz for spec in members])
            )
            charge = members[0].precursor_charge
            avg_rt = float(np.mean(rts[start_i:stop_i]))

            sorted_mz, sorted_int, bin_starts, bin_stops = bin_cluster_peaks(
                members, start_dim, n_bins, bin_size
            )
            if len(bin_starts) > 0:
                avg_mz, avg_int = _clip_and_average_bins(
                    sorted_mz,
                    sorted_int,
                    bin_starts,
                    bin_stops,
                    outlier_cutoff_lower,
                    outlier_cutoff_upper,
                )
            else:
                avg_mz = np.empty(0, dtype=np.float32)
                avg_int = np.empty(0, dtype=np.float32)

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
            singleton = spectra[int(order_map[start_i])]
            reps.append(
                make_consensus_tuple(
                    precursor_mz=singleton.precursor_mz,
                    precursor_charge=singleton.precursor_charge,
                    mz=singleton.mz,
                    intensity=singleton.intensity,
                    retention_time=float(rts[start_i]),
                    cluster_id=int(labels[start_i]),
                )
            )
    return reps
