"""Representative spectrum selection via medoids or peak averaging."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

    from ..config import ConsensusConfig
    from ..types import SpectrumTuple

_MIN_MEDOID_CLUSTER_SIZE = 3
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
    """Compute a representative spectrum per cluster in an interval."""
    if consensus.method == "medoid":
        if pdist is None:
            raise ValueError("pdist is required for medoid consensus")
        return get_cluster_medoids(spectra, labels, rts, order_map, pdist)
    if consensus.method == "average":
        if min_mz is None or max_mz is None or fragment_tol is None:
            raise ValueError(
                "min_mz, max_mz, and fragment_tol are required for "
                "average consensus"
            )
        return get_cluster_averages(
            spectra,
            labels,
            rts,
            order_map,
            min_mz=min_mz,
            max_mz=max_mz,
            fragment_tol=fragment_tol,
            outlier_cutoff_lower=consensus.outlier_cutoff_lower,
            outlier_cutoff_upper=consensus.outlier_cutoff_upper,
        )
    raise ValueError(f"Unsupported consensus config: {consensus!r}")


@nb.njit(cache=True, boundscheck=False)
def _select_medoid_positions(
    labels: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
    n_spectra: int,
) -> np.ndarray:
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


def get_cluster_medoids(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> list[ConsensusTuple]:
    """Select the medoid representative spectrum for each cluster."""
    if labels.shape[0] == 0:
        return []
    positions = _select_medoid_positions(
        labels, order_map, pdist, len(spectra)
    )
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

    Retains only bins occupied by at least 70% of cluster spectra.
    Returns ``(sorted_mz, sorted_intensity, bin_starts, bin_stops)``.
    """
    n_spectra = len(spectra)
    empty_f32 = np.empty(0, dtype=np.float32)
    empty_i64 = np.empty(0, dtype=np.int64)
    if n_spectra == 0:
        return empty_f32, empty_f32, empty_i64, empty_i64

    all_mz, all_int, offsets = pack_spectra_csr(spectra)
    if len(all_mz) == 0:
        return empty_f32, empty_f32, empty_i64, empty_i64

    spec_ids = np.repeat(
        np.arange(n_spectra, dtype=np.int32), np.diff(offsets)
    )
    bin_idx = np.floor((all_mz.astype(np.float64) - start_dim) / bin_size)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return empty_f32, empty_f32, empty_i64, empty_i64

    bin_idx_i64 = bin_idx[valid].astype(np.int64)
    all_mz = all_mz[valid]
    all_int = all_int[valid]
    spec_ids = spec_ids[valid]

    order = np.argsort(bin_idx_i64, kind="stable")
    sorted_bins = bin_idx_i64[order]
    sorted_mz = all_mz[order]
    sorted_int = all_int[order]
    sorted_spec_ids = spec_ids[order]

    bin_changed = sorted_bins[1:] != sorted_bins[:-1]
    change = np.flatnonzero(bin_changed) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change))
    stops = np.concatenate(
        (change, np.array([len(sorted_bins)], dtype=np.int64))
    )

    is_new_spec = np.empty(len(sorted_bins), dtype=np.int64)
    is_new_spec[0] = 1
    is_new_spec[1:] = bin_changed | (
        sorted_spec_ids[1:] != sorted_spec_ids[:-1]
    )
    distinct_spectra = np.add.reduceat(is_new_spec, starts)
    keep = distinct_spectra >= _MIN_BIN_OCCUPANCY_FRACTION * n_spectra
    if not np.any(keep):
        return empty_f32, empty_f32, empty_i64, empty_i64
    return sorted_mz, sorted_int, starts[keep], stops[keep]


@nb.njit(cache=True)
def _std_float64(values: np.ndarray) -> float:
    n = len(values)
    mean = 0.0
    m2 = 0.0
    for i in range(n):
        x = float(values[i])
        delta = x - mean
        mean += delta / (i + 1)
        m2 += delta * (x - mean)
    return math.sqrt(max(0.0, m2 / n))


@nb.njit(cache=True)
def sigma_clip(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip intensity outliers iteratively in an m/z bin in float64."""
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
    n_bins = len(bin_starts)
    out_mz = np.empty(n_bins, dtype=np.float32)
    out_int = np.empty(n_bins, dtype=np.float32)
    written = 0

    for k in range(n_bins):
        lo, hi = bin_starts[k], bin_stops[k]
        intensities = sorted_int[lo:hi]
        mzs = sorted_mz[lo:hi]
        if len(intensities) > 2:
            intensities, mzs = sigma_clip(
                intensities, mzs, outlier_cutoff_lower, outlier_cutoff_upper
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
    """
    if labels.shape[0] == 0:
        return []
    bin_size = 2.0 * fragment_tol
    n_bins, start_dim, _ = mz_bin_bounds(min_mz, max_mz, bin_size)
    reps: list[ConsensusTuple] = []

    for start_i, stop_i in iter_cluster_groups(labels):
        if stop_i - start_i > 1:
            members = [
                spectra[int(order_map[i])] for i in range(start_i, stop_i)
            ]
            avg_mz, avg_int = _clip_and_average_bins(
                *bin_cluster_peaks(members, start_dim, n_bins, bin_size),
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            reps.append(
                make_consensus_tuple(
                    precursor_mz=float(
                        np.mean(
                            [s.precursor_mz for s in members], dtype=np.float64
                        )
                    ),
                    precursor_charge=members[0].precursor_charge,
                    mz=avg_mz,
                    intensity=avg_int,
                    retention_time=float(
                        np.mean(rts[start_i:stop_i], dtype=np.float64)
                    ),
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
