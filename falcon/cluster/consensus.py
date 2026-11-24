"""Representative spectrum records, medoid selection, and averaging."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numba as nb
import numpy as np

from ..config import mz_bin_bounds
from .distance import condensed_index
from .refine import iter_cluster_groups

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..config import ConsensusConfig

_MIN_MEDOID_CLUSTER_SIZE = 3
_MIN_BIN_OCCUPANCY_FRACTION = 0.7


class SpectrumTuple(NamedTuple):
    """Compact spectrum record for similarity and consensus code."""

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


def get_representative_spectra(
    spectra: Sequence[SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    consensus: ConsensusConfig,
    *,
    pdist: np.ndarray | None = None,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
    fragment_tol: float = 0.05,
) -> list[ConsensusTuple]:
    """Compute a representative spectrum per cluster in an interval."""
    if consensus.method == "medoid":
        if pdist is None:
            raise ValueError("pdist is required for medoid consensus")
        return get_cluster_medoids(spectra, labels, rts, order_map, pdist)
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
    empty = (
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    lengths = [len(s.mz) for s in spectra]
    if n_spectra == 0 or sum(lengths) == 0:
        return empty

    all_mz = np.concatenate(
        [np.asarray(s.mz, dtype=np.float32) for s in spectra]
    )
    all_int = np.concatenate(
        [np.asarray(s.intensity, dtype=np.float32) for s in spectra]
    )
    spec_ids = np.repeat(np.arange(n_spectra, dtype=np.int32), lengths)

    bin_idx = np.floor((all_mz.astype(np.float64) - start_dim) / bin_size)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return empty

    bin_idx_i64 = bin_idx[valid].astype(np.int64)
    order = np.argsort(bin_idx_i64, kind="stable")
    sorted_bins = bin_idx_i64[order]
    sorted_mz = all_mz[valid][order]
    sorted_int = all_int[valid][order]
    sorted_spec_ids = spec_ids[valid][order]

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
        return empty
    return sorted_mz, sorted_int, starts[keep], stops[keep]


@nb.njit(cache=True)
def sigma_clip(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip intensity outliers iteratively in an m/z bin in float64."""
    while len(intensities) > 2:
        vals_f64 = intensities.astype(np.float64)
        med = float(np.median(vals_f64))
        std = float(np.std(vals_f64))
        if std == 0.0:
            break
        lower = med - outlier_cutoff_lower * std
        upper = med + outlier_cutoff_upper * std
        mask = (intensities >= lower) & (intensities <= upper)
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
    outlier_cutoff_lower: float = 1.5,
    outlier_cutoff_upper: float = 1.5,
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
