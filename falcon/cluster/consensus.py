"""Representative spectrum selection via medoids or peak averaging."""

import collections
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numba as nb
import numba.typed
import numpy as np

from . import similarity

# Minimum cluster size required to select a medoid; 1- and 2-member
# clusters fall back to their first spectrum as representative.
_MIN_MEDOID_CLUSTER_SIZE = 3
# Minimum fraction of cluster spectra that must contain a peak in a bin
# for that bin to be retained in an averaged consensus spectrum.
_MIN_BIN_OCCUPANCY_FRACTION = 0.7

ConsensusTuple = collections.namedtuple(
    "ConsensusTuple",
    [
        "precursor_mz",  # np.float32
        "precursor_charge",  # np.int32 or np.nan
        "mz",  # np.ndarray of np.float32
        "intensity",  # np.ndarray of np.float32
        "retention_time",  # np.float32
        "cluster_id",  # np.int32
        "mz_split",  # np.int32 or None before split tagging
    ],
)


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[tuple[int, int]]:
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
    start_i = 0
    # Yield noise points (-1) individually as singletons first.
    while start_i < clusters.shape[0] and clusters[start_i] == -1:
        yield start_i, start_i + 1
        start_i += 1
    # Yield contiguous spans for non-noise cluster labels.
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        label = clusters[stop_i]
        start_i = stop_i
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _get_representative_spectra(
    spectra: list[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    consensus_method: str,
    consensus_params: dict[str, Any],
) -> list[ConsensusTuple]:
    """
    Compute one representative spectrum per cluster in an interval.

    Parameters
    ----------
    spectra : list[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels for the interval.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Index mapping from positions in ``labels`` back to ``spectra``
        and the condensed distance matrix.
    consensus_method : {"medoid", "average"}
        Representative spectrum computation method.
    consensus_params : dict[str, Any]
        Method-specific keyword arguments (``pdist`` for ``"medoid"``;
        binning and sigma-clipping parameters for ``"average"``).

    Returns
    -------
    list[ConsensusTuple]
        Representative spectra ordered by cluster label.

    Raises
    ------
    ValueError
        If ``consensus_method`` is not ``"medoid"`` or ``"average"``.
    """
    if consensus_method == "medoid":
        columns = _get_cluster_medoids(
            spectra, labels, rts, order_map, **consensus_params
        )
    elif consensus_method == "average":
        columns = _get_cluster_average(
            spectra, labels, rts, order_map, **consensus_params
        )
    else:
        raise ValueError(
            f"Unknown consensus spectrum method: {consensus_method}"
        )
    # Build ConsensusTuples in Python (Numba cannot construct them);
    # both methods return the same six per-cluster column sequences.
    return _consensus_tuples_from_columns(*columns)


def _consensus_tuples_from_columns(
    precursor_mzs: Sequence[float],
    precursor_charges: Sequence[Any],
    mzs: Sequence[np.ndarray],
    intensities: Sequence[np.ndarray],
    retention_times: Sequence[float],
    cluster_ids: Sequence[int],
) -> list[ConsensusTuple]:
    """
    Assemble ``ConsensusTuple`` records from per-cluster column arrays.

    Parameters
    ----------
    precursor_mzs : Sequence[float]
        Per-cluster precursor m/z values.
    precursor_charges : Sequence[Any]
        Per-cluster precursor charges (``np.nan`` for missing charge).
    mzs : Sequence[np.ndarray]
        Per-cluster fragment m/z arrays.
    intensities : Sequence[np.ndarray]
        Per-cluster fragment intensity arrays.
    retention_times : Sequence[float]
        Per-cluster retention times (``np.nan`` if missing).
    cluster_ids : Sequence[int]
        Per-cluster integer labels.

    Returns
    -------
    list[ConsensusTuple]
        Representative spectra with float32/int32 dtypes and
        ``mz_split=None``.
    """
    return [
        ConsensusTuple(
            precursor_mz=np.float32(precursor_mzs[i]),
            precursor_charge=(
                np.int32(precursor_charges[i])
                if not np.isnan(precursor_charges[i])
                else np.nan
            ),
            mz=mzs[i].astype(np.float32),
            intensity=intensities[i].astype(np.float32),
            retention_time=np.float32(retention_times[i]),
            cluster_id=np.int32(cluster_ids[i]),
            mz_split=None,
        )
        for i in range(len(precursor_mzs))
    ]


@nb.njit(fastmath=True, boundscheck=False)
def _get_cluster_medoids(
    spectra: list[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> tuple[
    list[float],
    list[int | None],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[int],
]:
    """
    Select the medoid spectrum for each cluster.

    Clusters with at least three members choose the member minimizing
    summed float32 within-cluster distance; 1- and 2-member clusters
    select their first member.

    Parameters
    ----------
    spectra : list[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Mapping from ``labels`` positions to ``spectra`` / ``pdist``.
    pdist : np.ndarray
        Condensed pairwise distance matrix for ``spectra``.

    Returns
    -------
    tuple
        Six per-cluster lists: ``(precursor_mzs, precursor_charges,
        mzs, intensities, retention_times, cluster_ids)``.
    """
    m = len(spectra)
    precursor_mzs = []
    precursor_charges = []
    mzs = []
    intensities = []
    retention_times = []
    cluster_ids = []

    for start_i, stop_i in _get_cluster_group_idx(labels):
        # Clusters smaller than _MIN_MEDOID_CLUSTER_SIZE pick member 0.
        if stop_i - start_i >= _MIN_MEDOID_CLUSTER_SIZE:
            row_sum = np.zeros(stop_i - start_i, np.float32)
            for row in range(stop_i - start_i):
                for col in range(row + 1, stop_i - start_i):
                    i, j = order_map[start_i + row], order_map[start_i + col]
                    if i > j:
                        i, j = j, i
                    pdist_ij = pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    row_sum[row] += pdist_ij
                    row_sum[col] += pdist_ij
            medoid_idx = int(np.argmin(row_sum))
            medoid_spec = spectra[order_map[start_i + medoid_idx]]
            precursor_mzs.append(medoid_spec.precursor_mz)
            precursor_charges.append(medoid_spec.precursor_charge)
            mzs.append(medoid_spec.mz)
            intensities.append(medoid_spec.intensity)
            retention_times.append(rts[start_i + medoid_idx])
            cluster_ids.append(labels[start_i + medoid_idx])
        else:
            medoid_spec = spectra[order_map[start_i]]
            precursor_mzs.append(medoid_spec.precursor_mz)
            precursor_charges.append(medoid_spec.precursor_charge)
            mzs.append(medoid_spec.mz)
            intensities.append(medoid_spec.intensity)
            retention_times.append(rts[start_i])
            cluster_ids.append(labels[start_i])
    return (
        precursor_mzs,
        precursor_charges,
        mzs,
        intensities,
        retention_times,
        cluster_ids,
    )


def _get_cluster_average(
    spectra: list[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    min_mz: float,
    max_mz: float,
    bin_size: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[
    list[float],
    list[int | None],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[int],
]:
    """
    Compute intensity-averaged consensus spectra for each cluster.

    Multi-member clusters are binned in m/z, filtered by bin occupancy,
    sigma-clipped to reject intensity outliers, and averaged per bin.
    Adapted from Carr et al., Proteomics (2024):
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.202300234.

    Parameters
    ----------
    spectra : list[SpectrumTuple]
        Interval spectra in canonical input order.
    labels : np.ndarray
        Sorted cluster labels.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    order_map : np.ndarray
        Mapping from ``labels`` positions to ``spectra``.
    min_mz : float
        Minimum fragment m/z for binning.
    max_mz : float
        Maximum fragment m/z for binning.
    bin_size : float
        Width of each m/z bin (in Da).
    outlier_cutoff_lower : float
        Lower sigma-clipping cutoff in standard deviations below median.
    outlier_cutoff_upper : float
        Upper sigma-clipping cutoff in standard deviations above median.

    Returns
    -------
    tuple
        Six per-cluster lists: ``(precursor_mzs, precursor_charges,
        mzs, intensities, retention_times, cluster_ids)``.
    """
    precursor_mzs = []
    precursor_charges = []
    mzs = []
    intensities = []
    retention_times = []
    cluster_ids = []
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 1:
            spectra_to_average = [
                spectra[order_map[i]] for i in range(start_i, stop_i)
            ]
            # Average precursor m/z and RT; take charge from member 0.
            avg_mz = float(
                np.mean(
                    [spec.precursor_mz for spec in spectra_to_average],
                    axis=0,
                )
            )
            charge = spectra_to_average[0].precursor_charge
            avg_rt = float(np.mean(rts[start_i:stop_i]))

            # Bin peaks, reject intensity outliers, and average per bin.
            bins_idx, bins_peaks, bins_mz = _spectrum_binning(
                spectra_to_average, min_mz, max_mz, bin_size
            )
            del spectra_to_average
            bins_idx, bins_peaks, bins_mz = _outlier_rejection(
                bins_idx,
                bins_peaks,
                bins_mz,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            avg_spectrum = _construct_average_spectrum(
                bins_idx,
                bins_peaks,
                bins_mz,
                avg_mz,
                charge,
                avg_rt,
                labels[start_i],
            )
            precursor_mzs.append(avg_spectrum[0])
            precursor_charges.append(avg_spectrum[1])
            mzs.append(avg_spectrum[2])
            intensities.append(avg_spectrum[3])
            retention_times.append(avg_spectrum[4])
            cluster_ids.append(avg_spectrum[5])
        else:
            # Pass singleton clusters through unchanged.
            avg_spectrum = spectra[order_map[start_i]]
            precursor_mzs.append(avg_spectrum.precursor_mz)
            precursor_charges.append(avg_spectrum.precursor_charge)
            mzs.append(avg_spectrum.mz)
            intensities.append(avg_spectrum.intensity)
            retention_times.append(rts[start_i])
            cluster_ids.append(labels[start_i])
    return (
        precursor_mzs,
        precursor_charges,
        mzs,
        intensities,
        retention_times,
        cluster_ids,
    )


@nb.njit(cache=True)
def _spectrum_binning(
    spectra: list[similarity.SpectrumTuple],
    min_mz: float,
    max_mz: float,
    bin_size: float,
) -> tuple[np.ndarray, nb.typed.List, nb.typed.List]:
    """
    Bin fragment peaks from multiple spectra into fixed-width m/z bins.

    Drops bins present in fewer than ``_MIN_BIN_OCCUPANCY_FRACTION``
    of the input spectra.

    Parameters
    ----------
    spectra : list[SpectrumTuple]
        Cluster member spectra to bin jointly.
    min_mz : float
        Lower m/z boundary for binning.
    max_mz : float
        Upper m/z boundary for binning.
    bin_size : float
        Bin width in m/z units.

    Returns
    -------
    tuple[np.ndarray, nb.typed.List, nb.typed.List]
        Non-empty bin indices (int32), a Numba typed list of float32
        intensity arrays per retained bin, and a matching typed list of
        float32 m/z arrays.
    """
    start_dim = min_mz - (min_mz % bin_size)
    end_dim = max_mz + bin_size - (max_mz % bin_size)
    n_bins = math.ceil((end_dim - start_dim) / bin_size)

    n_spectra = len(spectra)

    bins_indices = -np.ones(n_bins, dtype=np.int32)
    bins_peaks = nb.typed.List()
    bins_mz = nb.typed.List()
    for _ in range(n_bins):
        nested_list_p = nb.typed.List.empty_list(nb.types.float32)
        nested_list_m = nb.typed.List.empty_list(nb.types.float32)
        bins_peaks.append(nested_list_p)
        bins_mz.append(nested_list_m)
    bins_spectra_count = np.zeros(n_bins, dtype=np.int32)

    for spec in spectra:
        bins_peak_presence = np.zeros(n_bins, dtype=np.int32)
        for mz, intensity in zip(spec.mz, spec.intensity):  # noqa: B905
            bin_idx = math.floor((mz - min_mz) / bin_size)
            if 0 <= bin_idx < n_bins:
                if bins_indices[bin_idx] == -1:
                    bins_indices[bin_idx] = bin_idx
                bins_peaks[bin_idx].append(intensity)
                bins_mz[bin_idx].append(mz)
                bins_peak_presence[bin_idx] = 1
        bins_spectra_count += bins_peak_presence
    # Discard bins present in fewer than the minimum spectrum fraction.
    for i in range(n_bins):
        if (
            bins_indices[i] != -1
            and bins_spectra_count[i] < _MIN_BIN_OCCUPANCY_FRACTION * n_spectra
        ):
            bins_indices[i] = -1
    # Compact retained bins into contiguous NumPy arrays.
    mask = bins_indices != -1
    bins_indices = bins_indices[mask]
    bins_peaks_nb = nb.typed.List()
    bins_mz_nb = nb.typed.List()
    for i in bins_indices:
        bins_peaks_nb.append(typed_list_to_numpy(bins_peaks[i]))
        bins_mz_nb.append(typed_list_to_numpy(bins_mz[i]))

    return bins_indices, bins_peaks_nb, bins_mz_nb


@nb.njit(cache=True)
def _outlier_rejection(
    bins_indices: np.ndarray,
    bins_peaks: nb.typed.List,
    bins_mz: nb.typed.List,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sigma-clip intensity outliers in each bin and average kept peaks.

    Bins with at most two peaks skip clipping and are averaged directly;
    bins left empty after clipping are dropped.

    Parameters
    ----------
    bins_indices : np.ndarray
        Indices of the non-empty bins.
    bins_peaks : nb.typed.List
        Per-bin float32 arrays of peak intensities.
    bins_mz : nb.typed.List
        Per-bin float32 arrays of peak m/z values.
    outlier_cutoff_lower : float
        Lower sigma-clipping threshold (standard deviations).
    outlier_cutoff_upper : float
        Upper sigma-clipping threshold (standard deviations).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Surviving bin indices, mean float32 peak intensities, and mean
        float32 peak m/z values.
    """
    n_peaks = len(bins_indices)
    cleaned_bins_peaks = np.zeros(n_peaks, dtype=np.float32)
    cleaned_bins_mz = np.zeros(n_peaks, dtype=np.float32)

    zero_peaks = np.zeros(len(bins_indices), dtype=np.bool_)
    for i in range(len(bins_indices)):
        intensities = bins_peaks[i]
        mzs = bins_mz[i]
        if len(intensities) > 2:
            clipped_p, clipped_m = _sigma_clipping(
                intensities,
                mzs,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            if len(clipped_p) < 1:
                zero_peaks[i] = True
            else:
                cleaned_bins_peaks[i] = np.mean(clipped_p)
                cleaned_bins_mz[i] = np.mean(clipped_m)
        else:
            cleaned_bins_peaks[i] = np.mean(intensities)
            cleaned_bins_mz[i] = np.mean(mzs)

    # Drop bins where sigma clipping removed every peak.
    return (
        bins_indices[~zero_peaks],
        cleaned_bins_peaks[~zero_peaks],
        cleaned_bins_mz[~zero_peaks],
    )


@nb.njit(cache=True)
def _sigma_clipping(
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
        mask = _sigma_clip(
            intensities, med, std, outlier_cutoff_lower, outlier_cutoff_upper
        )
        # Converged when no values fall outside the sigma bounds.
        if np.sum(mask) == len(intensities):
            break
        intensities = intensities[mask]
        mzs = mzs[mask]
    return intensities, mzs


@nb.njit(cache=True)
def _sigma_clip(
    values: np.ndarray,
    median: float,
    std: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> np.ndarray:
    """
    Compute a boolean mask selecting non-outlier intensities.

    Parameters
    ----------
    values : np.ndarray
        Intensity values to test.
    median : float
        Median intensity.
    std : float
        Standard deviation of intensities.
    outlier_cutoff_lower : float
        Standard deviations allowed below ``median``.
    outlier_cutoff_upper : float
        Standard deviations allowed above ``median``.

    Returns
    -------
    np.ndarray
        Boolean mask: ``True`` to keep, ``False`` to reject.
    """
    lower_bound = median - outlier_cutoff_lower * std
    upper_bound = median + outlier_cutoff_upper * std
    return (values >= lower_bound) & (values <= upper_bound)


@nb.njit(cache=True)
def _construct_average_spectrum(
    bins_indices: np.ndarray,
    bins_peaks: np.ndarray,
    bins_mz: np.ndarray,
    avg_precursor_mz: float,
    charge: int,
    avg_rt: float,
    cluster: int,
) -> tuple[float, int | None, np.ndarray, np.ndarray, float, int]:
    """
    Assemble averaged bin peaks and metadata into a consensus tuple.

    Parameters
    ----------
    bins_indices : np.ndarray
        Surviving bin indices (determines peak count).
    bins_peaks : np.ndarray
        Averaged float32 peak intensities per surviving bin.
    bins_mz : np.ndarray
        Averaged float32 peak m/z values per surviving bin.
    avg_precursor_mz : float
        Mean precursor m/z of the cluster.
    charge : int
        Precursor charge (or ``NaN``/``None`` when missing).
    avg_rt : float
        Mean retention time of the cluster.
    cluster : int
        Cluster label.

    Returns
    -------
    tuple[float, int | None, np.ndarray, np.ndarray, float, int]
        ``(avg_precursor_mz, charge, mz, intensity, avg_rt, cluster)``
        where missing charge is represented as ``np.nan``.
    """
    mz = np.empty(len(bins_indices), np.float32)
    intensity = np.empty(len(bins_indices), np.float32)

    for idx, (avg_intensity, avg_mz) in enumerate(
        zip(bins_peaks, bins_mz)  # noqa: B905 (Numba lacks strict=)
    ):
        mz[idx] = avg_mz
        intensity[idx] = avg_intensity
    return (
        avg_precursor_mz,
        charge if charge is not None else np.nan,
        mz,
        intensity,
        avg_rt,
        cluster,
    )


@nb.njit(cache=True)
def typed_list_to_numpy(lst: nb.typed.List) -> np.ndarray:
    """
    Copy a Numba typed list of float32 values into a 1D NumPy array.

    Parameters
    ----------
    lst : nb.typed.List
        Numba typed list of float32 scalars.

    Returns
    -------
    np.ndarray
        Contiguous 1D ``np.float32`` array with the same elements.
    """
    n = len(lst)
    arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        arr[i] = lst[i]
    return arr
