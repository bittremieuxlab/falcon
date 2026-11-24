"""1D complete-linkage refinement on precursor m/z and RT."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numba as nb
import numpy as np
import scipy.cluster.hierarchy as sch

from ..types import PrecursorTolMode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import ClusteringConfig

_VALID_TOL_MODES = frozenset({None, *get_args(PrecursorTolMode)})

__all__ = [
    "cut_linkage",
    "iter_cluster_groups",
    "linkage_1d",
    "postprocess_cluster",
    "refine_clusters",
]


def cut_linkage(linkage_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Cut a hierarchical linkage tree into 0-based int32 cluster labels.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        SciPy-compatible ``(n - 1, 4)`` hierarchical linkage matrix.
    threshold : float
        Maximum distance threshold for flat cluster formation.

    Returns
    -------
    np.ndarray
        Zero-based int32 flat cluster assignment array of length ``n``.
    """
    return (sch.fcluster(linkage_matrix, threshold, "distance") - 1).astype(
        np.int32
    )


@nb.njit(cache=True)
def iter_cluster_groups(clusters: np.ndarray) -> Iterator[tuple[int, int]]:
    """
    Yield ``(start, stop)`` index slices for contiguous cluster labels.

    Assumes ``clusters`` is grouped by label; each ``-1`` noise entry
    is yielded as its own length-1 slice regardless of position.

    Parameters
    ----------
    clusters : np.ndarray
        1D cluster label array (``-1`` denotes noise points).

    Yields
    ------
    tuple[int, int]
        Half-open ``(start_i, stop_i)`` slice bounds for each group.
    """
    n = clusters.shape[0]
    pos = 0
    while pos < n:
        label = clusters[pos]
        start_i = pos
        pos += 1
        if label != -1:
            while pos < n and clusters[pos] == label:
                pos += 1
        yield start_i, pos


def _assign_singletons(cluster_labels: np.ndarray, start_label: int) -> int:
    """Fill ``cluster_labels`` with contiguous singleton IDs."""
    n_points = cluster_labels.shape[0]
    cluster_labels[:] = np.arange(
        start_label, start_label + n_points, dtype=np.int32
    )
    return n_points


@nb.njit(cache=True)
def _linkage_1d_nb(values: np.ndarray, use_ppm: bool) -> np.ndarray:
    """Compute 1D complete-linkage tree without fastmath."""
    n = values.shape[0]
    linkage = np.zeros((n - 1, 4), np.double)
    # Track (min_val, max_val, cluster_id, size) in ascending order.
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(n - 1):
        min_i = 0
        min_dist = clusters[1][1] - clusters[0][0]
        if use_ppm:
            min_dist = min_dist / clusters[0][0] * 1e6
        for i in range(1, len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]
            if use_ppm:
                dist = dist / clusters[i][0] * 1e6
            if dist < min_dist or (np.isnan(min_dist) and not np.isnan(dist)):
                min_dist = dist
                min_i = i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, 0] = clusters[min_i][2]
        linkage[it, 1] = clusters[min_i + 1][2]
        linkage[it, 2] = min_dist
        linkage[it, 3] = n_points
        clusters[min_i] = (
            clusters[min_i][0],
            clusters[min_i + 1][1],
            n + it,
            n_points,
        )
        del clusters[min_i + 1]
    return linkage


def linkage_1d(values: np.ndarray, tol_mode: str | None = None) -> np.ndarray:
    """
    Compute 1D complete-linkage hierarchical clustering over values.

    Exploits 1D ordering so only adjacent clusters can merge, avoiding a
    quadratic pairwise distance matrix. Output follows SciPy's linkage
    matrix format.

    Parameters
    ----------
    values : np.ndarray
        1D array of at least 2 precursor m/z values or retention times.
    tol_mode : {"Da", "ppm"} | None, default=None
        Distance mode: ``"ppm"`` normalizes span by the lower cluster
        minimum (requires strictly positive values); ``"Da"`` or
        ``None`` uses absolute differences.

    Returns
    -------
    np.ndarray
        Float64 ``(n - 1, 4)`` SciPy-compatible linkage matrix.

    Raises
    ------
    ValueError
        If ``values`` is not 1D, has fewer than 2 entries, ``tol_mode``
        is unrecognized, or ``tol_mode == "ppm"`` with non-finite or
        non-positive values.
    """
    values_f64 = np.asarray(values, dtype=np.float64)
    if values_f64.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {values_f64.shape}")
    if values_f64.shape[0] < 2:
        raise ValueError(
            f"linkage_1d requires at least 2 values, got {values_f64.shape[0]}"
        )
    if tol_mode not in _VALID_TOL_MODES:
        raise ValueError(
            f"tol_mode must be None, 'Da', or 'ppm', got {tol_mode!r}"
        )
    use_ppm = tol_mode == "ppm"
    if use_ppm and not bool(
        np.all(np.isfinite(values_f64) & (values_f64 > 0.0))
    ):
        raise ValueError(
            "tol_mode='ppm' requires finite strictly positive values"
        )
    return _linkage_1d_nb(values_f64, use_ppm)


def _split_group_by_rt(group_rts: np.ndarray, rt_tol: float) -> np.ndarray:
    """Return 0-based RT sub-cluster IDs for one precursor m/z group."""
    valid_mask = np.isfinite(group_rts)
    n_valid = int(np.count_nonzero(valid_mask))
    n_group = group_rts.shape[0]
    if n_valid == 0:
        return np.zeros(n_group, dtype=np.int32)
    if n_valid == n_group:
        return cut_linkage(linkage_1d(group_rts), rt_tol)
    rt_labels = np.empty(n_group, dtype=np.int32)
    if n_valid >= 2:
        valid_labels = cut_linkage(linkage_1d(group_rts[valid_mask]), rt_tol)
        rt_labels[valid_mask] = valid_labels
        n_rt = int(valid_labels.max()) + 1
    else:
        rt_labels[valid_mask] = 0
        n_rt = 1
    rt_labels[~valid_mask] = n_rt
    return rt_labels


def _refine_assignments_by_rt(
    assignments: np.ndarray,
    cluster_rts: np.ndarray,
    rt_tol: float,
    min_split_size: int,
) -> np.ndarray:
    """Split each m/z group independently by 1D RT complete linkage."""
    counts = np.bincount(assignments)
    if len(counts) == 1:
        return _split_group_by_rt(cluster_rts, rt_tol)
    if int(np.max(counts)) < min_split_size:
        return assignments

    mz_order = np.argsort(assignments, kind="stable")
    sorted_mz = assignments[mz_order]
    refined = np.empty(assignments.shape[0], dtype=np.int32)
    next_id = 0
    for start_i, stop_i in iter_cluster_groups(sorted_mz):
        idx = mz_order[start_i:stop_i]
        if stop_i - start_i >= min_split_size:
            sub_labels = _split_group_by_rt(cluster_rts[idx], rt_tol)
            refined[idx] = sub_labels + next_id
            next_id += int(sub_labels.max()) + 1
        else:
            refined[idx] = next_id
            next_id += 1
    return refined


def postprocess_cluster(
    cluster_labels: np.ndarray,
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float | None,
    min_samples: int,
    start_label: int,
) -> int:
    """
    Refine an initial cluster by precursor m/z and optional RT span.

    Splits first by full precursor m/z span via 1D complete linkage,
    then optionally runs 1D RT complete linkage independently within
    each m/z sub-cluster. Writes contiguous output labels starting at
    ``start_label`` into ``cluster_labels`` in place; groups smaller
    than ``min_samples`` receive unique singleton labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Int32 slice mutated in place with refined cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z values of spectra in the initial cluster.
    cluster_rts : np.ndarray
        Retention times aligned with ``cluster_mzs``. Within each m/z
        sub-cluster, non-finite entries skip RT linkage and stay
        separate from finite-RT spectra.
    precursor_tol_mass : float
        Maximum precursor m/z span within a refined cluster.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    rt_tol : float | None
        Maximum RT span within a refined cluster, or ``None`` to skip
        RT splitting.
    min_samples : int
        Minimum spectra required to form a multi-spectrum cluster.
    start_label : int
        First cluster label to assign.

    Returns
    -------
    int
        Total number of output clusters (including singletons).
    """
    n_points = cluster_labels.shape[0]
    if n_points <= 1 or n_points < min_samples:
        return _assign_singletons(cluster_labels, start_label)

    # Split on full precursor m/z span via 1D complete linkage.
    mz_linkage = linkage_1d(cluster_mzs, precursor_tol_mode)
    assignments = cut_linkage(mz_linkage, precursor_tol_mass)

    # Optionally refine each m/z sub-cluster independently by 1D RT
    # complete linkage so interleaved RTs across distinct m/z groups
    # never split valid (m/z, RT) pairs.
    if rt_tol is not None:
        assignments = _refine_assignments_by_rt(
            assignments, cluster_rts, rt_tol, max(2, min_samples)
        )

    counts = np.bincount(assignments)
    n_groups = len(counts)
    if n_groups == 1:
        cluster_labels.fill(start_label)
        return 1

    if n_groups == n_points:
        return _assign_singletons(cluster_labels, start_label)

    keep_mask = counts >= min_samples
    n_clusters = int(np.count_nonzero(keep_mask))
    remap = np.full(n_groups, -1, dtype=np.int32)
    remap[keep_mask] = np.arange(
        start_label, start_label + n_clusters, dtype=np.int32
    )

    cluster_labels[:] = remap[assignments]
    singleton_mask = cluster_labels == -1
    n_singletons = int(np.count_nonzero(singleton_mask))
    if n_singletons:
        cluster_labels[singleton_mask] = np.arange(
            start_label + n_clusters,
            start_label + n_clusters + n_singletons,
            dtype=np.int32,
        )
        n_clusters += n_singletons
    return n_clusters


def refine_clusters(
    labels: np.ndarray,
    precursor_mzs: np.ndarray,
    rts: np.ndarray,
    clustering: ClusteringConfig,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Refine all initial clusters in an interval by precursor m/z and RT.

    Groups initial cluster labels using a stable sort so canonical
    within-cluster input order is preserved, then applies
    :func:`postprocess_cluster` to each initial cluster in turn.

    Parameters
    ----------
    labels : np.ndarray
        Initial 1D cluster labels aligned to canonical input order.
    precursor_mzs : np.ndarray
        Precursor m/z values aligned with ``labels``.
    rts : np.ndarray
        Retention times aligned with ``labels``.
    clustering : ClusteringConfig
        Validated clustering configuration providing precursor and RT
        tolerances.
    min_samples : int
        Minimum spectra required to form a multi-spectrum cluster.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(refined_labels, label_order, n_clusters)`` where
        ``refined_labels[k]`` is the refined int32 label for canonical
        input index ``label_order[k]``, and ``n_clusters`` is the total
        number of refined clusters (including singletons).

    Raises
    ------
    ValueError
        If ``labels``, ``precursor_mzs``, and ``rts`` are not 1D arrays
        of equal length.
    """
    if labels.ndim != 1 or precursor_mzs.ndim != 1 or rts.ndim != 1:
        raise ValueError("labels, precursor_mzs, and rts must be 1D arrays")
    if (
        precursor_mzs.shape[0] != labels.shape[0]
        or rts.shape[0] != labels.shape[0]
    ):
        raise ValueError(
            "labels, precursor_mzs, and rts must have matching lengths, "
            f"got {labels.shape[0]}, {precursor_mzs.shape[0]}, and "
            f"{rts.shape[0]}"
        )
    label_order = np.argsort(labels, kind="stable")
    refined_labels = np.array(labels[label_order], dtype=np.int32, copy=True)
    mzs_ord = precursor_mzs[label_order]
    rts_ord = rts[label_order]

    current_label = 0
    for start_i, stop_i in iter_cluster_groups(refined_labels):
        n_clusters = postprocess_cluster(
            refined_labels[start_i:stop_i],
            mzs_ord[start_i:stop_i],
            rts_ord[start_i:stop_i],
            clustering.precursor_tol.mass,
            clustering.precursor_tol.mode,
            clustering.rt_tol,
            min_samples,
            current_label,
        )
        current_label += n_clusters
    return refined_labels, label_order, current_label
