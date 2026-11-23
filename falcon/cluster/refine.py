"""1D complete-linkage refinement on precursor m/z and RT."""

from __future__ import annotations

import numba as nb
import numpy as np
import scipy.cluster.hierarchy as sch

__all__ = ["cut_linkage", "linkage_1d", "postprocess_cluster"]


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


def _assign_singletons(cluster_labels: np.ndarray, start_label: int) -> int:
    """Fill ``cluster_labels`` with contiguous singleton IDs."""
    n_points = cluster_labels.shape[0]
    cluster_labels[:] = np.arange(
        start_label, start_label + n_points, dtype=np.int32
    )
    return n_points


@nb.njit(cache=True, fastmath=True)
def linkage_1d(values: np.ndarray, tol_mode: str | None = None) -> np.ndarray:
    """
    Compute 1D complete-linkage hierarchical clustering over values.

    Exploits 1D ordering so only adjacent clusters can merge, avoiding a
    quadratic pairwise distance matrix. Output follows SciPy's linkage
    matrix format.

    Parameters
    ----------
    values : np.ndarray
        1D array of precursor m/z values or retention times.
    tol_mode : {"Da", "ppm"} | None, default=None
        Distance mode: ``"ppm"`` normalizes span by the lower cluster
        minimum; ``"Da"`` or ``None`` uses absolute differences.

    Returns
    -------
    np.ndarray
        Float64 ``(n - 1, 4)`` SciPy-compatible linkage matrix.
    """
    n = values.shape[0]
    linkage = np.zeros((n - 1, 4), np.double)
    # Track (min_val, max_val, cluster_id, size) in ascending order.
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(n - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]
            if tol_mode == "ppm":
                dist = dist / clusters[i][0] * 1e6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [
            clusters[min_i][2],
            clusters[min_i + 1][2],
            min_dist,
            n_points,
        ]
        clusters[min_i] = (
            clusters[min_i][0],
            clusters[min_i + 1][1],
            n + it,
            n_points,
        )
        del clusters[min_i + 1]
    return linkage


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

    Writes contiguous output labels starting at ``start_label`` into
    ``cluster_labels`` in place; groups smaller than ``min_samples``
    receive unique singleton labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Int32 slice mutated in place with refined cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z values of spectra in the initial cluster.
    cluster_rts : np.ndarray
        Retention times aligned with ``cluster_mzs``. NaN entries skip
        RT linkage and stay separate from known-RT spectra.
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
    if n_points < min_samples:
        return _assign_singletons(cluster_labels, start_label)

    # Split on full precursor m/z span via 1D complete linkage.
    mz_linkage = linkage_1d(cluster_mzs, precursor_tol_mode)
    assignments = cut_linkage(mz_linkage, precursor_tol_mass)

    # Optionally intersect with an RT partition. Known-RT spectra
    # undergo 1D RT linkage; NaN-RT spectra receive a dedicated RT code
    # so they stay grouped by m/z without merging into known-RT groups.
    if rt_tol is not None:
        nan_mask = np.isnan(cluster_rts)
        if not nan_mask.all():
            valid_mask = ~nan_mask
            valid_rts = cluster_rts[valid_mask]
            rt_labels = np.zeros(n_points, dtype=np.int32)
            if len(valid_rts) >= 2:
                valid_rt_labels = cut_linkage(linkage_1d(valid_rts), rt_tol)
                rt_labels[valid_mask] = valid_rt_labels
                n_rt = int(valid_rt_labels.max()) + 1
            else:
                n_rt = 1
            rt_labels[nan_mask] = n_rt
            # Injective mixed-radix encoding of (m/z, RT) pairs.
            assignments = np.unique(
                assignments.astype(np.int64) * (n_rt + 1) + rt_labels,
                return_inverse=True,
            )[1].astype(np.int32)

    counts = np.bincount(assignments)
    n_groups = len(counts)
    if n_groups == 1:
        cluster_labels.fill(start_label)
        return 1

    if n_groups == n_points:
        return _assign_singletons(cluster_labels, start_label)

    remap = np.full(n_groups, -1, dtype=np.int32)
    n_clusters = 0
    for group_id, count in enumerate(counts):
        if count >= min_samples:
            remap[group_id] = start_label + n_clusters
            n_clusters += 1

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
