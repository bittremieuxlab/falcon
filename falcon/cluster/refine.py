"""1D complete-linkage refinement on precursor m/z and RT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import scipy.cluster.hierarchy as sch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import ClusteringConfig

__all__ = [
    "cut_linkage",
    "iter_cluster_groups",
    "linkage_1d",
    "postprocess_cluster",
    "refine_clusters",
]


def cut_linkage(linkage_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Cut a linkage tree into 0-based int32 cluster labels."""
    return (sch.fcluster(linkage_matrix, threshold, "distance") - 1).astype(
        np.int32
    )


@nb.njit(cache=True)
def iter_cluster_groups(clusters: np.ndarray) -> Iterator[tuple[int, int]]:
    """
    Yield ``(start, stop)`` index slices for contiguous cluster labels.

    Each ``-1`` noise entry is yielded as its own length-1 slice.
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


@nb.njit(cache=True)
def _linkage_1d_nb(values: np.ndarray, use_ppm: bool) -> np.ndarray:
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

    Exploits 1D ordering so only adjacent clusters can merge, producing
    a SciPy-compatible float64 ``(n - 1, 4)`` linkage matrix. When
    ``tol_mode == "ppm"``, spans normalize by the lower cluster minimum.
    """
    return _linkage_1d_nb(
        np.asarray(values, dtype=np.float64), tol_mode == "ppm"
    )


def _split_group_by_rt(group_rts: np.ndarray, rt_tol: float) -> np.ndarray:
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

    Splits by full precursor m/z span via 1D complete linkage, then
    optionally refines each m/z sub-cluster independently by RT. Writes
    contiguous labels starting at ``start_label`` in place and returns
    the number of resulting clusters (including singletons).
    """
    n_points = cluster_labels.shape[0]
    if n_points <= 1 or n_points < min_samples:
        cluster_labels[:] = np.arange(
            start_label, start_label + n_points, dtype=np.int32
        )
        return n_points

    assignments = cut_linkage(
        linkage_1d(cluster_mzs, precursor_tol_mode), precursor_tol_mass
    )
    if rt_tol is not None:
        assignments = _refine_assignments_by_rt(
            assignments, cluster_rts, rt_tol, max(2, min_samples)
        )

    counts = np.bincount(assignments)
    keep_mask = counts >= min_samples
    n_clusters = int(np.count_nonzero(keep_mask))
    remap = np.full(len(counts), -1, dtype=np.int32)
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

    Returns ``(refined_labels, label_order, n_clusters)`` where
    ``refined_labels[k]`` is the int32 label for canonical index
    ``label_order[k]``.
    """
    label_order = np.argsort(labels, kind="stable")
    refined_labels = np.array(labels[label_order], dtype=np.int32, copy=True)
    mzs_ord = precursor_mzs[label_order]
    rts_ord = rts[label_order]

    current_label = 0
    for start_i, stop_i in iter_cluster_groups(refined_labels):
        current_label += postprocess_cluster(
            refined_labels[start_i:stop_i],
            mzs_ord[start_i:stop_i],
            rts_ord[start_i:stop_i],
            clustering.precursor_tol.mass,
            clustering.precursor_tol.mode,
            clustering.rt_tol,
            min_samples,
            current_label,
        )
    return refined_labels, label_order, current_label
