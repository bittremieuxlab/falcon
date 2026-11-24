"""Single-interval clustering, refinement, and consensus."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import fastcluster
import numpy as np

from ..config import MedoidConsensusConfig
from ..dataset import CANONICAL_SORT_COLUMNS
from ..types import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    ClusterResult,
    ConsensusTuple,
    SpectrumTuple,
)
from .consensus import get_representative_spectra
from .distance import (
    close_memmap,
    compute_condensed_distance_matrix,
    compute_condensed_distance_matrix_csr,
    extract_arrow_csr,
)
from .refine import cut_linkage, refine_clusters

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa

    from ..config import ClusteringConfig, ConsensusConfig

# Minimum spectra a refined cluster needs to keep a shared label;
# smaller groups become individually labeled singletons.
MIN_CLUSTER_SAMPLES = 2

__all__ = [
    "MIN_CLUSTER_SAMPLES",
    "ClusteringOptions",
    "cluster_arrow_interval",
    "cluster_mz_interval",
    "interval_spectrum_tuples",
]


@dataclass(frozen=True, slots=True)
class ClusteringOptions:
    """Bundled clustering, consensus, and m/z-range options."""

    clustering: ClusteringConfig
    consensus: ConsensusConfig = field(default_factory=MedoidConsensusConfig)
    min_mz: float = DEFAULT_MIN_MZ
    max_mz: float = DEFAULT_MAX_MZ
    compute_representatives: bool = True
    temp_dir: str | None = None


def _sorted_metadata(
    spectra: pd.DataFrame, already_sorted: bool
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return canonically sorted DataFrame plus precursor m/z and RT."""
    if not already_sorted:
        spectra = spectra.sort_values(
            list(CANONICAL_SORT_COLUMNS), kind="stable"
        )
    return (
        spectra,
        spectra["precursor_mz"].to_numpy(),
        spectra["retention_time"].to_numpy(),
    )


def interval_spectrum_tuples(
    spectra: pd.DataFrame,
    *,
    already_sorted: bool = False,
) -> tuple[list[SpectrumTuple], np.ndarray, np.ndarray]:
    """
    Build canonically ordered ``SpectrumTuple`` records and metadata.

    Parameters
    ----------
    spectra : pd.DataFrame
        Interval spectrum table containing ``SORTED_INTERVAL_COLUMNS``
        (plus ``"identifier"`` when ``already_sorted`` is ``False``).
    already_sorted : bool, default=False
        Skip sorting when the caller guarantees rows are already in
        ``CANONICAL_SORT_COLUMNS`` order (e.g. via Lance ``take``).

    Returns
    -------
    tuple[list[SpectrumTuple], np.ndarray, np.ndarray]
        ``(spec_tuples, precursor_mzs, rts)`` aligned to canonical
        order.
    """
    spectra, precursor_mz, rts = _sorted_metadata(spectra, already_sorted)
    precursor_charge = spectra["precursor_charge"].to_numpy()
    mz = spectra["mz"].to_numpy()
    intensity = spectra["intensity"].to_numpy()
    spec_tuples = [
        SpectrumTuple(
            precursor_mz[i], precursor_charge[i], mz[i], intensity[i]
        )
        for i in range(len(precursor_mz))
    ]
    return spec_tuples, precursor_mz, rts


def _arrow_spectrum_tuples(
    table: pa.Table,
    precursor_mzs: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
) -> list[SpectrumTuple]:
    """Build ``SpectrumTuple`` records backed by Arrow CSR slices."""
    precursor_charge = table["precursor_charge"].to_numpy(zero_copy_only=False)
    return [
        SpectrumTuple(
            precursor_mzs[i],
            precursor_charge[i],
            mz_flat[offsets[i] : offsets[i + 1]],
            intensity_flat[offsets[i] : offsets[i + 1]],
        )
        for i in range(len(precursor_mzs))
    ]


def _close_if_memmap(pdist: np.ndarray) -> bool:
    """Close ``pdist`` when it is a NumPy memmap and return True."""
    if isinstance(pdist, np.memmap):
        close_memmap(pdist)
        return True
    return False


def _compute_pdist(
    spec_tuples: Sequence[SpectrumTuple],
    csr: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    options: ClusteringOptions,
) -> np.ndarray:
    """Compute a condensed distance matrix from CSR or tuples."""
    clustering = options.clustering
    if csr is not None:
        mz_flat, intensity_flat, offsets = csr
        return compute_condensed_distance_matrix_csr(
            mz_flat,
            intensity_flat,
            offsets,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
            temp_dir=options.temp_dir,
        )
    return compute_condensed_distance_matrix(
        spec_tuples,
        clustering.fragment_tol,
        clustering.min_matched_peaks,
        temp_dir=options.temp_dir,
    )


def _build_representatives(
    spec_tuples: Sequence[SpectrumTuple],
    refined_labels: np.ndarray,
    label_order: np.ndarray,
    rts_orig: np.ndarray,
    n_clusters: int,
    options: ClusteringOptions,
    pdist: np.ndarray | None,
) -> list[ConsensusTuple]:
    """Build representative spectra ordered by ascending cluster ID."""
    if not options.compute_representatives:
        return []
    if n_clusters < len(label_order):
        refined_order = np.argsort(refined_labels, kind="stable")
        sorted_labels = refined_labels[refined_order]
        order_map = label_order[refined_order]
        return get_representative_spectra(
            spec_tuples,
            sorted_labels,
            rts_orig[order_map],
            order_map,
            options.consensus,
            pdist=pdist,
            min_mz=options.min_mz,
            max_mz=options.max_mz,
            fragment_tol=options.clustering.fragment_tol,
        )
    # Every spectrum became a singleton; refined_labels in label_order
    # are 0..n_spectra-1, so iterating pos emits ascending cluster_ids.
    return [
        ConsensusTuple.from_spectrum(
            spec_tuples[int(idx)],
            float(rts_orig[int(idx)]),
            int(refined_labels[pos]),
        )
        for pos, idx in enumerate(label_order)
    ]


def _cluster_interval_core(
    precursor_mzs: np.ndarray,
    rts_orig: np.ndarray,
    options: ClusteringOptions,
    *,
    spec_tuples: Sequence[SpectrumTuple] = (),
    csr: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pdist: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster one ordered interval from tuples, CSR, or pdist."""
    n_spectra = len(precursor_mzs)
    if pdist is not None:
        expected_pairs = n_spectra * (n_spectra - 1) // 2
        if pdist.ndim != 1 or pdist.shape[0] != expected_pairs:
            raise ValueError(
                f"pdist must be 1D of length {expected_pairs} for "
                f"{n_spectra} spectra, got shape {pdist.shape}"
            )

    if n_spectra == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), [])
    if n_spectra == 1:
        labels_single = np.zeros(1, dtype=np.int32)
        reps = (
            [
                ConsensusTuple.from_spectrum(
                    spec_tuples[0], float(rts_orig[0]), 0
                )
            ]
            if options.compute_representatives
            else []
        )
        return ClusterResult(labels_single, reps)

    clustering = options.clustering
    owns_pdist = pdist is None
    if pdist is None:
        pdist = _compute_pdist(spec_tuples, csr, options)

    needs_pdist_for_reps = (
        options.compute_representatives
        and options.consensus.method == "medoid"
    )
    try:
        linkage_mat = fastcluster.linkage(pdist, clustering.linkage)
        if owns_pdist and not needs_pdist_for_reps:
            was_memmap = _close_if_memmap(pdist)
            pdist = None
            if was_memmap:
                gc.collect()

        labels = cut_linkage(linkage_mat, clustering.distance_threshold)
        refined_labels, label_order, n_clusters = refine_clusters(
            labels,
            precursor_mzs,
            rts_orig,
            clustering,
            MIN_CLUSTER_SAMPLES,
        )

        cluster_labels = np.empty(n_spectra, dtype=np.int32)
        cluster_labels[label_order] = refined_labels
        rep_spectra = _build_representatives(
            spec_tuples,
            refined_labels,
            label_order,
            rts_orig,
            n_clusters,
            options,
            pdist,
        )
    finally:
        # Release owned distance storage eagerly on large intervals.
        if owns_pdist and pdist is not None:
            was_memmap = _close_if_memmap(pdist)
            pdist = None
            if was_memmap:
                gc.collect()
    return ClusterResult(cluster_labels, rep_spectra)


def cluster_mz_interval(
    spectra: pd.DataFrame,
    options: ClusteringOptions,
    *,
    pdist: np.ndarray | None = None,
    already_sorted: bool = False,
) -> ClusterResult:
    """
    Cluster spectra within one precursor m/z interval.

    Parameters
    ----------
    spectra : pd.DataFrame
        Pre-fetched interval spectra. When ``pdist`` is supplied and
        ``options.compute_representatives`` is ``False``, only
        ``precursor_mz`` and ``retention_time`` (plus ``identifier`` if
        ``already_sorted`` is ``False``) are required.
    options : ClusteringOptions
        Bundled clustering, consensus, and representative options.
    pdist : np.ndarray | None, default=None
        Optional precomputed condensed distance matrix (tiled giant
        path) indexed identically to canonical order.
    already_sorted : bool, default=False
        Whether ``spectra`` is already in canonical order.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical input order and
        representative spectra ordered by ascending cluster ID (or an
        empty list when ``options.compute_representatives`` is False).
    """
    needs_peaks = pdist is None or options.compute_representatives
    if needs_peaks:
        spec_tuples, precursor_mzs, rts_orig = interval_spectrum_tuples(
            spectra, already_sorted=already_sorted
        )
    else:
        _, precursor_mzs, rts_orig = _sorted_metadata(spectra, already_sorted)
        spec_tuples = []
    return _cluster_interval_core(
        precursor_mzs,
        rts_orig,
        options,
        spec_tuples=spec_tuples,
        pdist=pdist,
    )


def cluster_arrow_interval(
    table: pa.Table,
    options: ClusteringOptions,
    *,
    pdist: np.ndarray | None = None,
) -> ClusterResult:
    """
    Cluster one canonically ordered Arrow interval without pandas.

    Streams ``mz`` and ``intensity`` directly from PyArrow list buffers
    into CSR distance computation and only materializes
    ``SpectrumTuple`` views when ``options.compute_representatives`` is
    ``True``. When ``pdist`` is supplied and representatives are
    disabled, reads only ``precursor_mz`` and ``retention_time``.

    Parameters
    ----------
    table : pa.Table
        Canonically ordered Arrow table containing ``precursor_mz`` and
        ``retention_time`` (plus ``mz`` and ``intensity`` whenever
        ``pdist`` is ``None`` or ``options.compute_representatives`` is
        ``True``, and ``precursor_charge`` when
        ``options.compute_representatives`` is ``True``).
    options : ClusteringOptions
        Bundled clustering, consensus, and representative options.
    pdist : np.ndarray | None, default=None
        Optional precomputed condensed distance matrix (tiled giant
        path) indexed identically to canonical order.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical order and optional
        representative spectra ordered by ascending cluster ID.
    """
    precursor_mzs = table["precursor_mz"].to_numpy(zero_copy_only=False)
    rts_orig = table["retention_time"].to_numpy(zero_copy_only=False)
    n_spectra = len(precursor_mzs)
    needs_peaks = (pdist is None and n_spectra >= 2) or (
        options.compute_representatives and n_spectra >= 1
    )
    if needs_peaks:
        mz_flat, intensity_flat, offsets = extract_arrow_csr(table)
        csr: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            mz_flat,
            intensity_flat,
            offsets,
        )
        spec_tuples: Sequence[SpectrumTuple] = (
            _arrow_spectrum_tuples(
                table, precursor_mzs, mz_flat, intensity_flat, offsets
            )
            if options.compute_representatives
            else ()
        )
    else:
        csr = None
        spec_tuples = ()
    return _cluster_interval_core(
        precursor_mzs,
        rts_orig,
        options,
        spec_tuples=spec_tuples,
        csr=csr,
        pdist=pdist,
    )
