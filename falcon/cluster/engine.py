"""Hierarchical spectrum clustering engine and worker coordinator."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import fastcluster
import joblib
import numpy as np

from ..config.models import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    MedoidConsensusConfig,
)
from ..dataset import (
    CANONICAL_SORT_COLUMNS,
    INTERVAL_COLUMNS,
    bucket_label_from_uri,
)
from ..types import (
    ClusterResult,
    ConsensusTuple,
    SpectrumTuple,
    offset_cluster_result,
)
from ..utils import effective_n_workers, format_hms
from .consensus import get_representative_spectra, iter_cluster_groups
from .distance import (
    compute_condensed_distance_matrix,
    fill_condensed_rows,
    open_shared_condensed,
)
from .refine import cut_linkage, postprocess_cluster
from .scheduling import (
    CHUNKS_PER_WORKER,
    MIN_PAIRS_PER_TILE,
    ChunkTask,
    GiantInterval,
    TileTask,
    interval_row_bounds,
    prepare_bucket_chunks,
    tile_threshold,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import lance
    import pandas as pd

    from ..config.models import ClusteringConfig, ConsensusConfig
    from .scheduling import IntervalSpec

logger = logging.getLogger(__name__)

TaggedIntervalResult = tuple[int, ClusterResult]

# Minimum spectra a refined cluster needs to keep a shared label;
# smaller groups become individually labeled singletons.
_MIN_CLUSTER_SAMPLES = 2

# Run explicit gc.collect() only after intervals larger than this size;
# smaller intervals do not allocate enough to justify GC overhead.
_GC_COLLECT_MIN_SPECTRA = 2**11

__all__ = [
    "ClusteringOptions",
    "TaggedIntervalResult",
    "cluster_chunk",
    "cluster_mz_interval",
    "finalize_bucket",
    "generate_clusters",
    "generate_clusters_multi",
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


def _parallel_pool() -> joblib.Parallel:
    """Create the shared loky pool returning unordered task results."""
    return joblib.Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=0,
        return_as="generator_unordered",
    )


def interval_spectrum_tuples(
    spectra: pd.DataFrame,
    *,
    already_sorted: bool = False,
) -> tuple[list[SpectrumTuple], np.ndarray]:
    """
    Build canonically ordered ``SpectrumTuple`` records and RT array.

    Parameters
    ----------
    spectra : pd.DataFrame
        Interval spectrum table containing ``INTERVAL_COLUMNS``.
    already_sorted : bool, default=False
        Skip sorting when the caller guarantees rows are already in
        ``CANONICAL_SORT_COLUMNS`` order (e.g. via Lance ``take``).

    Returns
    -------
    tuple[list[SpectrumTuple], np.ndarray]
        Canonically ordered ``SpectrumTuple`` objects and matching
        retention time array.
    """
    if not already_sorted:
        spectra = spectra.sort_values(list(CANONICAL_SORT_COLUMNS))
    rts = spectra["retention_time"].to_numpy()
    precursor_mz = spectra["precursor_mz"].to_numpy()
    precursor_charge = spectra["precursor_charge"].to_numpy()
    mz = spectra["mz"].to_numpy()
    intensity = spectra["intensity"].to_numpy()
    spec_tuples = [
        SpectrumTuple(
            precursor_mz[i], precursor_charge[i], mz[i], intensity[i]
        )
        for i in range(len(precursor_mz))
    ]
    return spec_tuples, rts


def cluster_mz_interval(
    spectra: pd.DataFrame,
    precursor_mzs: np.ndarray,
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
        Pre-fetched interval spectra containing ``INTERVAL_COLUMNS``.
    precursor_mzs : np.ndarray
        Precursor m/z values aligned with canonical order.
    options : ClusteringOptions
        Bundled clustering, consensus, and representative options.
    pdist : np.ndarray | None, default=None
        Optional precomputed condensed distance matrix (tiled giant
        path) indexed identically to ``interval_spectrum_tuples``.
    already_sorted : bool, default=False
        Whether ``spectra`` is already in canonical order.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical input order and
        representative spectra ordered by ascending cluster ID (or an
        empty list when ``options.compute_representatives`` is False).
    """
    spec_tuples, rts_orig = interval_spectrum_tuples(
        spectra, already_sorted=already_sorted
    )
    n_spectra = len(spec_tuples)
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
    if pdist is None:
        pdist = compute_condensed_distance_matrix(
            spec_tuples,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
        )
    labels = cut_linkage(
        fastcluster.linkage(pdist, clustering.linkage),
        clustering.distance_threshold,
    )

    # Group initial labels for precursor m/z and optional RT refinement.
    label_order = np.argsort(labels)
    mzs_ord = precursor_mzs[label_order]
    rts_ord = rts_orig[label_order]
    labels = labels[label_order]

    current_label = 0
    for start_i, stop_i in iter_cluster_groups(labels):
        n_clusters = postprocess_cluster(
            labels[start_i:stop_i],
            mzs_ord[start_i:stop_i],
            rts_ord[start_i:stop_i],
            clustering.precursor_tol.mass,
            clustering.precursor_tol.mode,
            clustering.rt_tol,
            _MIN_CLUSTER_SAMPLES,
            current_label,
        )
        current_label += n_clusters

    cluster_labels = np.empty(n_spectra, dtype=np.int32)
    if not options.compute_representatives:
        cluster_labels[label_order] = labels
        rep_spectra: list[ConsensusTuple] = []
    elif current_label < n_spectra:
        refined_order = np.argsort(labels)
        labels = labels[refined_order]
        rts_ref = rts_ord[refined_order]
        order_map = label_order[refined_order]
        rep_spectra = get_representative_spectra(
            spec_tuples,
            labels,
            rts_ref,
            order_map,
            options.consensus,
            pdist=pdist,
            min_mz=options.min_mz,
            max_mz=options.max_mz,
            fragment_tol=clustering.fragment_tol,
        )
        cluster_labels[order_map] = labels
    else:
        # Every spectrum ended up as its own singleton; labels in
        # label_order are already 0..n_spectra-1, so iterating pos emits
        # representatives strictly ordered by ascending cluster_id.
        cluster_labels[label_order] = labels
        rep_spectra = [
            ConsensusTuple.from_spectrum(
                spec_tuples[int(label_order[pos])],
                float(rts_ord[pos]),
                int(labels[pos]),
            )
            for pos in range(n_spectra)
        ]

    # Release memmap backing storage eagerly for large intervals.
    del pdist
    if n_spectra > _GC_COLLECT_MIN_SPECTRA:
        gc.collect()
    return ClusterResult(cluster_labels, rep_spectra)


def cluster_chunk(
    chunk: Sequence[IntervalSpec],
    dataset: lance.LanceDataset,
    options: ClusteringOptions,
    *,
    read_batch_size: int | None = None,
) -> list[TaggedIntervalResult]:
    """
    Cluster all intervals in a worker chunk with batched Lance reads.

    Parameters
    ----------
    chunk : Sequence[IntervalSpec]
        Intervals assigned to this worker chunk.
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    options : ClusteringOptions
        Bundled clustering, consensus, and representative options.
    read_batch_size : int | None, default=None
        Cap on spectra fetched per batched Lance ``take`` call. Reads
        the entire chunk at once when ``None``.

    Returns
    -------
    list[TaggedIntervalResult]
        List of ``(task_id, ClusterResult)`` interval results.
    """
    if not chunk:
        return []
    columns = list(INTERVAL_COLUMNS)

    def _cluster_group(
        group: list[IntervalSpec],
    ) -> list[TaggedIntervalResult]:
        # Batch Lance reads across small intervals; `take` preserves
        # row_ids order (already canonical) so slices stay contiguous.
        group_row_ids = [rid for spec in group for rid in spec.row_ids]
        group_spectra = dataset.take(
            indices=group_row_ids, columns=columns
        ).to_pandas()
        out: list[TaggedIntervalResult] = []
        offset = 0
        for spec in group:
            n_rows = spec.n_spectra
            interval_spectra = group_spectra.iloc[offset : offset + n_rows]
            offset += n_rows
            out.append(
                (
                    spec.task_id,
                    cluster_mz_interval(
                        interval_spectra,
                        spec.precursor_mzs,
                        options,
                        already_sorted=True,
                    ),
                )
            )
        return out

    results: list[TaggedIntervalResult] = []
    group: list[IntervalSpec] = []
    group_size = 0
    for spec in chunk:
        m = spec.n_spectra
        if (
            group
            and read_batch_size is not None
            and group_size + m > read_batch_size
        ):
            results.extend(_cluster_group(group))
            group, group_size = [], 0
        group.append(spec)
        group_size += m
    results.extend(_cluster_group(group))
    return results


def _cluster_chunk_tagged(
    task: ChunkTask,
    options: ClusteringOptions,
) -> tuple[str, int, int, int, list[TaggedIntervalResult]]:
    """Execute one ``ChunkTask`` and tag results for Pass 1."""
    chunk_results = cluster_chunk(
        task.intervals,
        dataset=task.dataset,
        options=options,
        read_batch_size=options.clustering.batch_size,
    )
    return ("chunk", task.bucket_id, task.cost, task.n_spectra, chunk_results)


def _tile_distance(
    tile: TileTask,
    fragment_tol: float,
    min_matched_peaks: int,
) -> tuple[str, int, float]:
    """Compute rows ``[tile.r0, tile.r1)`` of a giant memmap."""
    spectra = tile.dataset.take(
        indices=tile.row_ids, columns=list(INTERVAL_COLUMNS)
    )
    spec_tuples, _ = interval_spectrum_tuples(
        spectra.to_pandas(), already_sorted=True
    )
    condensed = np.lib.format.open_memmap(tile.memmap_path, mode="r+")
    fill_condensed_rows(
        condensed,
        spec_tuples,
        tile.r0,
        tile.r1,
        fragment_tol,
        min_matched_peaks,
    )
    condensed.flush()
    return ("tile", tile.giant_idx, tile.tile_cost)


def _finalize_giant(
    giant: GiantInterval,
    options: ClusteringOptions,
) -> tuple[int, int, ClusterResult]:
    """Cluster a giant interval from its precomputed distance memmap."""
    spectra = giant.dataset.take(
        indices=giant.spec.row_ids, columns=list(INTERVAL_COLUMNS)
    )
    pdist = np.lib.format.open_memmap(giant.memmap_path, mode="r")
    result = cluster_mz_interval(
        spectra.to_pandas(),
        giant.spec.precursor_mzs,
        options,
        pdist=pdist,
        already_sorted=True,
    )
    return (giant.bucket_id, giant.spec.task_id, result)


def _allocate_giant_tiles(
    raw_giants: Sequence[tuple[int, lance.LanceDataset, IntervalSpec]],
    n_workers: int,
    memmap_paths: list[str],
) -> tuple[list[GiantInterval], list[TileTask]]:
    """Allocate shared condensed memmaps and balanced row-band tiles."""
    giants: list[GiantInterval] = []
    tile_tasks: list[TileTask] = []
    for gi, (bucket_id, dataset, spec) in enumerate(raw_giants):
        n = spec.n_spectra
        path, mm = open_shared_condensed(n)
        del mm  # Workers reopen by path; release writer handle here.
        memmap_paths.append(path)
        giants.append(
            GiantInterval(
                bucket_id=bucket_id,
                dataset=dataset,
                spec=spec,
                memmap_path=path,
            )
        )
        total_pairs = n * (n - 1) // 2
        n_tiles = min(n_workers, max(1, total_pairs // MIN_PAIRS_PER_TILE))
        for r0, r1 in interval_row_bounds(n, n_tiles):
            tile_pairs = (2 * n - 1 - r0 - r1) * (r1 - r0) / 2
            tile_cost = spec.cost * tile_pairs / total_pairs
            tile_tasks.append(
                TileTask(
                    giant_idx=gi,
                    dataset=dataset,
                    row_ids=spec.row_ids,
                    r0=r0,
                    r1=r1,
                    memmap_path=path,
                    tile_cost=tile_cost,
                )
            )
    return giants, tile_tasks


def finalize_bucket(
    n: int,
    splits: Sequence[int],
    interval_results: Sequence[TaggedIntervalResult],
    bucket_label: str,
) -> ClusterResult:
    """
    Assemble per-split results into bucket-wide labels and consensus.

    Computes cumulative label offsets in a single sorted pass over split
    results so labels and representatives are unique across the bucket.

    Parameters
    ----------
    n : int
        Total number of spectra in the bucket.
    splits : Sequence[int]
        Interval split boundaries from ``prepare_bucket_chunks``.
    interval_results : Sequence[TaggedIntervalResult]
        Unordered ``(task_id, ClusterResult)`` split results.
    bucket_label : str
        Human-readable charge label for summary logging.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical order and
        representative spectra with bucket-unique int32 cluster IDs.
    """
    cluster_labels = np.full(n, -1, dtype=np.int32)
    representative_spectra: list[ConsensusTuple] = []
    current_offset = 0

    for task_id, interval_result in sorted(
        interval_results, key=lambda item: item[0]
    ):
        start, stop = int(splits[task_id]), int(splits[task_id + 1])
        shifted, current_offset = offset_cluster_result(
            interval_result, current_offset
        )
        cluster_labels[start:stop] = shifted.labels
        if shifted.representatives:
            representative_spectra.extend(shifted.representatives)

    _, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = int(np.count_nonzero(counts > 1))
    n_singletons = int(np.count_nonzero(counts == 1))
    logger.info(
        "Charge %s: %d spectra grouped in %d clusters, %d spectra remain "
        "as singletons",
        bucket_label,
        int(counts[counts > 1].sum()),
        n_clusters,
        n_singletons,
    )
    return ClusterResult(cluster_labels, representative_spectra)


def _run_pass1(
    chunk_tasks: Sequence[ChunkTask],
    tile_tasks: Sequence[TileTask],
    options: ClusteringOptions,
    *,
    n_buckets: int,
    n_workers: int,
    total_spectra: int,
) -> dict[int, list[TaggedIntervalResult]]:
    """Run Pass 1 (giant tiles and normal chunks) with progress logs."""
    results_by_bucket: dict[int, list[TaggedIntervalResult]] = {}
    n_pass1 = len(chunk_tasks) + len(tile_tasks)
    if n_pass1 == 0:
        return results_by_bucket

    n_giants = len({tile.giant_idx for tile in tile_tasks})
    total_cost = sum(t.cost for t in chunk_tasks) + sum(
        t.tile_cost for t in tile_tasks
    )
    logger.info(
        "Clustering %d spectra across %d charge bucket(s) in %d task(s) "
        "(%d giant interval(s) tiled) on %d worker(s)...",
        total_spectra,
        n_buckets,
        n_pass1,
        n_giants,
        n_workers,
    )
    run_chunk = partial(_cluster_chunk_tagged, options=options)
    clustering = options.clustering

    def _jobs() -> Iterator[Any]:
        for tile in tile_tasks:
            yield joblib.delayed(_tile_distance)(
                tile, clustering.fragment_tol, clustering.min_matched_peaks
            )
        for task in chunk_tasks:
            yield joblib.delayed(run_chunk)(task)

    done_cost = 0.0
    done_spectra = 0
    next_log_pct = 5
    start = time.time()

    for done_tasks, result in enumerate(_parallel_pool()(_jobs()), start=1):
        if result[0] == "chunk":
            _, bucket_id, cost, n_chunk, chunk_results = result
            results_by_bucket.setdefault(bucket_id, []).extend(chunk_results)
            done_cost += cost
            done_spectra += n_chunk
        else:
            done_cost += result[2]
        pct = (
            min(100.0, 100.0 * done_cost / total_cost) if total_cost else 100.0
        )
        if pct >= next_log_pct or done_tasks == n_pass1:
            elapsed = time.time() - start
            remaining_cost = max(0.0, total_cost - done_cost)
            eta = elapsed * remaining_cost / done_cost if done_cost else 0.0
            logger.info(
                "Clustering progress: %3.0f%% | %d/%d tasks | "
                "%d/%d spectra processed | elapsed %s | ETA ~%s",
                pct,
                done_tasks,
                n_pass1,
                done_spectra,
                total_spectra,
                format_hms(elapsed),
                format_hms(eta),
            )
            next_log_pct = (int(pct) // 5 + 1) * 5

    return results_by_bucket


def generate_clusters_multi(
    bucket_datasets: Sequence[lance.LanceDataset],
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = DEFAULT_MIN_MZ,
    max_mz: float = DEFAULT_MAX_MZ,
    bucket_labels: Sequence[str] | None = None,
    compute_representatives: bool = True,
) -> list[ClusterResult]:
    """
    Cluster multiple per-charge datasets in one shared worker pool.

    All buckets' m/z-split tasks share a single ``joblib`` pool so idle
    workers steal work across charge buckets rather than waiting at
    per-bucket barriers.

    Parameters
    ----------
    bucket_datasets : Sequence[lance.LanceDataset]
        Per-charge Lance datasets to cluster.
    clustering : ClusteringConfig
        Validated clustering configuration.
    consensus : ConsensusConfig | None, default=None
        Validated consensus configuration (defaults to medoid).
    min_mz, max_mz : float
        Configured fragment m/z range for average consensus.
    bucket_labels : Sequence[str] | None, default=None
        Optional human-readable labels aligned with ``bucket_datasets``.
    compute_representatives : bool, default=True
        Whether to compute representative spectra for each cluster.

    Returns
    -------
    list[ClusterResult]
        Per-dataset ``ClusterResult(labels, representatives)`` results
        in the same order as ``bucket_datasets``.
    """
    options = ClusteringOptions(
        clustering=clustering,
        consensus=(
            MedoidConsensusConfig() if consensus is None else consensus
        ),
        min_mz=min_mz,
        max_mz=max_mz,
        compute_representatives=compute_representatives,
    )
    labels_list = (
        list(bucket_labels)
        if bucket_labels is not None
        else [
            bucket_label_from_uri(ds.uri, str(i))
            for i, ds in enumerate(bucket_datasets)
        ]
    )
    logger.debug(
        "Hierarchical clustering (linkage=%s, distance_threshold=%.4f, "
        "min_samples=%d)",
        clustering.linkage,
        clustering.distance_threshold,
        _MIN_CLUSTER_SAMPLES,
    )

    n_workers = effective_n_workers()
    num_chunks = CHUNKS_PER_WORKER * n_workers
    threshold = tile_threshold(clustering.batch_size)

    prepared: list[tuple[int, Sequence[int]]] = []
    chunk_tasks: list[ChunkTask] = []
    raw_giants: list[tuple[int, lance.LanceDataset, IntervalSpec]] = []

    for bucket_id, (dataset, label) in enumerate(
        zip(bucket_datasets, labels_list, strict=True)
    ):
        n, splits, normal_chunks, giant_specs = prepare_bucket_chunks(
            dataset, label, clustering, num_chunks, threshold
        )
        prepared.append((n, splits))
        for specs in normal_chunks:
            chunk_tasks.append(
                ChunkTask(
                    bucket_id=bucket_id,
                    cost=sum(spec.cost for spec in specs),
                    n_spectra=sum(spec.n_spectra for spec in specs),
                    dataset=dataset,
                    intervals=tuple(specs),
                )
            )
        for spec in giant_specs:
            raw_giants.append((bucket_id, dataset, spec))

    # Schedule heaviest chunks first so small buckets backfill cleanly.
    chunk_tasks.sort(key=lambda task: task.cost, reverse=True)

    memmap_paths: list[str] = []
    try:
        giants, tile_tasks = _allocate_giant_tiles(
            raw_giants, n_workers, memmap_paths
        )
        total_spectra = sum(n for n, _ in prepared)
        results_by_bucket = _run_pass1(
            chunk_tasks,
            tile_tasks,
            options,
            n_buckets=len(bucket_datasets),
            n_workers=n_workers,
            total_spectra=total_spectra,
        )

        if giants:
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_results = _parallel_pool()(
                joblib.delayed(_finalize_giant)(giant, options)
                for giant in giants
            )
            for bucket_id, task_id, result in finalize_results:
                results_by_bucket.setdefault(bucket_id, []).append(
                    (task_id, result)
                )
    finally:
        for path in memmap_paths:
            with contextlib.suppress(OSError):
                os.unlink(path)

    return [
        finalize_bucket(
            n,
            splits,
            results_by_bucket.get(bucket_id, []),
            labels_list[bucket_id],
        )
        for bucket_id, (n, splits) in enumerate(prepared)
    ]


def generate_clusters(
    dataset: lance.LanceDataset,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = DEFAULT_MIN_MZ,
    max_mz: float = DEFAULT_MAX_MZ,
    bucket_label: str | None = None,
    compute_representatives: bool = True,
) -> ClusterResult:
    """
    Cluster spectra in one Lance dataset via hierarchical clustering.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge dataset containing spectra to cluster.
    clustering : ClusteringConfig
        Validated clustering configuration.
    consensus : ConsensusConfig | None, default=None
        Validated consensus configuration (defaults to medoid).
    min_mz, max_mz : float
        Configured fragment m/z range for average consensus.
    bucket_label : str | None, default=None
        Optional charge label for logging.
    compute_representatives : bool, default=True
        Whether to compute representative spectra.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical metadata order and
        representative spectra (or empty list when skipped).
    """
    return generate_clusters_multi(
        [dataset],
        clustering,
        consensus,
        min_mz=min_mz,
        max_mz=max_mz,
        bucket_labels=[bucket_label] if bucket_label is not None else None,
        compute_representatives=compute_representatives,
    )[0]
