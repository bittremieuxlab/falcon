"""Hierarchical spectrum clustering engine and worker coordinator."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any

import fastcluster
import joblib
import lance
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from ..config.models import (
    ClusteringConfig,
    ConsensusConfig,
    MedoidConsensusConfig,
)
from ..staging import CANONICAL_SORT_COLUMNS, INTERVAL_COLUMNS
from ..types import ConsensusTuple, SpectrumTuple, make_consensus_tuple
from ..utils import format_hms
from .consensus import get_representative_spectra, iter_cluster_groups
from .distance import (
    compute_condensed_distance_matrix,
    fill_condensed_rows,
    open_shared_condensed,
)
from .refine import postprocess_cluster
from .scheduling import (
    CHUNKS_PER_WORKER,
    MIN_PAIRS_PER_TILE,
    ChunkTask,
    GiantInterval,
    IntervalSpec,
    TileTask,
    interval_row_bounds,
    prepare_bucket_chunks,
    tile_threshold,
)

logger = logging.getLogger("falcon")

IntervalResult = tuple[list[ConsensusTuple], np.ndarray]
TaggedIntervalResult = tuple[int, IntervalResult]
ClusterResult = tuple[np.ndarray, list[ConsensusTuple]]

# Minimum spectra a refined cluster needs to keep a shared label;
# smaller groups become individually labeled singletons.
_MIN_CLUSTER_SAMPLES = 2

# Run explicit gc.collect() only after intervals larger than this size;
# smaller intervals do not allocate enough to justify GC overhead.
_GC_COLLECT_MIN_SPECTRA = 2**11


def _default_bucket_label(dataset: lance.LanceDataset, index: int) -> str:
    """Fallback charge label derived from dataset URI when omitted."""
    uri = dataset.uri
    if "spectra_charge_" in uri:
        return (
            uri.split("spectra_charge_")[-1].split(".")[0].replace("_", ", ")
        )
    return str(index)


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
    clustering: ClusteringConfig,
    consensus: ConsensusConfig,
    *,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
    pdist: np.ndarray | None = None,
    compute_representatives: bool = True,
    already_sorted: bool = False,
) -> IntervalResult:
    """
    Cluster spectra within one precursor m/z interval.

    Parameters
    ----------
    spectra : pd.DataFrame
        Pre-fetched interval spectra containing ``INTERVAL_COLUMNS``.
    precursor_mzs : np.ndarray
        Precursor m/z values aligned with canonical order.
    clustering : ClusteringConfig
        Validated clustering parameters.
    consensus : ConsensusConfig
        Validated consensus parameters.
    min_mz : float, default=101.0
        Lower fragment m/z bound (used for average consensus).
    max_mz : float, default=1500.0
        Upper fragment m/z bound (used for average consensus).
    pdist : np.ndarray | None, default=None
        Optional precomputed condensed distance matrix (tiled giant
        path) indexed identically to ``interval_spectrum_tuples``.
    compute_representatives : bool, default=True
        Whether to compute representative spectra. When ``False``,
        returns an empty list and avoids representative work and the
        second label sort.
    already_sorted : bool, default=False
        Whether ``spectra`` is already in canonical order.

    Returns
    -------
    IntervalResult
        Representative spectra (or empty list) and int32 cluster labels
        aligned to the interval's canonical input order.
    """
    spec_tuples, rts_orig = interval_spectrum_tuples(
        spectra, already_sorted=already_sorted
    )
    n_spectra = len(spec_tuples)
    if n_spectra == 1:
        labels_single = np.zeros(1, dtype=np.int32)
        reps = (
            [
                make_consensus_tuple(
                    precursor_mz=spec_tuples[0].precursor_mz,
                    precursor_charge=spec_tuples[0].precursor_charge,
                    mz=spec_tuples[0].mz,
                    intensity=spec_tuples[0].intensity,
                    retention_time=float(rts_orig[0]),
                    cluster_id=0,
                )
            ]
            if compute_representatives
            else []
        )
        return reps, labels_single

    if pdist is None:
        pdist = compute_condensed_distance_matrix(
            spec_tuples,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
        )
    labels = (
        sch.fcluster(
            fastcluster.linkage(pdist, clustering.linkage),
            clustering.distance_threshold,
            "distance",
        )
        - 1
    ).astype(np.int32)

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
    if not compute_representatives:
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
            consensus,
            pdist=pdist,
            min_mz=min_mz,
            max_mz=max_mz,
            fragment_tol=clustering.fragment_tol,
        )
        cluster_labels[order_map] = labels
    else:
        # Every spectrum ended up as its own singleton.
        cluster_labels[label_order] = labels
        rep_spectra = [
            make_consensus_tuple(
                precursor_mz=spec.precursor_mz,
                precursor_charge=spec.precursor_charge,
                mz=spec.mz,
                intensity=spec.intensity,
                retention_time=float(rts_orig[i]),
                cluster_id=int(cluster_labels[i]),
            )
            for i, spec in enumerate(spec_tuples)
        ]

    # Release memmap backing storage eagerly for large intervals.
    del pdist
    if n_spectra > _GC_COLLECT_MIN_SPECTRA:
        gc.collect()
    return rep_spectra, cluster_labels


def cluster_chunk(
    chunk: Sequence[IntervalSpec],
    dataset: lance.LanceDataset,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig,
    *,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
    compute_representatives: bool = True,
    batch_size: int | None = None,
) -> list[TaggedIntervalResult]:
    """
    Cluster all intervals in a worker chunk with batched Lance reads.

    Parameters
    ----------
    chunk : Sequence[IntervalSpec]
        Intervals assigned to this worker chunk.
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    clustering : ClusteringConfig
        Validated clustering parameters.
    consensus : ConsensusConfig
        Validated consensus parameters.
    min_mz, max_mz : float
        Configured fragment m/z range for average consensus.
    compute_representatives : bool, default=True
        Whether to compute cluster representative spectra.
    batch_size : int | None, default=None
        Cap on spectra fetched per batched Lance ``take`` call. Reads
        the entire chunk at once when ``None``.

    Returns
    -------
    list[TaggedIntervalResult]
        List of ``(task_id, (rep_spectra, cluster_labels))`` results.
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
                        clustering,
                        consensus,
                        min_mz=min_mz,
                        max_mz=max_mz,
                        compute_representatives=compute_representatives,
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
        if group and batch_size is not None and group_size + m > batch_size:
            results.extend(_cluster_group(group))
            group, group_size = [], 0
        group.append(spec)
        group_size += m
    if group:
        results.extend(_cluster_group(group))
    return results


def _cluster_chunk_tagged(
    task: ChunkTask,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig,
    min_mz: float,
    max_mz: float,
    compute_representatives: bool,
) -> tuple[str, int, int, int, list[TaggedIntervalResult]]:
    """Execute one ``ChunkTask`` and tag results for Pass 1."""
    chunk_results = cluster_chunk(
        task.intervals,
        dataset=task.dataset,
        clustering=clustering,
        consensus=consensus,
        min_mz=min_mz,
        max_mz=max_mz,
        compute_representatives=compute_representatives,
        batch_size=clustering.batch_size,
    )
    return ("chunk", task.bucket_id, task.cost, task.n_spectra, chunk_results)


def _tile_distance(
    tile: TileTask,
    fragment_mz_tol: float,
    min_matches: int,
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
        condensed, spec_tuples, tile.r0, tile.r1, fragment_mz_tol, min_matches
    )
    condensed.flush()
    return ("tile", tile.giant_idx, tile.tile_cost)


def _finalize_giant(
    giant: GiantInterval,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig,
    min_mz: float,
    max_mz: float,
    compute_representatives: bool,
) -> tuple[int, int, IntervalResult]:
    """Cluster a giant interval from its precomputed distance memmap."""
    spectra = giant.dataset.take(
        indices=giant.spec.row_ids, columns=list(INTERVAL_COLUMNS)
    )
    pdist = np.lib.format.open_memmap(giant.memmap_path, mode="r")
    result = cluster_mz_interval(
        spectra.to_pandas(),
        giant.spec.precursor_mzs,
        clustering,
        consensus,
        min_mz=min_mz,
        max_mz=max_mz,
        pdist=pdist,
        compute_representatives=compute_representatives,
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
        mm.flush()
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
        Unordered ``(task_id, (rep_spectra, labels))`` split results.
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

    for task_id, (interval_reps, labels) in sorted(
        interval_results, key=lambda item: item[0]
    ):
        start, stop = int(splits[task_id]), int(splits[task_id + 1])
        offset_labels = np.asarray(labels, dtype=np.int32) + np.int32(
            current_offset
        )
        cluster_labels[start:stop] = offset_labels
        if interval_reps:
            representative_spectra.extend(
                rep._replace(
                    cluster_id=np.int32(int(rep.cluster_id) + current_offset)
                )
                for rep in interval_reps
            )
        if len(offset_labels):
            current_offset = int(np.max(offset_labels)) + 1

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
    return cluster_labels, representative_spectra


def _run_pass1(
    chunk_tasks: Sequence[ChunkTask],
    tile_tasks: Sequence[TileTask],
    n_buckets: int,
    n_giants: int,
    n_workers: int,
    total_spectra: int,
    total_cost: int,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig,
    min_mz: float,
    max_mz: float,
    compute_representatives: bool,
    results_by_bucket: dict[int, list[TaggedIntervalResult]],
) -> None:
    """Run Pass 1 (giant tiles and normal chunks) with progress logs."""
    n_pass1 = len(chunk_tasks) + len(tile_tasks)
    if n_pass1 == 0:
        return

    logger.info(
        "Clustering %d spectra across %d charge bucket(s) in %d task(s) "
        "(%d giant interval(s) tiled) on %d worker(s)...",
        total_spectra,
        n_buckets,
        n_pass1,
        n_giants,
        n_workers,
    )
    run_chunk = partial(
        _cluster_chunk_tagged,
        clustering=clustering,
        consensus=consensus,
        min_mz=min_mz,
        max_mz=max_mz,
        compute_representatives=compute_representatives,
    )

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

    results = joblib.Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=0,
        return_as="generator_unordered",
    )(_jobs())
    for done_tasks, result in enumerate(results, start=1):
        if result[0] == "chunk":
            _, bucket_id, cost, n_chunk, chunk_results = result
            results_by_bucket[bucket_id].extend(chunk_results)
            done_cost += cost
            done_spectra += n_chunk
        else:
            done_cost += result[2]
        pct = 100.0 * done_cost / total_cost if total_cost else 100.0
        if pct >= next_log_pct or done_tasks == n_pass1:
            elapsed = time.time() - start
            eta = (
                elapsed * (total_cost - done_cost) / done_cost
                if done_cost
                else 0.0
            )
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


def generate_clusters_multi(
    bucket_datasets: Sequence[lance.LanceDataset],
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
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
        Per-dataset ``(labels, representative_spectra)`` results in the
        same order as ``bucket_datasets``.
    """
    effective_consensus: ConsensusConfig = (
        MedoidConsensusConfig() if consensus is None else consensus
    )
    labels_list = (
        list(bucket_labels)
        if bucket_labels is not None
        else [
            _default_bucket_label(ds, i)
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

    n_workers = max(1, int(joblib.effective_n_jobs(-1)))
    num_chunks = CHUNKS_PER_WORKER * n_workers
    threshold = tile_threshold(clustering.batch_size)

    prepared: list[tuple[int, Sequence[int]]] = []
    chunk_tasks: list[ChunkTask] = []
    raw_giants: list[tuple[int, lance.LanceDataset, IntervalSpec]] = []

    for bucket_id, (dataset, label) in enumerate(
        zip(bucket_datasets, labels_list, strict=True)
    ):
        n, splits, normal_chunks, giant_specs = prepare_bucket_chunks(
            dataset,
            label,
            clustering.precursor_tol.mass,
            clustering.precursor_tol.mode,
            clustering.batch_size,
            num_chunks,
            threshold,
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

    results_by_bucket: dict[int, list[TaggedIntervalResult]] = defaultdict(
        list
    )
    memmap_paths: list[str] = []
    try:
        giants, tile_tasks = _allocate_giant_tiles(
            raw_giants, n_workers, memmap_paths
        )
        total_cost = sum(t.cost for t in chunk_tasks) + sum(
            g.spec.cost for g in giants
        )
        total_spectra = sum(n for n, _ in prepared)

        _run_pass1(
            chunk_tasks=chunk_tasks,
            tile_tasks=tile_tasks,
            n_buckets=len(bucket_datasets),
            n_giants=len(giants),
            n_workers=n_workers,
            total_spectra=total_spectra,
            total_cost=total_cost,
            clustering=clustering,
            consensus=effective_consensus,
            min_mz=min_mz,
            max_mz=max_mz,
            compute_representatives=compute_representatives,
            results_by_bucket=results_by_bucket,
        )

        if giants:
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_results = joblib.Parallel(
                n_jobs=-1,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(
                joblib.delayed(_finalize_giant)(
                    giant,
                    clustering,
                    effective_consensus,
                    min_mz,
                    max_mz,
                    compute_representatives,
                )
                for giant in giants
            )
            for bucket_id, task_id, result in finalize_results:
                results_by_bucket[bucket_id].append((task_id, result))
    finally:
        for path in memmap_paths:
            with contextlib.suppress(OSError):
                os.unlink(path)

    return [
        finalize_bucket(
            n, splits, results_by_bucket[bucket_id], labels_list[bucket_id]
        )
        for bucket_id, (n, splits) in enumerate(prepared)
    ]


def generate_clusters(
    dataset: lance.LanceDataset,
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
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
