"""Hierarchical spectrum clustering, refinement, and scheduling."""

import contextlib
import gc
import logging
import math
import multiprocessing
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any

import fastcluster
import joblib
import lance
import numba as nb
import numba.typed
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import spectrum_utils.utils as suu
from scipy.cluster.hierarchy import fcluster

from . import similarity
from .consensus import (
    ConsensusTuple,
    _get_cluster_group_idx,
    _get_representative_spectra,
)
from .distance_matrix import (
    _condensed_rows,
    compute_condensed_distance_matrix,
    open_shared_condensed,
)

logger = logging.getLogger("falcon")

IntervalTask = tuple[int, list[int], np.ndarray, np.ndarray]
GiantIntervalTask = tuple[int, list[int], np.ndarray, np.ndarray, int]
IntervalResult = tuple[list[ConsensusTuple], np.ndarray]
TaggedIntervalResult = tuple[int, IntervalResult]
TileTask = tuple[int, list[int], int, int, str, float]

# Number of cost-balanced m/z-split chunks to create per CPU worker.
# Having several times more tasks than workers lets joblib load-balance
# dynamically instead of stranding cores on the tail of a fixed split.
_CHUNKS_PER_WORKER = 8

# Columns fetched per interval from a staged Lance dataset.
_INTERVAL_COLUMNS = [
    "identifier",
    "precursor_mz",
    "precursor_charge",
    "retention_time",
    "mz",
    "intensity",
]

# Minimum pairs per distance tile. Giant intervals split into at most
# cpu_count tiles, never finer than this pair budget so each tile
# amortizes its Lance read over enough cosine distance work.
_MIN_PAIRS_PER_TILE = 4_000_000

# Minimum spectra a refined cluster needs to keep a shared label;
# smaller groups become individually labeled singletons.
_MIN_CLUSTER_SAMPLES = 2

# Run explicit gc.collect() only after intervals larger than this size;
# smaller intervals do not allocate enough to justify GC overhead.
_GC_COLLECT_MIN_SPECTRA = 2**11


def _tile_threshold(batch_size: int) -> int:
    """
    Return the minimum interval size that triggers tiled distance work.

    Intervals at least this large would otherwise grind in one worker
    and strand cores at the tail of a run. Tracking ``batch_size // 2``
    self-scales with ``batch_size`` and catches forced-split giants in
    ``[batch_size / 2, batch_size]``.

    Parameters
    ----------
    batch_size : int
        Configured maximum interval size.

    Returns
    -------
    int
        Minimum spectrum count for tiling a distance matrix build.
    """
    return max(2, batch_size // 2)


def _format_hms(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS``."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _charge_label(dataset: lance.LanceDataset) -> str:
    """
    Derive a readable charge label from a per-charge dataset URI.

    For example, ``spectra_charge_2.lance`` becomes ``"2"``,
    ``spectra_charge_1_2.lance`` becomes ``"1, 2"``, and
    ``spectra_charge_unknown.lance`` becomes ``"unknown"``.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge Lance dataset.

    Returns
    -------
    str
        Human-readable charge label for logging.
    """
    return (
        dataset.uri.split("spectra_charge_")[-1]
        .split(".")[0]
        .replace("_", ", ")
    )


def generate_clusters(
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float | None,
    fragment_tol: float,
    batch_size: int,
    consensus_method: str,
    consensus_params: dict[str, Any],
) -> tuple[np.ndarray, list[ConsensusTuple]]:
    """
    Cluster spectra in one Lance dataset via hierarchical clustering.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge dataset containing spectra to cluster.
    linkage : {"single", "complete", "average"}
        Hierarchical clustering linkage criterion.
    distance_threshold : float
        Cosine distance threshold at which flat clusters are cut.
    min_matches : int
        Minimum matched fragment peaks required for nonzero similarity.
    precursor_tol_mass : float
        Maximum precursor m/z tolerance within a cluster.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    rt_tol : float | None
        Maximum retention time tolerance within a cluster. NaN RTs are
        grouped separately from known RTs and are not split by RT.
        Disabled when ``None``.
    fragment_tol : float
        Fragment m/z tolerance (in Da).
    batch_size : int
        Maximum number of spectra per precursor m/z interval.
    consensus_method : {"medoid", "average"}
        Representative spectrum computation method.
    consensus_params : dict[str, Any]
        Extra keyword arguments for the consensus routine.

    Returns
    -------
    tuple[np.ndarray, list[ConsensusTuple]]
        Int32 cluster labels aligned to canonical metadata order
        ``(precursor_mz, retention_time, identifier)`` and one
        representative spectrum per cluster.
    """
    # Convenience wrapper delegating to the shared-pool implementation.
    return generate_clusters_multi(
        [dataset],
        linkage,
        distance_threshold,
        min_matches,
        precursor_tol_mass,
        precursor_tol_mode,
        rt_tol,
        fragment_tol,
        batch_size,
        consensus_method,
        consensus_params,
    )[0]


def _build_giant_tile_tasks(
    giants: list[dict[str, Any]],
    n_workers: int,
    memmap_paths: list[str],
) -> list[TileTask]:
    """
    Allocate shared distance memmaps and row-band tiles for giants.

    Records each backing path on ``g["path"]`` and appends it to
    ``memmap_paths`` immediately so caller ``finally`` cleanup removes
    every created file even on partial failure.

    Parameters
    ----------
    giants : list[dict[str, Any]]
        Giant interval descriptors (with ``n``, ``cost``, ``row_ids``),
        mutated in place to store ``"path"``.
    n_workers : int
        Worker count capping tiles per giant interval.
    memmap_paths : list[str]
        Output list accumulating allocated memmap paths for cleanup.

    Returns
    -------
    list[TileTask]
        Tile descriptors ``(giant_idx, row_ids, r0, r1, path,
        tile_cost)`` filling disjoint upper-triangle row bands.
    """
    tile_tasks: list[TileTask] = []
    for gi, g in enumerate(giants):
        n = g["n"]
        path, mm = open_shared_condensed(n)
        mm.flush()
        del mm  # Workers reopen by path; release writer handle here.
        memmap_paths.append(path)
        g["path"] = path
        total_pairs = n * (n - 1) // 2
        n_tiles = min(n_workers, max(1, total_pairs // _MIN_PAIRS_PER_TILE))
        for r0, r1 in _interval_row_bounds(n, n_tiles):
            tile_pairs = (2 * n - 1 - r0 - r1) * (r1 - r0) / 2
            tile_cost = g["cost"] * tile_pairs / total_pairs
            tile_tasks.append((gi, g["row_ids"], r0, r1, path, tile_cost))
    return tile_tasks


def generate_clusters_multi(
    bucket_datasets: Sequence[lance.LanceDataset],
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float | None,
    fragment_tol: float,
    batch_size: int,
    consensus_method: str,
    consensus_params: dict[str, Any],
) -> list[tuple[np.ndarray, list[ConsensusTuple]]]:
    """
    Cluster multiple per-charge datasets in one shared worker pool.

    All buckets' m/z-split tasks share a single ``joblib`` pool so idle
    workers immediately steal work across charge buckets rather than
    waiting at per-bucket barriers. Each m/z split is independent, so
    partitions match clustering every bucket individually.

    Parameters
    ----------
    bucket_datasets : Sequence[lance.LanceDataset]
        Per-charge Lance datasets to cluster.
    linkage : {"single", "complete", "average"}
        Hierarchical clustering linkage criterion.
    distance_threshold : float
        Cosine distance cut threshold.
    min_matches : int
        Minimum matched fragment peaks required for nonzero similarity.
    precursor_tol_mass : float
        Maximum precursor m/z tolerance within a cluster.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    rt_tol : float | None
        Maximum retention time tolerance, or ``None`` to disable.
    fragment_tol : float
        Fragment m/z tolerance (in Da).
    batch_size : int
        Maximum number of spectra per precursor m/z interval.
    consensus_method : {"medoid", "average"}
        Representative spectrum computation method.
    consensus_params : dict[str, Any]
        Extra keyword arguments for the consensus routine.

    Returns
    -------
    list[tuple[np.ndarray, list[ConsensusTuple]]]
        Per-dataset ``(labels, representative_spectra)`` results in the
        same order as ``bucket_datasets``.
    """
    logger.debug(
        "Hierarchical clustering (distance_threshold=%.4f, min_samples=%d)",
        distance_threshold,
        _MIN_CLUSTER_SAMPLES,
    )
    # Build m/z-split tasks for all buckets up front; `prepared` keeps
    # only lightweight (n, splits) after metadata DataFrames are freed.
    n_workers = multiprocessing.cpu_count()
    num_chunks = _CHUNKS_PER_WORKER * n_workers
    # Tile intervals >= batch_size // 2 so giant splits do not strand
    # cores in a single worker at the end of the run.
    tile_threshold = _tile_threshold(batch_size)
    prepared: list[tuple[int, nb.typed.List]] = []
    chunk_tasks: list[
        tuple[int, int, int, lance.LanceDataset, list[IntervalTask]]
    ] = []
    giants: list[dict[str, Any]] = []
    for bucket_id, dataset in enumerate(bucket_datasets):
        n, splits, data_chunks, giant_intervals = _prepare_bucket_chunks(
            dataset,
            precursor_tol_mass,
            precursor_tol_mode,
            batch_size,
            num_chunks,
            tile_threshold,
        )
        prepared.append((n, splits))
        for data_chunk in data_chunks:
            # Quadratic cost tracks pairwise distance work, guiding
            # heaviest-first scheduling and linear-ish progress ETAs.
            cost = sum(
                len(idx_interval) ** 2 for _, _, idx_interval, _ in data_chunk
            )
            n_chunk = sum(
                len(idx_interval) for _, _, idx_interval, _ in data_chunk
            )
            chunk_tasks.append((cost, bucket_id, n_chunk, dataset, data_chunk))
        for (
            task_id,
            row_ids,
            idx_interval,
            mz_interval,
            n_interval,
        ) in giant_intervals:
            giants.append(
                {
                    "bucket_id": bucket_id,
                    "dataset": dataset,
                    "task_id": task_id,
                    "row_ids": row_ids,
                    "idx": idx_interval,
                    "mz": mz_interval,
                    "n": n_interval,
                    # Progress weight matching chunk quadratic cost.
                    "cost": n_interval * n_interval,
                }
            )

    # Schedule heaviest chunks first so small buckets backfill around
    # dominant ones instead of idling cores at the tail. Sort by cost
    # only because LanceDataset objects are not orderable.
    chunk_tasks.sort(key=lambda t: t[0], reverse=True)

    results_by_bucket: dict[int, list[TaggedIntervalResult]] = defaultdict(
        list
    )
    memmap_paths: list[str] = []
    try:
        # Allocate shared condensed memmaps and row-band tiles; all
        # paths are unlinked in `finally` once Pass 2 completes.
        tile_tasks = _build_giant_tile_tasks(giants, n_workers, memmap_paths)

        total_cost = sum(t[0] for t in chunk_tasks) + sum(
            g["cost"] for g in giants
        )
        total_spectra = sum(n for n, _ in prepared)
        n_pass1 = len(chunk_tasks) + len(tile_tasks)
        done_cost = done_spectra = done_tasks = 0
        next_log_pct, start = 5, time.time()
        if n_pass1:
            logger.info(
                "Clustering %d spectra across %d charge bucket(s) in %d "
                "task(s) (%d giant interval(s) tiled) on %d worker(s)...",
                total_spectra,
                len(bucket_datasets),
                n_pass1,
                len(giants),
                n_workers,
            )
            process_chunk = partial(
                _cluster_chunk_tagged,
                linkage=linkage,
                distance_threshold=distance_threshold,
                min_matches=min_matches,
                precursor_tol_mass=precursor_tol_mass,
                precursor_tol_mode=precursor_tol_mode,
                rt_tol=rt_tol,
                fragment_mz_tol=fragment_tol,
                consensus_method=consensus_method,
                consensus_params=consensus_params,
                batch_size=batch_size,
            )

            # Submit giant tiles first (critical path) and normal chunks
            # second into one pool; `generator_unordered` streams tagged
            # completions as they finish for live progress logging.
            def _pass1_jobs() -> Iterator[Any]:
                for gi, row_ids, r0, r1, path, tile_cost in tile_tasks:
                    yield joblib.delayed(_tile_distance)(
                        giants[gi]["dataset"],
                        row_ids,
                        r0,
                        r1,
                        path,
                        fragment_tol,
                        min_matches,
                        gi,
                        tile_cost,
                    )
                for (
                    cost,
                    bucket_id,
                    n_chunk,
                    dataset,
                    data_chunk,
                ) in chunk_tasks:
                    yield joblib.delayed(process_chunk)(
                        data_chunk,
                        dataset=dataset,
                        bucket_id=bucket_id,
                        cost=cost,
                        n_chunk=n_chunk,
                    )

            results = joblib.Parallel(
                n_jobs=-1,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(_pass1_jobs())
            for result in results:
                if result[0] == "chunk":
                    _, bucket_id, cost, n_chunk, chunk_results = result
                    results_by_bucket[bucket_id].extend(chunk_results)
                    done_cost += cost
                    done_spectra += n_chunk
                else:  # "tile": one giant interval row-band completed
                    done_cost += result[2]
                done_tasks += 1
                pct = 100 * done_cost / total_cost if total_cost else 100
                if pct >= next_log_pct or done_tasks == n_pass1:
                    elapsed = time.time() - start
                    eta = (
                        elapsed * (total_cost - done_cost) / done_cost
                        if done_cost
                        else 0
                    )
                    logger.info(
                        "Clustering progress: %3.0f%% | %d/%d tasks | "
                        "%d/%d spectra processed | elapsed %s | ETA ~%s",
                        pct,
                        done_tasks,
                        n_pass1,
                        done_spectra,
                        total_spectra,
                        _format_hms(elapsed),
                        _format_hms(eta),
                    )
                    # Advance to the next 5% progress milestone.
                    next_log_pct = (int(pct) // 5 + 1) * 5

        # Pass 2: run linkage, refinement, and consensus for each giant
        # interval using its now-filled shared distance memmap.
        if giants:
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_results = joblib.Parallel(
                n_jobs=-1,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(
                joblib.delayed(_finalize_giant)(
                    g["dataset"],
                    g["row_ids"],
                    g["idx"],
                    g["mz"],
                    g["path"],
                    g["bucket_id"],
                    g["task_id"],
                    g["n"],
                    linkage=linkage,
                    distance_threshold=distance_threshold,
                    min_matches=min_matches,
                    precursor_tol_mass=precursor_tol_mass,
                    precursor_tol_mode=precursor_tol_mode,
                    rt_tol=rt_tol,
                    fragment_mz_tol=fragment_tol,
                    consensus_method=consensus_method,
                    consensus_params=consensus_params,
                )
                for g in giants
            )
            for _, bucket_id, task_id, n_chunk, result in finalize_results:
                results_by_bucket[bucket_id].append((task_id, result))
                done_spectra += n_chunk
    finally:
        for path in memmap_paths:
            with contextlib.suppress(OSError):
                os.unlink(path)

    # Reassemble per-bucket labels and representatives independently.
    return [
        _finalize_bucket(
            n,
            splits,
            results_by_bucket[bucket_id],
            _charge_label(bucket_datasets[bucket_id]),
        )
        for bucket_id, (n, splits) in enumerate(prepared)
    ]


def _cluster_chunk_tagged(
    data_chunk: list[IntervalTask],
    dataset: lance.LanceDataset,
    **cluster_kwargs: Any,
) -> tuple[str, int, int, int, list[TaggedIntervalResult]]:
    """
    Run ``cluster_chunk`` and tag results for routing and progress.

    Pops ``bucket_id``, ``cost``, and ``n_chunk`` from
    ``cluster_kwargs`` and forwards remaining options to
    ``cluster_chunk``.

    Parameters
    ----------
    data_chunk : list[IntervalTask]
        Intervals assigned to this worker chunk.
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    **cluster_kwargs : Any
        Routing metadata (``bucket_id``, ``cost``, ``n_chunk``) plus
        keyword arguments for ``cluster_chunk``.

    Returns
    -------
    tuple[str, int, int, int, list[TaggedIntervalResult]]
        ``("chunk", bucket_id, cost, n_chunk, chunk_results)`` so the
        unordered consumer can distinguish chunks from distance tiles.
    """
    bucket_id = cluster_kwargs.pop("bucket_id")
    cost = cluster_kwargs.pop("cost")
    n_chunk = cluster_kwargs.pop("n_chunk")
    return (
        "chunk",
        bucket_id,
        cost,
        n_chunk,
        cluster_chunk(data_chunk, dataset=dataset, **cluster_kwargs),
    )


def _interval_row_bounds(n: int, n_tiles: int) -> list[tuple[int, int]]:
    """
    Partition upper-triangle rows ``[0, n)`` into balanced tiles.

    Row ``i`` contributes ``n - 1 - i`` pairs, so splitting on
    cumulative pair count gives roughly equal pair work per tile (few
    heavy early rows, many light late rows). Empty ranges are dropped.

    Parameters
    ----------
    n : int
        Number of spectra in the interval.
    n_tiles : int
        Target number of work-balanced row bands.

    Returns
    -------
    list[tuple[int, int]]
        Non-empty contiguous ``(r0, r1)`` half-open row ranges.
    """
    cum = np.cumsum(n - 1 - np.arange(n))  # cum[-1] == n * (n - 1) // 2
    edges = [
        int(np.searchsorted(cum, cum[-1] * p / n_tiles))
        for p in range(1, n_tiles)
    ]
    bounds = [0, *edges, n]
    bounds = sorted({min(b, n) for b in bounds})
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i] < bounds[i + 1]
    ]


def _tile_distance(
    dataset: lance.LanceDataset,
    row_ids: list[int],
    r0: int,
    r1: int,
    memmap_path: str,
    fragment_mz_tol: float,
    min_matches: int,
    giant_idx: int,
    tile_cost: float,
) -> tuple[str, int, float]:
    """
    Compute rows ``[r0, r1)`` of a giant interval's distance matrix.

    Reads the interval's spectra, sorts them with
    ``_interval_spectrum_tuples`` so indexing matches Pass 2, and fills
    its disjoint row band in the shared memmap single-threaded.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    row_ids : list[int]
        Lance row indices belonging to the giant interval.
    r0 : int
        Start row index (inclusive).
    r1 : int
        Stop row index (exclusive).
    memmap_path : str
        Path to the shared ``.npy`` condensed distance memmap.
    fragment_mz_tol : float
        Fragment m/z tolerance (in Da).
    min_matches : int
        Minimum matched fragment peaks required.
    giant_idx : int
        Index into the caller's ``giants`` list.
    tile_cost : float
        Estimated pair-work weight for progress accounting.

    Returns
    -------
    tuple[str, int, float]
        ``("tile", giant_idx, tile_cost)`` progress/routing tag.
    """
    spectra = dataset.take(indices=row_ids, columns=_INTERVAL_COLUMNS)
    spec_tuples, _ = _interval_spectrum_tuples(spectra.to_pandas())
    condensed = np.lib.format.open_memmap(memmap_path, mode="r+")
    _condensed_rows(
        condensed, spec_tuples, r0, r1, fragment_mz_tol, min_matches
    )
    condensed.flush()
    return ("tile", giant_idx, tile_cost)


def _finalize_giant(
    dataset: lance.LanceDataset,
    row_ids: list[int],
    idx_interval: np.ndarray,
    mz_interval: np.ndarray,
    memmap_path: str,
    bucket_id: int,
    task_id: int,
    n_chunk: int,
    **cluster_kwargs: Any,
) -> tuple[str, int, int, int, IntervalResult]:
    """
    Cluster a giant interval from its precomputed distance memmap.

    Opens the shared memmap filled by ``_tile_distance`` in read-only
    mode and runs linkage, refinement, and consensus selection via
    ``_cluster_mz_interval``.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    row_ids : list[int]
        Lance row indices belonging to the giant interval.
    idx_interval : np.ndarray
        Positional indices within the bucket's sorted ordering.
    mz_interval : np.ndarray
        Precursor m/z values aligned with ``idx_interval``.
    memmap_path : str
        Path to the filled condensed distance ``.npy`` memmap.
    bucket_id : int
        Originating bucket index.
    task_id : int
        Interval split index within the bucket.
    n_chunk : int
        Number of spectra in the interval.
    **cluster_kwargs : Any
        Clustering and consensus options for ``_cluster_mz_interval``.

    Returns
    -------
    tuple[str, int, int, int, IntervalResult]
        ``("giant", bucket_id, task_id, n_chunk, (reps, labels))``.
    """
    spectra = dataset.take(indices=row_ids, columns=_INTERVAL_COLUMNS)
    pdist = np.lib.format.open_memmap(memmap_path, mode="r")
    result = _cluster_mz_interval(
        spectra.to_pandas(),
        idx_interval,
        mz_interval,
        pdist=pdist,
        **cluster_kwargs,
    )
    return ("giant", bucket_id, task_id, n_chunk, result)


def _prepare_bucket_chunks(
    dataset: lance.LanceDataset,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
    num_chunks: int,
    tile_threshold: int,
) -> tuple[
    int, nb.typed.List, list[list[IntervalTask]], list[GiantIntervalTask]
]:
    """
    Load one bucket's metadata and partition it into m/z-split tasks.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge Lance dataset.
    precursor_tol_mass : float
        Maximum precursor m/z gap before starting a new split.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    batch_size : int
        Maximum spectra per interval.
    num_chunks : int
        Target number of cost-balanced chunks for normal intervals.
        Empty chunks are dropped when fewer splits exist.
    tile_threshold : int
        Intervals with at least this many spectra are returned
        separately as giants for tiled distance computation.

    Returns
    -------
    tuple
        ``(n_spectra, splits, data_chunks, giant_intervals)``. The
        metadata DataFrame is released before returning so only compact
        task arrays survive.
    """
    # Load only metadata needed for sorting and splitting; peak arrays
    # are fetched per interval inside worker processes.
    data = dataset.to_table(
        columns=["identifier", "precursor_mz", "retention_time"]
    ).to_pandas()
    # Sort by (precursor_mz, retention_time, identifier) for a total
    # order matching `_export_results` when reattaching labels later.
    data = data.reset_index().sort_values(
        ["precursor_mz", "retention_time", "identifier"],
    )
    n = data.shape[0]
    logger.info("Cluster %d spectra with charge %s", n, _charge_label(dataset))
    idx = data["index"].values
    mzs = data["precursor_mz"].values
    splits = _get_precursor_mz_splits(
        mzs, precursor_tol_mass, precursor_tol_mode, batch_size
    )
    # Bundle splits into cost-balanced chunks to amortize dispatch
    # overhead while keeping enough tasks for dynamic load balancing.
    chunks = cost_based_chunking(splits, num_chunks)
    data_chunks: list[list[IntervalTask]] = []
    giant_intervals: list[GiantIntervalTask] = []
    for chunk in chunks:
        data_chunk: list[IntervalTask] = []
        for task_id, (interval_start, interval_stop) in chunk:
            row_ids = idx[interval_start:interval_stop]
            idx_interval = idx[interval_start:interval_stop]
            mz_interval = mzs[interval_start:interval_stop]
            n_interval = interval_stop - interval_start
            if n_interval >= tile_threshold:
                # Route giant intervals to Pass 1 tiled distance build.
                giant_intervals.append(
                    (
                        task_id,
                        row_ids.tolist(),
                        idx_interval,
                        mz_interval,
                        n_interval,
                    )
                )
            else:
                data_chunk.append(
                    (task_id, row_ids.tolist(), idx_interval, mz_interval)
                )
        if data_chunk:
            data_chunks.append(data_chunk)
    return n, splits, data_chunks, giant_intervals


def _finalize_bucket(
    n: int,
    splits: nb.typed.List,
    flattened_results: list[TaggedIntervalResult],
    charge: str,
) -> tuple[np.ndarray, list[ConsensusTuple]]:
    """
    Assemble per-split results into bucket-wide labels and consensus.

    Parameters
    ----------
    n : int
        Total number of spectra in the bucket.
    splits : nb.typed.List
        Interval split boundaries from ``_prepare_bucket_chunks``.
    flattened_results : list[TaggedIntervalResult]
        Unordered ``(task_id, (rep_spectra, labels))`` split results.
    charge : str
        Human-readable charge label for summary logging.

    Returns
    -------
    tuple[np.ndarray, list[ConsensusTuple]]
        Int32 cluster labels aligned to the bucket's canonical order
        and representative spectra with bucket-unique cluster IDs.
    """
    with tempfile.NamedTemporaryFile(suffix=".npy") as cluster_file:
        cluster_filename = cluster_file.name
        cluster_labels = np.lib.format.open_memmap(
            cluster_filename, mode="w+", dtype=np.int32, shape=(n,)
        )
        cluster_labels.fill(-1)
        representative_spectra: list[ConsensusTuple] = []
        # Sort by task_id so positional scatter matches split slices
        # regardless of worker completion order.
        for task_id, (interval_rep_spectra, labels) in sorted(
            flattened_results, key=lambda x: x[0]
        ):
            if interval_rep_spectra is not None:
                # Tag each representative with its split index for
                # global label offsetting below.
                interval_rep_spectra = [
                    s._replace(mz_split=np.int32(task_id))
                    for s in interval_rep_spectra
                ]
                representative_spectra.extend(interval_rep_spectra)
                cluster_labels[splits[task_id] : splits[task_id + 1]] = labels
        representative_spectra = _assign_global_cluster_labels(
            cluster_labels, representative_spectra, splits
        )
        _, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = np.count_nonzero(counts > 1)
        n_noise = np.count_nonzero(counts == 1)
        logger.info(
            "Charge %s: %d spectra grouped in %d clusters, %d spectra remain "
            "as singletons",
            charge,
            counts[counts > 1].sum(),
            n_clusters,
            n_noise,
        )
        cluster_labels.flush()
        return cluster_labels, representative_spectra


@nb.njit
def _get_precursor_mz_splits(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
) -> nb.typed.List:
    """
    Split sorted precursor m/z values into disjoint clustering blocks.

    Cuts whenever adjacent precursor m/z values differ by more than
    ``precursor_tol_mass`` or whenever a contiguous block reaches
    ``batch_size`` spectra (dividing oversized blocks evenly).

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Precursor m/z array sorted ascending.
    precursor_tol_mass : float
        Maximum gap between adjacent spectra within a block.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    batch_size : int
        Maximum number of spectra allowed in a single interval.

    Returns
    -------
    nb.typed.List
        Monotonic split-point indices starting at ``0`` and ending at
        ``len(precursor_mzs)`` (or ``[0]`` when empty).
    """
    splits = nb.typed.List([0])
    for i in range(1, len(precursor_mzs)):
        block_size = i - splits[-1]
        if (
            suu.mass_diff(
                precursor_mzs[i],
                precursor_mzs[i - 1],
                precursor_tol_mode == "Da",
            )
            > precursor_tol_mass
        ):
            if block_size < batch_size:
                splits.append(i)
            else:
                # Partition oversized blocks into evenly sized slices.
                n_chunks = math.ceil(block_size / batch_size)
                chunk_size = block_size // n_chunks
                for _ in range(block_size % n_chunks):
                    splits.append(splits[-1] + chunk_size + 1)
                for _ in range(n_chunks - (block_size % n_chunks)):
                    splits.append(splits[-1] + chunk_size)
        elif block_size >= batch_size:
            splits.append(i)
    if splits[-1] != len(precursor_mzs):
        splits.append(len(precursor_mzs))
    return splits


def cost_based_chunking(
    tasks: Sequence[int], num_chunks: int
) -> list[list[tuple[int, tuple[int, int]]]]:
    """
    Distribute interval tasks across chunks by quadratic cost.

    Uses greedy longest-processing-time-first (LPT) scheduling where
    each interval's estimated cost is ``length ** 2``.

    Parameters
    ----------
    tasks : Sequence[int]
        Sorted split-point indices defining contiguous intervals.
    num_chunks : int
        Maximum number of chunks to produce.

    Returns
    -------
    list[list[tuple[int, tuple[int, int]]]]
        Non-empty chunks of ``(task_index, (start, stop))`` entries.
    """
    split_tuples = [(tasks[i], tasks[i + 1]) for i in range(len(tasks) - 1)]
    indexed_tasks = list(enumerate(split_tuples))
    indexed_tasks.sort(key=lambda x: (x[1][1] - x[1][0]) ** 2, reverse=True)

    chunks: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(num_chunks)
    ]
    costs = [0] * num_chunks

    # Greedily place each interval into the least-loaded chunk.
    for index, task in indexed_tasks:
        idx = costs.index(min(costs))
        chunks[idx].append((index, task))
        costs[idx] += (task[1] - task[0]) ** 2

    return [chunk for chunk in chunks if chunk]


def cluster_chunk(
    chunk: list[IntervalTask],
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float | None,
    fragment_mz_tol: float,
    consensus_method: str,
    consensus_params: dict[str, Any],
    batch_size: int | None = None,
) -> list[TaggedIntervalResult]:
    """
    Cluster all m/z intervals in a worker chunk with batched reads.

    Parameters
    ----------
    chunk : list[IntervalTask]
        Interval tasks as ``(task_id, row_ids, idx_interval,
        mz_interval)`` tuples.
    dataset : lance.LanceDataset
        Source per-charge Lance dataset.
    linkage : {"single", "complete", "average"}
        Hierarchical clustering linkage criterion.
    distance_threshold : float
        Cosine distance cut threshold.
    min_matches : int
        Minimum matched fragment peaks required for nonzero similarity.
    precursor_tol_mass : float
        Maximum precursor m/z tolerance within a cluster.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    rt_tol : float | None
        Maximum retention time tolerance, or ``None`` to disable.
    fragment_mz_tol : float
        Fragment m/z tolerance (in Da).
    consensus_method : {"medoid", "average"}
        Representative spectrum computation method.
    consensus_params : dict[str, Any]
        Extra keyword arguments for the consensus routine.
    batch_size : int | None, default=None
        Cap on spectra fetched per batched Lance ``take`` call,
        bounding worker memory while amortizing small-interval reads.
        Reads the entire chunk at once when ``None``.

    Returns
    -------
    list[TaggedIntervalResult]
        List of ``(task_id, (rep_spectra, cluster_labels))`` results.
    """
    if not chunk:
        return []
    columns = _INTERVAL_COLUMNS

    def _cluster_group(
        group: list[IntervalTask],
    ) -> list[TaggedIntervalResult]:
        # Batch Lance reads across small intervals; `take` preserves
        # row_ids order so each interval slices contiguously.
        group_row_ids = [rid for _, row_ids, _, _ in group for rid in row_ids]
        group_spectra = dataset.take(
            indices=group_row_ids, columns=columns
        ).to_pandas()
        out: list[TaggedIntervalResult] = []
        offset = 0
        for i, row_ids, idx, mzs in group:
            interval_spectra = group_spectra.iloc[
                offset : offset + len(row_ids)
            ]
            offset += len(row_ids)
            out.append(
                (
                    i,
                    _cluster_mz_interval(
                        interval_spectra,
                        idx,
                        mzs,
                        linkage,
                        distance_threshold,
                        min_matches,
                        precursor_tol_mass,
                        precursor_tol_mode,
                        rt_tol,
                        fragment_mz_tol,
                        consensus_method,
                        consensus_params,
                    ),
                )
            )
        return out

    # Group adjacent intervals up to `batch_size` spectra per read; a
    # single interval is never split across reads.
    results: list[TaggedIntervalResult] = []
    group: list[IntervalTask] = []
    group_size = 0
    for task in chunk:
        m = len(task[1])
        if group and batch_size is not None and group_size + m > batch_size:
            results.extend(_cluster_group(group))
            group, group_size = [], 0
        group.append(task)
        group_size += m
    if group:
        results.extend(_cluster_group(group))
    return results


def _interval_spectrum_tuples(
    spectra: pd.DataFrame,
) -> tuple[list[similarity.SpectrumTuple], np.ndarray]:
    """
    Sort an interval's spectra in canonical order and build tuples.

    Uses the exact total-order sort key from ``_prepare_bucket_chunks``
    (``precursor_mz``, ``retention_time``, ``identifier``) so both
    ``_cluster_mz_interval`` and ``_tile_distance`` index spectra
    identically.

    Parameters
    ----------
    spectra : pd.DataFrame
        Interval spectrum table containing ``_INTERVAL_COLUMNS``.

    Returns
    -------
    tuple[list[SpectrumTuple], np.ndarray]
        Canonically ordered ``SpectrumTuple`` objects and matching
        retention time array.
    """
    spectra = spectra.sort_values(
        ["precursor_mz", "retention_time", "identifier"]
    )
    rts = spectra["retention_time"].values
    # Extract NumPy column views directly rather than row-wise pandas
    # apply, which would otherwise dominate single-threaded interval
    # setup cost.
    precursor_mz = spectra["precursor_mz"].to_numpy()
    precursor_charge = spectra["precursor_charge"].to_numpy()
    mz = spectra["mz"].to_numpy()
    intensity = spectra["intensity"].to_numpy()
    spec_tuples = [
        similarity.SpectrumTuple(
            precursor_mz[i], precursor_charge[i], mz[i], intensity[i]
        )
        for i in range(len(precursor_mz))
    ]
    return spec_tuples, rts


def _singleton_consensus_tuple(
    spec: similarity.SpectrumTuple,
    retention_time: float,
    cluster_id: int,
) -> ConsensusTuple:
    """
    Build a ``ConsensusTuple`` representing a single-spectrum cluster.

    Parameters
    ----------
    spec : SpectrumTuple
        Member spectrum of the singleton cluster.
    retention_time : float
        Spectrum retention time (or ``NaN`` when missing).
    cluster_id : int
        Cluster label assigned to the singleton.

    Returns
    -------
    ConsensusTuple
        Representative tuple with float32/int32 dtypes and
        ``mz_split=None``.
    """
    return ConsensusTuple(
        precursor_mz=np.float32(spec.precursor_mz),
        precursor_charge=(
            np.int32(spec.precursor_charge)
            if not np.isnan(spec.precursor_charge)
            else np.nan
        ),
        mz=spec.mz.astype(np.float32),
        intensity=spec.intensity.astype(np.float32),
        retention_time=np.float32(retention_time),
        cluster_id=np.int32(cluster_id),
        mz_split=None,
    )


def _cluster_mz_interval(
    spectra: pd.DataFrame,
    idx: np.ndarray,
    mzs: np.ndarray,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float | None,
    fragment_mz_tol: float,
    consensus_method: str,
    consensus_params: dict[str, Any],
    pdist: np.ndarray | None = None,
) -> IntervalResult:
    """
    Cluster spectra within one precursor m/z interval.

    Parameters
    ----------
    spectra : pd.DataFrame
        Pre-fetched interval spectra containing ``_INTERVAL_COLUMNS``.
    idx : np.ndarray
        Positional indices within the bucket's sorted ordering.
    mzs : np.ndarray
        Precursor m/z values aligned with ``idx``.
    linkage : {"single", "complete", "average"}
        Hierarchical clustering linkage criterion.
    distance_threshold : float
        Cosine distance cut threshold.
    min_matches : int
        Minimum matched fragment peaks required for nonzero similarity.
    precursor_tol_mass : float
        Maximum precursor m/z tolerance within a refined cluster.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    rt_tol : float | None
        Maximum retention time tolerance, or ``None`` to disable.
    fragment_mz_tol : float
        Fragment m/z tolerance (in Da).
    consensus_method : {"medoid", "average"}
        Representative spectrum computation method.
    consensus_params : dict[str, Any]
        Extra keyword arguments for the consensus routine.
    pdist : np.ndarray | None, default=None
        Optional precomputed condensed distance matrix (used by the
        tiled giant path) indexed identically to
        ``_interval_spectrum_tuples``.

    Returns
    -------
    IntervalResult
        Representative spectra and int32 cluster labels aligned to the
        interval's canonical input order.
    """
    spec_tuples, rts = _interval_spectrum_tuples(spectra)
    n_spectra = len(spec_tuples)
    cluster_labels = -np.ones(n_spectra, np.int32)
    if n_spectra > 1:
        # Build condensed fragment cosine distances and cut dendrogram
        # (shift SciPy 1-based labels to 0-based).
        if pdist is None:
            pdist = compute_condensed_distance_matrix(
                spec_tuples,
                fragment_mz_tol,
                min_matches,
            )
        labels = (
            sch.fcluster(
                fastcluster.linkage(pdist, linkage),
                distance_threshold,
                "distance",
            )
            - 1
        )
        # Refine clusters on precursor m/z (and optional RT), tracking
        # forward permutations and inverses across two sorts:
        # - `label_order` groups labels for `_postprocess_cluster`.
        # - `refined_order` groups final labels for representative work.
        # `order_map` maps final positions to `pdist`/`spec_tuples`;
        # composed inverses scatter labels back to input order.
        label_order = np.argsort(labels)
        label_order_inv = np.argsort(label_order)
        idx, mzs, rts = (
            idx[label_order],
            mzs[label_order],
            rts[label_order],
        )
        labels, current_label = labels[label_order], 0
        for start_i, stop_i in _get_cluster_group_idx(labels):
            n_clusters = _postprocess_cluster(
                labels[start_i:stop_i],
                mzs[start_i:stop_i],
                rts[start_i:stop_i],
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                _MIN_CLUSTER_SAMPLES,
                current_label,
            )
            current_label += n_clusters
        # Compute cluster representatives.
        if current_label < n_spectra:
            refined_order = np.argsort(labels)
            refined_order_inv = np.argsort(refined_order)
            idx = idx[refined_order]
            labels = labels[refined_order]
            rts = rts[refined_order]
            order_map = np.arange(len(labels))[label_order][refined_order]
            if consensus_method == "medoid":
                consensus_params["pdist"] = pdist
            # Representative spectra are returned sorted by label.
            rep_spectra = _get_representative_spectra(
                spec_tuples,
                labels,
                rts,
                order_map,
                consensus_method,
                consensus_params,
            )
            cluster_labels = labels[refined_order_inv[label_order_inv]]
        else:  # Every spectrum ended up as its own singleton.
            rts_in = rts[label_order_inv]
            labels_in = labels[label_order_inv]
            rep_spectra = [
                _singleton_consensus_tuple(spec, rts_in[i], labels_in[i])
                for i, spec in enumerate(spec_tuples)
            ]
            cluster_labels = labels[label_order_inv]
        # Release memmap backing storage eagerly for large intervals.
        del pdist
        if n_spectra > _GC_COLLECT_MIN_SPECTRA:
            gc.collect()
    else:  # Single-spectrum interval.
        rep_spectra = [_singleton_consensus_tuple(spec_tuples[0], rts[0], 0)]
        cluster_labels[0] = 0
    return rep_spectra, cluster_labels


@nb.njit(boundscheck=False)
def _postprocess_cluster(
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
    # Assign singleton labels directly if the group is already small.
    if cluster_labels.shape[0] < min_samples:
        cluster_labels[:] = np.arange(
            start_label, start_label + cluster_labels.shape[0]
        )
        return cluster_labels.shape[0]
    else:
        # Split on full precursor m/z span via 1D complete linkage
        # (shift SciPy 1-based flat cluster IDs to 0-based).
        linkage = _linkage(cluster_mzs, precursor_tol_mode)
        with nb.objmode(cluster_assignments="int32[:]"):
            cluster_assignments = (
                sch.fcluster(linkage, precursor_tol_mass, "distance") - 1
            ).astype(np.int32)
        # Optionally intersect with an RT partition. Known-RT spectra
        # undergo 1D RT linkage; NaN-RT spectra receive a dedicated RT
        # code so they stay grouped by m/z without merging into known
        # RT clusters.
        if rt_tol is not None:
            with nb.objmode(cluster_assignments="int32[:]"):
                nan_mask = np.isnan(cluster_rts)
                if not nan_mask.all():
                    valid_rts = cluster_rts[~nan_mask]
                    rt_labels = np.zeros(len(cluster_rts), np.int32)
                    if len(valid_rts) >= 2:
                        rt_labels[~nan_mask] = (
                            fcluster(_linkage(valid_rts), rt_tol, "distance")
                            - 1
                        ).astype(np.int32)
                    # Place all NaN-RT spectra in a distinct RT bin
                    # above the highest valid RT label.
                    n_rt = int(rt_labels[~nan_mask].max()) + 1
                    rt_labels[nan_mask] = n_rt
                    # Injective mixed-radix encoding of (m/z, RT) pairs
                    # with radix (n_rt + 1) covering labels 0..n_rt.
                    cluster_assignments = np.unique(
                        cluster_assignments * (n_rt + 1) + rt_labels,
                        return_inverse=True,
                    )[1].astype(np.int32)
                else:
                    # All RTs are NaN: keep m/z-only assignments.
                    cluster_assignments = cluster_assignments.copy()

        n_clusters = int(np.max(cluster_assignments)) + 1
        # Relabel multi-spectrum clusters and mark small groups as -1.
        if n_clusters == 1:
            cluster_labels.fill(start_label)
        elif n_clusters == cluster_labels.shape[0]:
            cluster_labels.fill(-1)
            n_clusters = 0
        else:
            labels = nb.typed.Dict.empty(
                key_type=nb.int64, value_type=nb.int64
            )
            for label in cluster_assignments:
                labels[label] = labels.get(label, 0) + 1
            n_clusters = 0
            for label, count in labels.items():
                if count < min_samples:
                    labels[label] = -1
                else:
                    labels[label] = start_label + n_clusters
                    n_clusters += 1
            for i, label in enumerate(cluster_assignments):
                cluster_labels[i] = labels[label]
        # Give every -1 singleton its own sequential label.
        mask = cluster_labels == -1
        n_singletons = int(np.count_nonzero(mask))
        cluster_labels[mask] = np.arange(
            start_label + n_clusters,
            start_label + n_clusters + n_singletons,
        )
        n_clusters += n_singletons
        return n_clusters


@nb.njit(cache=True, fastmath=True)
def _linkage(values: np.ndarray, tol_mode: str | None = None) -> np.ndarray:
    """
    Compute 1D complete-linkage hierarchical clustering over values.

    Exploits 1D ordering so only adjacent clusters can merge, avoiding a
    quadratic pairwise distance matrix. Output follows SciPy's linkage
    matrix format:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

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
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # Track (min_val, max_val, cluster_id, size) in ascending order.
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            # Complete-linkage span between adjacent sorted clusters.
            dist = clusters[i + 1][1] - clusters[i][0]
            if tol_mode == "ppm":
                dist = dist / clusters[i][0] * 10**6
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
            values.shape[0] + it,
            n_points,
        )
        del clusters[min_i + 1]

    return linkage


@nb.njit(boundscheck=False)
def _offset_cluster_labels(
    cluster_labels: np.ndarray,
    splits: nb.typed.List,
) -> np.ndarray:
    """
    Offset per-split labels in place so labels are unique per bucket.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Bucket-wide int32 cluster labels mutated in place.
    splits : nb.typed.List
        Monotonic split-point indices delineating each interval.

    Returns
    -------
    np.ndarray
        Int64 array of label offsets applied to each split.
    """
    n_splits = len(splits) - 1
    offsets = np.empty(n_splits, np.int64)
    current_label = 0
    for i in range(n_splits):
        start, end = splits[i], splits[i + 1]
        offsets[i] = current_label
        cluster_labels[start:end] += current_label
        current_label = np.max(cluster_labels[start:end]) + 1
    return offsets


def _assign_global_cluster_labels(
    cluster_labels: np.ndarray,
    rep_spectra: list[ConsensusTuple],
    splits: nb.typed.List,
) -> list[ConsensusTuple]:
    """
    Offset split-local labels and representatives to bucket-unique IDs.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Bucket-wide int32 cluster labels mutated in place.
    rep_spectra : list[ConsensusTuple]
        Representative spectra tagged with their ``mz_split`` index.
    splits : nb.typed.List
        Monotonic split-point indices delineating each interval.

    Returns
    -------
    list[ConsensusTuple]
        Representative spectra with bucket-unique int32 ``cluster_id``
        values.
    """
    offsets = _offset_cluster_labels(cluster_labels, splits)
    # Group representatives by split in O(R) instead of O(S * R).
    reps_by_split: dict[int, list[ConsensusTuple]] = {}
    for s in rep_spectra:
        reps_by_split.setdefault(int(s.mz_split), []).append(s)
    relabeled: list[ConsensusTuple] = []
    for i, offset in enumerate(offsets):
        for s in reps_by_split.get(i, ()):
            relabeled.append(
                s._replace(cluster_id=np.int32(s.cluster_id + int(offset)))
            )
    return relabeled
