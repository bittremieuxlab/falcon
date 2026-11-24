"""Parallel clustering coordinator, chunk execution, and assembly."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pyarrow as pa

from ..config import MedoidConsensusConfig
from ..dataset import SORTED_INTERVAL_COLUMNS, bucket_label_from_uri
from ..types import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    ClusterResult,
    ConsensusTuple,
    offset_cluster_result,
)
from ..utils import (
    effective_memory_bytes,
    effective_n_workers,
    format_hms,
    loky_unordered_pool,
)
from .distance import close_memmap
from .interval import (
    MIN_CLUSTER_SAMPLES,
    ClusteringOptions,
    cluster_arrow_interval,
    cluster_mz_interval,
    interval_spectrum_tuples,
)
from .scheduling import (
    CHUNKS_PER_WORKER,
    effective_tile_threshold,
    prepare_bucket_chunks,
)
from .tiling import (
    GiantMemmapStore,
    TileDone,
    TileTask,
    allocate_giant_tiles,
    tile_distance,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import lance

    from ..config import ClusteringConfig, ConsensusConfig
    from ..types import PathLike
    from .scheduling import BucketPlan, IntervalSpec

logger = logging.getLogger(__name__)

TaggedIntervalResult = tuple[int, ClusterResult]

# Columns read for normal chunks when representative spectra are off.
_CHUNK_NO_REP_COLUMNS: tuple[str, ...] = (
    "precursor_mz",
    "retention_time",
    "mz",
    "intensity",
)

# Columns needed for Pass 2 giant finalization when representatives are
# skipped (distances are already precomputed in the shared memmap).
_GIANT_METADATA_COLUMNS: tuple[str, ...] = (
    "precursor_mz",
    "retention_time",
)

# Estimated bytes per condensed pair during Pass 2 giant linkage:
# memmap page cache (4 B) plus fastcluster's float32 copy (4 B) and
# float64 working copy (8 B). Reserve 30% headroom for OS/workers.
_PASS2_BYTES_PER_PAIR = 16
_PASS2_RAM_BUDGET_FRACTION = 0.7

__all__ = [
    "ChunkDone",
    "ChunkTask",
    "ClusteringOptions",
    "GiantDone",
    "GiantInterval",
    "TaggedIntervalResult",
    "cluster_chunk",
    "cluster_mz_interval",
    "finalize_bucket",
    "finalize_giant",
    "generate_clusters",
    "generate_clusters_multi",
    "interval_spectrum_tuples",
]


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """Cost-balanced bundle of normal intervals for one worker."""

    bucket_id: int
    cost: int
    n_spectra: int
    dataset: lance.LanceDataset
    intervals: tuple[IntervalSpec, ...]


@dataclass(frozen=True, slots=True)
class ChunkDone:
    """Completion record emitted by a Pass 1 normal-chunk worker."""

    bucket_id: int
    cost: int
    n_spectra: int
    results: list[TaggedIntervalResult]


@dataclass(frozen=True, slots=True)
class GiantInterval:
    """Large interval whose distance matrix is built via tiling."""

    giant_idx: int
    bucket_id: int
    dataset: lance.LanceDataset
    spec: IntervalSpec
    memmap_path: str
    options: ClusteringOptions


@dataclass(frozen=True, slots=True)
class GiantDone:
    """Completion record emitted by a Pass 2 giant worker."""

    giant_idx: int
    bucket_id: int
    task_id: int
    result: ClusterResult


def _pass2_worker_limit(
    giants: Sequence[GiantInterval], n_workers: int
) -> int:
    """Cap concurrent Pass 2 giant workers against effective RAM."""
    max_workers = min(max(1, n_workers), len(giants))
    if max_workers <= 1:
        return max(1, max_workers)
    total_ram = effective_memory_bytes()
    if total_ram is None:
        return max_workers
    budget_bytes = int(total_ram * _PASS2_RAM_BUDGET_FRACTION)
    max_n = max(giant.spec.n_spectra for giant in giants)
    per_giant_bytes = max(1, _PASS2_BYTES_PER_PAIR * max_n * (max_n - 1) // 2)
    return min(max_workers, max(1, budget_bytes // per_giant_bytes))


def cluster_chunk(
    chunk: Sequence[IntervalSpec],
    dataset: lance.LanceDataset,
    options: ClusteringOptions,
    *,
    read_batch_size: int | None = None,
) -> list[TaggedIntervalResult]:
    """
    Cluster all intervals in a worker chunk with batched Lance reads.

    Reads Arrow tables directly from Lance and streams CSR peak buffers
    into :func:`~falcon.cluster.interval.cluster_arrow_interval`,
    omitting ``precursor_charge`` when representatives are disabled.

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
    columns = list(
        SORTED_INTERVAL_COLUMNS
        if options.compute_representatives
        else _CHUNK_NO_REP_COLUMNS
    )

    def _cluster_group(
        group: list[IntervalSpec],
    ) -> list[TaggedIntervalResult]:
        # Batch Lance reads across small intervals; `take` preserves
        # row_ids order (already canonical) so slices stay contiguous.
        group_row_ids = np.concatenate([spec.row_ids for spec in group])
        group_table = dataset.take(
            indices=pa.array(group_row_ids), columns=columns
        )
        out: list[TaggedIntervalResult] = []
        offset = 0
        for spec in group:
            n_rows = spec.n_spectra
            interval_table = group_table.slice(offset, n_rows)
            offset += n_rows
            out.append(
                (spec.task_id, cluster_arrow_interval(interval_table, options))
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
) -> ChunkDone:
    """Execute one ``ChunkTask`` and return its completion record."""
    chunk_results = cluster_chunk(
        task.intervals,
        dataset=task.dataset,
        options=options,
        read_batch_size=options.clustering.batch_size,
    )
    return ChunkDone(
        bucket_id=task.bucket_id,
        cost=task.cost,
        n_spectra=task.n_spectra,
        results=chunk_results,
    )


def finalize_giant(giant: GiantInterval) -> GiantDone:
    """
    Cluster a giant interval from its precomputed distance memmap.

    When representative spectra are not requested, reads only
    ``precursor_mz`` and ``retention_time`` to avoid loading fragment
    peak arrays or constructing ``SpectrumTuple`` records.

    Parameters
    ----------
    giant : GiantInterval
        Giant interval descriptor whose memmap has been populated.

    Returns
    -------
    GiantDone
        Typed completion record carrying the interval's cluster result.
    """
    columns = (
        list(SORTED_INTERVAL_COLUMNS)
        if giant.options.compute_representatives
        else list(_GIANT_METADATA_COLUMNS)
    )
    table = giant.dataset.take(
        indices=pa.array(giant.spec.row_ids), columns=columns
    )
    pdist = np.lib.format.open_memmap(giant.memmap_path, mode="r")
    try:
        result = cluster_arrow_interval(table, giant.options, pdist=pdist)
    finally:
        close_memmap(pdist)
    return GiantDone(
        giant_idx=giant.giant_idx,
        bucket_id=giant.bucket_id,
        task_id=giant.spec.task_id,
        result=result,
    )


def finalize_bucket(
    n: int,
    splits: Sequence[int] | np.ndarray,
    interval_results: Sequence[TaggedIntervalResult],
    bucket_label: str,
) -> ClusterResult:
    """
    Assemble per-split results into bucket-wide labels and consensus.

    Validates split monotonicity, coverage, and per-interval label
    lengths, then computes cumulative label offsets in a single sorted
    pass so labels and representatives are unique across the bucket.

    Parameters
    ----------
    n : int
        Total number of spectra in the bucket.
    splits : Sequence[int] | np.ndarray
        Monotonic interval split boundaries from
        ``prepare_bucket_chunks``.
    interval_results : Sequence[TaggedIntervalResult]
        Unordered ``(task_id, ClusterResult)`` split results.
    bucket_label : str
        Human-readable charge label for summary logging.

    Returns
    -------
    ClusterResult
        Int32 cluster labels aligned to canonical order and
        representative spectra with bucket-unique int32 cluster IDs.

    Raises
    ------
    ValueError
        If ``splits`` is not 1D monotonic covering ``[0, n]``, if
        ``interval_results`` task IDs do not cover ``0..len(splits)-2``,
        or if any split has mismatched label length or negative labels.
    """
    splits_arr = np.asarray(splits, dtype=np.int64)
    if (
        splits_arr.ndim != 1
        or splits_arr.size == 0
        or int(splits_arr[0]) != 0
        or int(splits_arr[-1]) != n
        or bool(np.any(splits_arr[1:] < splits_arr[:-1]))
    ):
        raise ValueError(
            f"Invalid splits for n={n}: expected 1D monotonic bounds "
            f"starting at 0 and ending at {n}"
        )
    n_splits = int(splits_arr.size) - 1
    sorted_results = sorted(interval_results, key=lambda item: item[0])
    task_ids = [task_id for task_id, _ in sorted_results]
    if task_ids != list(range(n_splits)):
        raise ValueError(
            f"Expected interval results for task IDs 0..{n_splits - 1}, "
            f"got {task_ids}"
        )

    cluster_labels = np.full(n, -1, dtype=np.int32)
    representative_spectra: list[ConsensusTuple] = []
    current_offset = 0

    for task_id, interval_result in sorted_results:
        start, stop = int(splits_arr[task_id]), int(splits_arr[task_id + 1])
        expected_len = stop - start
        if len(interval_result.labels) != expected_len:
            raise ValueError(
                f"Split {task_id} expected {expected_len} label(s), got "
                f"{len(interval_result.labels)}"
            )
        if expected_len > 0 and int(np.min(interval_result.labels)) < 0:
            raise ValueError(
                f"Split {task_id} contains negative cluster labels"
            )
        shifted, current_offset = offset_cluster_result(
            interval_result, current_offset
        )
        cluster_labels[start:stop] = shifted.labels
        if shifted.representatives:
            representative_spectra.extend(shifted.representatives)

    counts = np.bincount(cluster_labels, minlength=current_offset)
    n_clusters = int(np.count_nonzero(counts > 1))
    n_singletons = int(np.count_nonzero(counts == 1))
    logger.info(
        "Charge %s: %d spectra grouped in %d clusters, %d spectra remain "
        "as singletons",
        bucket_label,
        n - n_singletons,
        n_clusters,
        n_singletons,
    )
    return ClusterResult(cluster_labels, representative_spectra)


@dataclass(slots=True)
class _Pass1Progress:
    """Track Pass 1 cost/spectrum completion and throttled ETA logs."""

    n_tasks: int
    total_cost: float
    total_spectra: int
    done_tasks: int = 0
    done_cost: float = 0.0
    done_spectra: float = 0.0
    next_log_pct: int = 5
    start: float = 0.0

    def __post_init__(self) -> None:
        self.start = time.monotonic()

    def record(self, cost: float, n_spectra: float) -> None:
        """Record one finished task and log every 5% or on finish."""
        self.done_tasks += 1
        self.done_cost += cost
        self.done_spectra += n_spectra
        pct = (
            min(100.0, 100.0 * self.done_cost / self.total_cost)
            if self.total_cost
            else 100.0
        )
        if pct >= self.next_log_pct or self.done_tasks == self.n_tasks:
            elapsed = time.monotonic() - self.start
            remaining_cost = max(0.0, self.total_cost - self.done_cost)
            eta = (
                elapsed * remaining_cost / self.done_cost
                if self.done_cost
                else 0.0
            )
            logger.info(
                "Clustering progress: %3.0f%% | %d/%d tasks | "
                "%d/%d spectra processed | elapsed %s | ETA ~%s",
                pct,
                self.done_tasks,
                self.n_tasks,
                min(self.total_spectra, round(self.done_spectra)),
                self.total_spectra,
                format_hms(elapsed),
                format_hms(eta),
            )
            self.next_log_pct = (int(pct) // 5 + 1) * 5


def _run_pass1(
    chunk_tasks: Sequence[ChunkTask],
    tile_tasks: Sequence[TileTask],
    options: ClusteringOptions,
    *,
    n_buckets: int,
    n_workers: int,
    total_spectra: int,
) -> dict[int, list[TaggedIntervalResult]]:
    """Run Pass 1 (giant tiles and normal chunks) in joint LPT order."""
    results_by_bucket: dict[int, list[TaggedIntervalResult]] = {}
    n_pass1 = len(chunk_tasks) + len(tile_tasks)
    if n_pass1 == 0:
        return results_by_bucket

    pool_workers = min(n_workers, n_pass1)
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
        pool_workers,
    )
    run_chunk = partial(_cluster_chunk_tagged, options=options)

    # Sort combined Pass 1 tasks jointly by descending cost (LPT) so
    # heavy chunks never queue behind light giant tiles.
    pass1_tasks: list[tuple[float, ChunkTask | TileTask]] = [
        *((tile.tile_cost, tile) for tile in tile_tasks),
        *((float(task.cost), task) for task in chunk_tasks),
    ]
    pass1_tasks.sort(key=lambda item: item[0], reverse=True)

    def _jobs() -> Iterator[Any]:
        for _cost, work_item in pass1_tasks:
            if isinstance(work_item, TileTask):
                yield joblib.delayed(tile_distance)(work_item)
            else:
                yield joblib.delayed(run_chunk)(work_item)

    progress = _Pass1Progress(
        n_tasks=n_pass1,
        total_cost=total_cost,
        total_spectra=total_spectra,
    )
    done_gen: Any = loky_unordered_pool(pool_workers)(_jobs())
    with contextlib.closing(done_gen) as done_iter:
        for done in done_iter:
            if isinstance(done, ChunkDone):
                results_by_bucket.setdefault(done.bucket_id, []).extend(
                    done.results
                )
                progress.record(done.cost, done.n_spectra)
            elif isinstance(done, TileDone):
                progress.record(done.tile_cost, done.tile_spectra)
            else:
                raise TypeError(
                    f"Unexpected Pass 1 completion record: "
                    f"{type(done).__name__}"
                )

    return results_by_bucket


def _run_pass2(
    giants: Sequence[GiantInterval],
    memmap_store: GiantMemmapStore,
    results_by_bucket: dict[int, list[TaggedIntervalResult]],
    n_workers: int,
) -> None:
    """Run Pass 2 RAM-bounded giant finalization and release memmaps."""
    if not giants:
        return
    pass2_workers = _pass2_worker_limit(giants, n_workers)
    logger.info("Finalizing %d giant interval(s)...", len(giants))
    finalize_ctx: contextlib.AbstractContextManager[Iterable[GiantDone]]
    if pass2_workers <= 1:
        finalize_ctx = contextlib.nullcontext(
            finalize_giant(giant) for giant in giants
        )
    else:
        finalize_gen: Any = loky_unordered_pool(pass2_workers)(
            joblib.delayed(finalize_giant)(giant) for giant in giants
        )
        finalize_ctx = contextlib.closing(finalize_gen)
    with finalize_ctx as finalize_results:
        for done in finalize_results:
            results_by_bucket.setdefault(done.bucket_id, []).append(
                (done.task_id, done.result)
            )
            memmap_store.release(done.giant_idx)


def _plan_buckets(
    bucket_datasets: Sequence[lance.LanceDataset],
    labels_list: Sequence[str],
    clustering: ClusteringConfig,
    n_workers: int,
) -> tuple[
    list[BucketPlan],
    list[ChunkTask],
    list[tuple[int, lance.LanceDataset, IntervalSpec]],
]:
    """Build per-bucket plans, normal chunks, and giant specs."""
    num_chunks = CHUNKS_PER_WORKER * n_workers
    threshold = effective_tile_threshold(clustering.batch_size, n_workers)
    plans: list[BucketPlan] = []
    chunk_tasks: list[ChunkTask] = []
    raw_giants: list[tuple[int, lance.LanceDataset, IntervalSpec]] = []

    for bucket_id, (dataset, label) in enumerate(
        zip(bucket_datasets, labels_list, strict=True)
    ):
        plan = prepare_bucket_chunks(
            dataset, label, clustering, num_chunks, threshold
        )
        plans.append(plan)
        for specs in plan.normal_chunks:
            chunk_tasks.append(
                ChunkTask(
                    bucket_id=bucket_id,
                    cost=sum(spec.cost for spec in specs),
                    n_spectra=sum(spec.n_spectra for spec in specs),
                    dataset=dataset,
                    intervals=tuple(specs),
                )
            )
        for spec in plan.giant_specs:
            raw_giants.append((bucket_id, dataset, spec))
    return plans, chunk_tasks, raw_giants


def _prepare_giants(
    raw_giants: Sequence[tuple[int, lance.LanceDataset, IntervalSpec]],
    options: ClusteringOptions,
    n_workers: int,
    memmap_store: GiantMemmapStore,
) -> tuple[list[GiantInterval], list[TileTask]]:
    """Allocate giant memmaps, sort giants by LPT, and plan tiles."""
    memmap_paths, tile_tasks = allocate_giant_tiles(
        [(dataset, spec) for _bucket_id, dataset, spec in raw_giants],
        options.clustering.fragment_tol,
        options.clustering.min_matched_peaks,
        n_workers,
        memmap_store,
    )
    giants = [
        GiantInterval(
            giant_idx=giant_idx,
            bucket_id=bucket_id,
            dataset=dataset,
            spec=spec,
            memmap_path=path,
            options=options,
        )
        for giant_idx, ((bucket_id, dataset, spec), path) in enumerate(
            zip(raw_giants, memmap_paths, strict=True)
        )
    ]
    # LPT ordering for Pass 2 giant finalization.
    giants.sort(key=lambda giant: giant.spec.cost, reverse=True)
    return giants, tile_tasks


def generate_clusters_multi(
    bucket_datasets: Sequence[lance.LanceDataset],
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = DEFAULT_MIN_MZ,
    max_mz: float = DEFAULT_MAX_MZ,
    bucket_labels: Sequence[str] | None = None,
    compute_representatives: bool = True,
    temp_dir: PathLike | None = None,
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
    temp_dir : PathLike | None, default=None
        Optional directory for temporary distance memmaps; defaults to
        Python's system temporary directory.

    Returns
    -------
    list[ClusterResult]
        Per-dataset ``ClusterResult(labels, representatives)`` results
        in the same order as ``bucket_datasets``.

    Raises
    ------
    ValueError
        If ``bucket_labels`` is supplied with a length different from
        ``bucket_datasets``.
    """
    if bucket_labels is not None:
        labels_list = list(bucket_labels)
        if len(labels_list) != len(bucket_datasets):
            raise ValueError(
                f"bucket_labels length ({len(labels_list)}) must match "
                f"bucket_datasets length ({len(bucket_datasets)})"
            )
    else:
        labels_list = [
            bucket_label_from_uri(ds.uri, str(i))
            for i, ds in enumerate(bucket_datasets)
        ]

    # Absolutize temp_dir so warm loky workers do not resolve relative
    # paths against a stale working directory after os.chdir().
    temp_dir_str = os.path.abspath(temp_dir) if temp_dir is not None else None
    options = ClusteringOptions(
        clustering=clustering,
        consensus=(
            MedoidConsensusConfig() if consensus is None else consensus
        ),
        min_mz=min_mz,
        max_mz=max_mz,
        compute_representatives=compute_representatives,
        temp_dir=temp_dir_str,
    )
    logger.debug(
        "Hierarchical clustering (linkage=%s, distance_threshold=%.4f, "
        "min_samples=%d)",
        clustering.linkage,
        clustering.distance_threshold,
        MIN_CLUSTER_SAMPLES,
    )

    n_workers = effective_n_workers()
    plans, chunk_tasks, raw_giants = _plan_buckets(
        bucket_datasets, labels_list, clustering, n_workers
    )

    with GiantMemmapStore(temp_dir=temp_dir_str) as memmap_store:
        giants, tile_tasks = _prepare_giants(
            raw_giants, options, n_workers, memmap_store
        )
        total_spectra = sum(plan.n_spectra for plan in plans)
        results_by_bucket = _run_pass1(
            chunk_tasks,
            tile_tasks,
            options,
            n_buckets=len(bucket_datasets),
            n_workers=n_workers,
            total_spectra=total_spectra,
        )
        _run_pass2(giants, memmap_store, results_by_bucket, n_workers)

    return [
        finalize_bucket(
            plan.n_spectra,
            plan.splits,
            results_by_bucket.get(bucket_id, []),
            labels_list[bucket_id],
        )
        for bucket_id, plan in enumerate(plans)
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
    temp_dir: PathLike | None = None,
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
    temp_dir : PathLike | None, default=None
        Optional directory for temporary distance memmaps; defaults to
        Python's system temporary directory.

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
        temp_dir=temp_dir,
    )[0]
