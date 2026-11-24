"""Interval splitting, LPT scheduling, tiling, and clustering."""

from __future__ import annotations

import contextlib
import gc
import heapq
import logging
import math
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Self

import fastcluster
import joblib
import numba as nb
import numba.typed
import numpy as np
import pyarrow as pa

from ..config import MedoidConsensusConfig
from ..staging import (
    CANONICAL_SORT_COLUMNS,
    SORTED_INTERVAL_COLUMNS,
    bucket_label_from_uri,
    read_canonical_metadata,
)
from ..types import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    ClusterResult,
    ConsensusTuple,
    SpectrumTuple,
    offset_cluster_result,
)
from ..utils import (
    effective_memory_bytes,
    effective_n_workers,
    format_hms,
    loky_unordered_pool,
)
from .consensus import get_representative_spectra
from .distance import (
    close_memmap,
    compute_condensed_distance_matrix_csr,
    extract_arrow_csr,
    fill_condensed_rows_csr,
    open_shared_condensed,
)
from .refine import cut_linkage, refine_clusters

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import lance

    from ..config import ClusteringConfig, ConsensusConfig
    from ..types import PathLike

logger = logging.getLogger(__name__)

MIN_CLUSTER_SAMPLES = 2
CHUNKS_PER_WORKER = 8
MIN_PAIRS_PER_TILE = 4_000_000

_CHUNK_NO_REP_COLUMNS: tuple[str, ...] = (
    "precursor_mz",
    "retention_time",
    "mz",
    "intensity",
)
_GIANT_METADATA_COLUMNS: tuple[str, ...] = ("precursor_mz", "retention_time")
_TILE_PEAK_COLUMNS: tuple[str, ...] = ("mz", "intensity")

# Estimated bytes per condensed pair during Pass 2 giant linkage:
# memmap page cache (4 B) + fastcluster float32 copy (4 B) + float64
# working copy (8 B), reserving 30% headroom for OS and workers.
_PASS2_BYTES_PER_PAIR = 16
_PASS2_RAM_BUDGET_FRACTION = 0.7

TaggedIntervalResult = tuple[int, ClusterResult]

__all__ = [
    "CHUNKS_PER_WORKER",
    "MIN_CLUSTER_SAMPLES",
    "MIN_PAIRS_PER_TILE",
    "ClusteringOptions",
    "GiantMemmapStore",
    "IntervalSpec",
    "TileTask",
    "cluster_arrow_interval",
    "cluster_chunk",
    "cost_based_chunking",
    "effective_tile_threshold",
    "finalize_bucket",
    "generate_clusters",
    "generate_clusters_multi",
    "interval_row_bounds",
    "plan_tile_bounds",
    "precursor_mz_splits",
    "prepare_bucket_chunks",
    "tile_distance",
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


@dataclass(frozen=True, slots=True)
class IntervalSpec:
    """One disjoint precursor m/z interval within a charge bucket."""

    task_id: int
    row_ids: np.ndarray

    @property
    def n_spectra(self) -> int:
        """Return the number of spectra in this interval."""
        return int(self.row_ids.shape[0])

    @property
    def cost(self) -> int:
        """Return estimated quadratic pairwise distance cost."""
        n = self.n_spectra
        return n * n


@dataclass(frozen=True, slots=True)
class _ChunkTask:
    bucket_id: int
    cost: int
    n_spectra: int
    dataset: lance.LanceDataset
    intervals: tuple[IntervalSpec, ...]


@dataclass(frozen=True, slots=True)
class TileTask:
    """Single row-band distance tile for a giant interval."""

    giant_idx: int
    dataset: lance.LanceDataset
    row_ids: np.ndarray
    r0: int
    r1: int
    memmap_path: str
    tile_cost: float
    tile_spectra: float
    fragment_tol: float
    min_matched_peaks: int


@dataclass(frozen=True, slots=True)
class _GiantInterval:
    giant_idx: int
    bucket_id: int
    dataset: lance.LanceDataset
    spec: IntervalSpec
    memmap_path: str
    options: ClusteringOptions


# ----------------------------------------------------------------------
# Precursor splitting, LPT chunking, and tile planning
# ----------------------------------------------------------------------


def effective_tile_threshold(batch_size: int, n_workers: int) -> int:
    """Return minimum interval size that justifies at least 2 tiles."""
    if n_workers <= 1:
        return batch_size + 1
    min_two_tile = (math.isqrt(16 * MIN_PAIRS_PER_TILE) + 3) // 2
    return max(2, batch_size // 2, min_two_tile)


@nb.njit(cache=True)
def _append_even_splits(
    splits: nb.typed.List, block_end: int, batch_size: int
) -> None:
    block_start = splits[-1]
    block_size = block_end - block_start
    if block_size <= batch_size:
        splits.append(block_end)
        return
    n_chunks = (block_size + batch_size - 1) // batch_size
    chunk_size = block_size // n_chunks
    rem = block_size % n_chunks
    pos = block_start
    for k in range(n_chunks):
        pos += chunk_size + (1 if k < rem else 0)
        splits.append(pos)


@nb.njit(cache=True)
def _precursor_mz_splits_nb(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    mode_is_da: bool,
    batch_size: int,
) -> np.ndarray:
    n = len(precursor_mzs)
    splits = nb.typed.List([0])
    for i in range(1, n):
        mz_curr = precursor_mzs[i]
        mz_prev = precursor_mzs[i - 1]
        diff = (
            mz_curr - mz_prev
            if mode_is_da
            else (mz_curr - mz_prev) / mz_prev * 1e6
        )
        if diff > precursor_tol_mass:
            _append_even_splits(splits, i, batch_size)
    if splits[-1] != n:
        _append_even_splits(splits, n, batch_size)
    out = np.empty(len(splits), dtype=np.int64)
    for k in range(len(splits)):
        out[k] = splits[k]
    return out


def precursor_mz_splits(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
) -> np.ndarray:
    """
    Split sorted precursor m/z values into disjoint clustering blocks.

    Partitions at adjacent gaps exceeding ``precursor_tol_mass`` and
    subdivides blocks larger than ``batch_size`` evenly (differing in
    length by at most 1).
    """
    return _precursor_mz_splits_nb(
        np.asarray(precursor_mzs, dtype=np.float64),
        precursor_tol_mass,
        precursor_tol_mode == "Da",
        batch_size,
    )


def cost_based_chunking(
    intervals: Sequence[IntervalSpec], num_chunks: int
) -> list[list[IntervalSpec]]:
    """Distribute interval tasks across chunks by LPT quadratic cost."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda spec: spec.cost, reverse=True)
    chunks: list[list[IntervalSpec]] = [[] for _ in range(num_chunks)]
    heap: list[tuple[int, int]] = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    for spec in ordered:
        current_cost, chunk_idx = heapq.heappop(heap)
        chunks[chunk_idx].append(spec)
        heapq.heappush(heap, (current_cost + spec.cost, chunk_idx))
    return [chunk for chunk in chunks if chunk]


def interval_row_bounds(n: int, n_tiles: int) -> list[tuple[int, int]]:
    """Partition rows ``[0, n)`` into pair-balanced tiles."""
    cum = np.zeros(n, dtype=np.int64)
    cum[1:] = np.cumsum(n - 1 - np.arange(n - 1, dtype=np.int64))
    total_pairs = int(cum[-1])
    edges = [
        edge
        for p in range(1, n_tiles)
        if 0
        < (edge := int(np.searchsorted(cum, total_pairs * p / n_tiles)))
        < n - 1
    ]
    bounds = sorted({0, *edges, n})
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def plan_tile_bounds(n: int, n_workers: int) -> list[tuple[int, int, float]]:
    """Plan pair-balanced ``(r0, r1, tile_cost)`` bands for a giant."""
    total_pairs = n * (n - 1) // 2
    n_tiles = min(n_workers, max(1, total_pairs // MIN_PAIRS_PER_TILE))
    interval_cost = n * n
    return [
        (
            r0,
            r1,
            interval_cost
            * ((2 * n - 1 - r0 - r1) * (r1 - r0) / 2)
            / total_pairs,
        )
        for r0, r1 in interval_row_bounds(n, n_tiles)
    ]


def prepare_bucket_chunks(
    dataset: lance.LanceDataset,
    bucket_label: str,
    clustering: ClusteringConfig,
    num_chunks: int,
    threshold: int,
) -> tuple[int, np.ndarray, list[list[IntervalSpec]], list[IntervalSpec]]:
    """Load bucket metadata and split into normal chunks and giants."""
    data = read_canonical_metadata(
        dataset, CANONICAL_SORT_COLUMNS, with_row_index=True
    )
    n = data.shape[0]
    logger.info("Cluster %d spectra with charge %s", n, bucket_label)
    idx = data["index"].to_numpy(dtype=np.int64)
    splits = precursor_mz_splits(
        data["precursor_mz"].to_numpy(),
        clustering.precursor_tol.mass,
        clustering.precursor_tol.mode,
        clustering.batch_size,
    )

    normal_specs: list[IntervalSpec] = []
    giant_specs: list[IntervalSpec] = []
    for task_id in range(len(splits) - 1):
        spec = IntervalSpec(
            task_id=task_id,
            row_ids=idx[
                int(splits[task_id]) : int(splits[task_id + 1])
            ].copy(),
        )
        if spec.n_spectra < threshold:
            normal_specs.append(spec)
        else:
            giant_specs.append(spec)
    return (
        n,
        splits,
        cost_based_chunking(normal_specs, num_chunks),
        giant_specs,
    )


# ----------------------------------------------------------------------
# Single-interval clustering and giant memmap tiling
# ----------------------------------------------------------------------


class GiantMemmapStore:
    """Manage lifecycle and eager unlinking of giant memmap files."""

    def __init__(self, temp_dir: PathLike | None = None) -> None:
        self._temp_dir = temp_dir
        self._paths: dict[int, str] = {}

    def allocate(self, giant_idx: int, n_spectra: int) -> str:
        """Allocate a shared condensed memmap for ``giant_idx``."""
        path, mm = open_shared_condensed(n_spectra, temp_dir=self._temp_dir)
        self._paths[giant_idx] = path
        close_memmap(mm)
        return path

    def release(self, giant_idx: int) -> None:
        """Unlink and forget the backing file for ``giant_idx``."""
        path = self._paths.pop(giant_idx, None)
        if path is not None:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def __enter__(self) -> Self:
        """Enter the managed memmap store context."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Unlink all remaining tracked memmaps."""
        for giant_idx in list(self._paths):
            self.release(giant_idx)


def tile_distance(task: TileTask) -> tuple[str, int, float, float]:
    """Fill rows ``[task.r0, task.r1)`` of a giant condensed memmap."""
    table = task.dataset.take(
        indices=pa.array(task.row_ids), columns=list(_TILE_PEAK_COLUMNS)
    )
    mz_flat, intensity_flat, offsets = extract_arrow_csr(table)
    condensed = np.lib.format.open_memmap(task.memmap_path, mode="r+")
    try:
        fill_condensed_rows_csr(
            condensed,
            mz_flat,
            intensity_flat,
            offsets,
            task.r0,
            task.r1,
            task.fragment_tol,
            task.min_matched_peaks,
        )
        condensed.flush()
    finally:
        close_memmap(condensed)
    return ("tile", task.giant_idx, task.tile_cost, task.tile_spectra)


def _build_representatives(
    spec_tuples: Sequence[SpectrumTuple],
    refined_labels: np.ndarray,
    label_order: np.ndarray,
    rts_orig: np.ndarray,
    n_clusters: int,
    options: ClusteringOptions,
    pdist: np.ndarray | None,
) -> list[ConsensusTuple]:
    if not options.compute_representatives:
        return []
    if n_clusters < len(label_order):
        refined_order = np.argsort(refined_labels, kind="stable")
        order_map = label_order[refined_order]
        return get_representative_spectra(
            spec_tuples,
            refined_labels[refined_order],
            rts_orig[order_map],
            order_map,
            options.consensus,
            pdist=pdist,
            min_mz=options.min_mz,
            max_mz=options.max_mz,
            fragment_tol=options.clustering.fragment_tol,
        )
    # Every spectrum became a singleton; refined_labels in label_order
    # are 0..n_spectra-1, emitting ascending cluster_ids directly.
    return [
        ConsensusTuple.from_spectrum(
            spec_tuples[int(idx)],
            float(rts_orig[int(idx)]),
            int(refined_labels[pos]),
        )
        for pos, idx in enumerate(label_order)
    ]


def _arrow_spectrum_tuples(
    table: pa.Table,
    precursor_mzs: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
) -> list[SpectrumTuple]:
    charges = table["precursor_charge"].to_numpy(zero_copy_only=False)
    return [
        SpectrumTuple(
            precursor_mzs[i],
            charges[i],
            mz_flat[offsets[i] : offsets[i + 1]],
            intensity_flat[offsets[i] : offsets[i + 1]],
        )
        for i in range(len(precursor_mzs))
    ]


def cluster_arrow_interval(
    table: pa.Table,
    options: ClusteringOptions,
    *,
    pdist: np.ndarray | None = None,
) -> ClusterResult:
    """
    Cluster one canonically ordered Arrow interval.

    Streams ``mz`` and ``intensity`` directly from PyArrow list buffers
    into CSR distance computation and only materializes
    ``SpectrumTuple`` views when ``options.compute_representatives`` is
    ``True``.
    """
    precursor_mzs = table["precursor_mz"].to_numpy(zero_copy_only=False)
    rts_orig = table["retention_time"].to_numpy(zero_copy_only=False)
    n_spectra = len(precursor_mzs)
    if n_spectra == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), [])

    csr = (
        extract_arrow_csr(table)
        if (pdist is None and n_spectra >= 2)
        or options.compute_representatives
        else None
    )
    spec_tuples = (
        _arrow_spectrum_tuples(table, precursor_mzs, *csr)
        if options.compute_representatives and csr is not None
        else []
    )
    if n_spectra == 1:
        reps = (
            [
                ConsensusTuple.from_spectrum(
                    spec_tuples[0], float(rts_orig[0]), 0
                )
            ]
            if options.compute_representatives
            else []
        )
        return ClusterResult(np.zeros(1, dtype=np.int32), reps)

    clustering = options.clustering
    owns_pdist = pdist is None
    if pdist is None and csr is not None:
        pdist = compute_condensed_distance_matrix_csr(
            *csr,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
            temp_dir=options.temp_dir,
        )

    needs_pdist_for_reps = (
        options.compute_representatives
        and options.consensus.method == "medoid"
    )
    try:
        linkage_mat = fastcluster.linkage(pdist, clustering.linkage)
        if owns_pdist and not needs_pdist_for_reps:
            if isinstance(pdist, np.memmap):
                close_memmap(pdist)
                gc.collect()
            pdist = None

        labels = cut_linkage(linkage_mat, clustering.distance_threshold)
        refined_labels, label_order, n_clusters = refine_clusters(
            labels, precursor_mzs, rts_orig, clustering, MIN_CLUSTER_SAMPLES
        )
        cluster_labels = np.empty(n_spectra, dtype=np.int32)
        cluster_labels[label_order] = refined_labels
        reps = _build_representatives(
            spec_tuples,
            refined_labels,
            label_order,
            rts_orig,
            n_clusters,
            options,
            pdist,
        )
    finally:
        if owns_pdist and isinstance(pdist, np.memmap):
            close_memmap(pdist)
            pdist = None
            gc.collect()
    return ClusterResult(cluster_labels, reps)


# ----------------------------------------------------------------------
# Chunk execution, Pass 1/Pass 2 coordination, and bucket assembly
# ----------------------------------------------------------------------


def cluster_chunk(
    chunk: Sequence[IntervalSpec],
    dataset: lance.LanceDataset,
    options: ClusteringOptions,
    *,
    read_batch_size: int | None = None,
) -> list[TaggedIntervalResult]:
    """Cluster all intervals in a chunk with batched Lance reads."""
    if not chunk:
        return []
    columns = list(
        SORTED_INTERVAL_COLUMNS
        if options.compute_representatives
        else _CHUNK_NO_REP_COLUMNS
    )

    def _run_group(group: list[IntervalSpec]) -> list[TaggedIntervalResult]:
        row_ids = np.concatenate([spec.row_ids for spec in group])
        table = dataset.take(indices=pa.array(row_ids), columns=columns)
        out: list[TaggedIntervalResult] = []
        offset = 0
        for spec in group:
            n_rows = spec.n_spectra
            out.append(
                (
                    spec.task_id,
                    cluster_arrow_interval(
                        table.slice(offset, n_rows), options
                    ),
                )
            )
            offset += n_rows
        return out

    results: list[TaggedIntervalResult] = []
    group: list[IntervalSpec] = []
    group_size = 0
    for spec in chunk:
        if (
            group
            and read_batch_size is not None
            and group_size + spec.n_spectra > read_batch_size
        ):
            results.extend(_run_group(group))
            group, group_size = [], 0
        group.append(spec)
        group_size += spec.n_spectra
    results.extend(_run_group(group))
    return results


def _run_chunk_task(
    task: _ChunkTask, options: ClusteringOptions
) -> tuple[str, int, int, int, list[TaggedIntervalResult]]:
    chunk_results = cluster_chunk(
        task.intervals,
        task.dataset,
        options,
        read_batch_size=options.clustering.batch_size,
    )
    return ("chunk", task.bucket_id, task.cost, task.n_spectra, chunk_results)


def _finalize_giant(
    giant: _GiantInterval,
) -> tuple[int, int, int, ClusterResult]:
    columns = list(
        SORTED_INTERVAL_COLUMNS
        if giant.options.compute_representatives
        else _GIANT_METADATA_COLUMNS
    )
    table = giant.dataset.take(
        indices=pa.array(giant.spec.row_ids), columns=columns
    )
    pdist = np.lib.format.open_memmap(giant.memmap_path, mode="r")
    try:
        result = cluster_arrow_interval(table, giant.options, pdist=pdist)
    finally:
        close_memmap(pdist)
    return (giant.giant_idx, giant.bucket_id, giant.spec.task_id, result)


def finalize_bucket(
    n: int,
    splits: Sequence[int] | np.ndarray,
    interval_results: Sequence[TaggedIntervalResult],
    bucket_label: str,
) -> ClusterResult:
    """Assemble split results into bucket-wide labels and consensus."""
    splits_arr = np.asarray(splits, dtype=np.int64)
    cluster_labels = np.full(n, -1, dtype=np.int32)
    representative_spectra: list[ConsensusTuple] = []
    current_offset = 0

    for task_id, interval_result in sorted(
        interval_results, key=lambda item: item[0]
    ):
        start, stop = int(splits_arr[task_id]), int(splits_arr[task_id + 1])
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


def _pass2_worker_limit(
    giants: Sequence[_GiantInterval], n_workers: int
) -> int:
    max_workers = min(max(1, n_workers), len(giants))
    if max_workers <= 1:
        return max(1, max_workers)
    total_ram = effective_memory_bytes()
    if total_ram is None:
        return max_workers
    budget = int(total_ram * _PASS2_RAM_BUDGET_FRACTION)
    max_n = max(g.spec.n_spectra for g in giants)
    per_giant = max(1, _PASS2_BYTES_PER_PAIR * max_n * (max_n - 1) // 2)
    return min(max_workers, max(1, budget // per_giant))


def _run_pass1(
    chunk_tasks: Sequence[_ChunkTask],
    tile_tasks: Sequence[TileTask],
    options: ClusteringOptions,
    *,
    n_buckets: int,
    n_workers: int,
    total_spectra: int,
) -> dict[int, list[TaggedIntervalResult]]:
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
    run_chunk = partial(_run_chunk_task, options=options)

    pass1_tasks: list[tuple[float, _ChunkTask | TileTask]] = sorted(
        [
            *((tile.tile_cost, tile) for tile in tile_tasks),
            *((float(task.cost), task) for task in chunk_tasks),
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    def _jobs() -> Iterator[Any]:
        for _cost, item in pass1_tasks:
            if isinstance(item, TileTask):
                yield joblib.delayed(tile_distance)(item)
            else:
                yield joblib.delayed(run_chunk)(item)

    done_tasks = 0
    done_cost = 0.0
    done_spectra = 0.0
    next_log_pct = 5
    start_time = time.monotonic()

    done_gen: Any = loky_unordered_pool(pool_workers)(_jobs())
    with contextlib.closing(done_gen) as done_iter:
        for done in done_iter:
            done_tasks += 1
            if done[0] == "chunk":
                _, bucket_id, cost, n_spec, chunk_res = done
                results_by_bucket.setdefault(bucket_id, []).extend(chunk_res)
                done_cost += cost
                done_spectra += n_spec
            else:
                _, _giant_idx, tile_cost, tile_spec = done
                done_cost += tile_cost
                done_spectra += tile_spec

            pct = (
                min(100.0, 100.0 * done_cost / total_cost)
                if total_cost
                else 100.0
            )
            if pct >= next_log_pct or done_tasks == n_pass1:
                elapsed = time.monotonic() - start_time
                eta = (
                    elapsed * max(0.0, total_cost - done_cost) / done_cost
                    if done_cost
                    else 0.0
                )
                logger.info(
                    "Clustering progress: %3.0f%% | %d/%d tasks | "
                    "%d/%d spectra processed | elapsed %s | ETA ~%s",
                    pct,
                    done_tasks,
                    n_pass1,
                    min(total_spectra, round(done_spectra)),
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
    temp_dir: PathLike | None = None,
) -> list[ClusterResult]:
    """
    Cluster multiple per-charge datasets in one shared worker pool.

    Normal intervals and giant-interval distance tiles share a single
    Pass 1 pool in joint LPT order; giant intervals finalize in a
    RAM-bounded Pass 2 phase.
    """
    labels_list = (
        list(bucket_labels)
        if bucket_labels is not None
        else [
            bucket_label_from_uri(ds.uri, str(i))
            for i, ds in enumerate(bucket_datasets)
        ]
    )
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
    num_chunks = CHUNKS_PER_WORKER * n_workers
    threshold = effective_tile_threshold(clustering.batch_size, n_workers)

    bucket_meta: list[tuple[int, np.ndarray]] = []
    chunk_tasks: list[_ChunkTask] = []
    raw_giants: list[tuple[int, lance.LanceDataset, IntervalSpec]] = []
    for bucket_id, (dataset, label) in enumerate(
        zip(bucket_datasets, labels_list, strict=True)
    ):
        n, splits, normal_chunks, giant_specs = prepare_bucket_chunks(
            dataset, label, clustering, num_chunks, threshold
        )
        bucket_meta.append((n, splits))
        for specs in normal_chunks:
            chunk_tasks.append(
                _ChunkTask(
                    bucket_id=bucket_id,
                    cost=sum(spec.cost for spec in specs),
                    n_spectra=sum(spec.n_spectra for spec in specs),
                    dataset=dataset,
                    intervals=tuple(specs),
                )
            )
        for spec in giant_specs:
            raw_giants.append((bucket_id, dataset, spec))

    with GiantMemmapStore(temp_dir=temp_dir_str) as memmap_store:
        giants: list[_GiantInterval] = []
        tile_tasks: list[TileTask] = []
        for giant_idx, (bucket_id, dataset, spec) in enumerate(raw_giants):
            n = spec.n_spectra
            path = memmap_store.allocate(giant_idx, n)
            giants.append(
                _GiantInterval(
                    giant_idx=giant_idx,
                    bucket_id=bucket_id,
                    dataset=dataset,
                    spec=spec,
                    memmap_path=path,
                    options=options,
                )
            )
            for r0, r1, tile_cost in plan_tile_bounds(n, n_workers):
                tile_tasks.append(
                    TileTask(
                        giant_idx=giant_idx,
                        dataset=dataset,
                        row_ids=spec.row_ids,
                        r0=r0,
                        r1=r1,
                        memmap_path=path,
                        tile_cost=tile_cost,
                        tile_spectra=tile_cost / n,
                        fragment_tol=clustering.fragment_tol,
                        min_matched_peaks=clustering.min_matched_peaks,
                    )
                )
        giants.sort(key=lambda g: g.spec.cost, reverse=True)

        results_by_bucket = _run_pass1(
            chunk_tasks,
            tile_tasks,
            options,
            n_buckets=len(bucket_datasets),
            n_workers=n_workers,
            total_spectra=sum(n for n, _ in bucket_meta),
        )

        if giants:
            pass2_workers = _pass2_worker_limit(giants, n_workers)
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_ctx: contextlib.AbstractContextManager[
                Iterable[tuple[int, int, int, ClusterResult]]
            ]
            if pass2_workers <= 1:
                finalize_ctx = contextlib.nullcontext(
                    _finalize_giant(giant) for giant in giants
                )
            else:
                finalize_gen: Any = loky_unordered_pool(pass2_workers)(
                    joblib.delayed(_finalize_giant)(giant) for giant in giants
                )
                finalize_ctx = contextlib.closing(finalize_gen)
            with finalize_ctx as finalize_results:
                for giant_idx, bucket_id, task_id, result in finalize_results:
                    results_by_bucket.setdefault(bucket_id, []).append(
                        (task_id, result)
                    )
                    memmap_store.release(giant_idx)

    return [
        finalize_bucket(
            n,
            splits,
            results_by_bucket.get(bucket_id, []),
            labels_list[bucket_id],
        )
        for bucket_id, (n, splits) in enumerate(bucket_meta)
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
    """Cluster spectra in a Lance dataset via hierarchical linkage."""
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
