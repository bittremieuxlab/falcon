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
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import fastcluster
import joblib
import numpy as np
import pyarrow as pa

from ..config import ConsensusConfig, PathLike
from ..staging import (
    BUCKET_PREFIX,
    BUCKET_SUFFIX,
    CANONICAL_SORT_COLUMNS,
    SORTED_INTERVAL_COLUMNS,
    read_canonical_metadata,
)
from .consensus import (
    ConsensusTuple,
    SpectrumTuple,
    get_representative_spectra,
)
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

    from ..config import ClusteringConfig

logger = logging.getLogger(__name__)

MIN_CLUSTER_SAMPLES = 2
CHUNKS_PER_WORKER = 8
MIN_PAIRS_PER_TILE = 4_000_000

_CHUNK_NO_REP_COLUMNS = ("precursor_mz", "retention_time", "mz", "intensity")
_GIANT_METADATA_COLUMNS = ("precursor_mz", "retention_time")
_TILE_PEAK_COLUMNS = ("mz", "intensity")

# Estimated bytes per condensed pair during Pass 2 giant linkage:
# memmap page cache (4 B) + fastcluster float32 copy (4 B) + float64
# working copy (8 B), reserving 30% headroom for OS and workers.
_PASS2_BYTES_PER_PAIR = 16
_PASS2_RAM_BUDGET_FRACTION = 0.7
_CGROUP_MEMORY_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

Interval = tuple[int, np.ndarray]  # (task_id, row_ids)
TaggedResult = tuple[int, "ClusterResult"]
_GiantJob = tuple[int, int, int, Any, np.ndarray, str]


class ClusterResult(NamedTuple):
    """Cluster label assignments and representative spectra."""

    labels: np.ndarray
    representatives: list[ConsensusTuple]


@dataclass(frozen=True, slots=True)
class _ClusterOptions:
    clustering: ClusteringConfig
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    min_mz: float = 101.0
    max_mz: float = 1500.0
    compute_representatives: bool = True
    temp_dir: str | None = None


def offset_cluster_result(
    result: ClusterResult,
    offset: int,
    *,
    include_representatives: bool = True,
) -> tuple[ClusterResult, int]:
    """
    Shift cluster labels and representative IDs by ``offset``.

    Does not mutate ``result.labels`` in place. Returns the shifted
    ``ClusterResult`` and the next available global label offset.
    """
    labels_arr = np.asarray(result.labels, dtype=np.int32)
    if labels_arr.size == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), []), offset

    offset_labels = labels_arr + np.int32(offset)
    offset_reps = (
        [
            rep._replace(cluster_id=np.int32(int(rep.cluster_id) + offset))
            for rep in result.representatives
        ]
        if include_representatives and result.representatives
        else []
    )
    return ClusterResult(offset_labels, offset_reps), int(
        np.max(offset_labels)
    ) + 1


# ----------------------------------------------------------------------
# Precursor splitting, LPT chunking, and tile planning
# ----------------------------------------------------------------------


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
    mzs = np.asarray(precursor_mzs, dtype=np.float64)
    n = len(mzs)
    if n == 0:
        return np.array([0], dtype=np.int64)

    diffs = (
        np.diff(mzs)
        if precursor_tol_mode == "Da"
        else np.diff(mzs) / mzs[:-1] * 1e6
    )
    gap_bounds = (np.flatnonzero(diffs > precursor_tol_mass) + 1).tolist()

    splits = [0]
    for block_end in (*gap_bounds, n):
        block_start = splits[-1]
        block_size = block_end - block_start
        if block_size <= batch_size:
            splits.append(block_end)
        else:
            n_chunks = (block_size + batch_size - 1) // batch_size
            q, rem = divmod(block_size, n_chunks)
            pos = block_start
            for k in range(n_chunks):
                pos += q + (1 if k < rem else 0)
                splits.append(pos)
    return np.array(splits, dtype=np.int64)


def _effective_tile_threshold(batch_size: int, n_workers: int) -> int:
    if n_workers <= 1:
        return batch_size + 1
    min_two_tile = (math.isqrt(16 * MIN_PAIRS_PER_TILE) + 3) // 2
    return max(2, batch_size // 2, min_two_tile)


def _cost_based_chunking(
    intervals: Sequence[Interval], num_chunks: int
) -> list[list[Interval]]:
    if not intervals:
        return []
    ordered = sorted(
        intervals, key=lambda item: len(item[1]) ** 2, reverse=True
    )
    chunks: list[list[Interval]] = [[] for _ in range(num_chunks)]
    heap: list[tuple[int, int]] = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    for interval in ordered:
        current_cost, chunk_idx = heapq.heappop(heap)
        chunks[chunk_idx].append(interval)
        heapq.heappush(heap, (current_cost + len(interval[1]) ** 2, chunk_idx))
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


def _plan_tile_bounds(n: int, n_workers: int) -> list[tuple[int, int, float]]:
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


def _bucket_label_from_uri(uri: str, fallback: str) -> str:
    name = Path(uri).name
    if (
        len(name) > len(BUCKET_PREFIX) + len(BUCKET_SUFFIX)
        and name.startswith(BUCKET_PREFIX)
        and name.endswith(BUCKET_SUFFIX)
    ):
        return name[len(BUCKET_PREFIX) : -len(BUCKET_SUFFIX)].replace(
            "_", ", "
        )
    return fallback


# ----------------------------------------------------------------------
# Interval clustering and worker tasks
# ----------------------------------------------------------------------


def _extract_spec_tuples(
    table: pa.Table,
    precursor_mzs: np.ndarray,
    mz_flat: np.ndarray,
    int_flat: np.ndarray,
    offsets: np.ndarray,
) -> list[SpectrumTuple]:
    charges = table["precursor_charge"].to_numpy(zero_copy_only=False)
    return [
        SpectrumTuple(
            precursor_mzs[i],
            charges[i],
            mz_flat[offsets[i] : offsets[i + 1]],
            int_flat[offsets[i] : offsets[i + 1]],
        )
        for i in range(len(precursor_mzs))
    ]


def cluster_arrow_interval(
    table: pa.Table,
    options: _ClusterOptions,
    *,
    pdist: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster one canonically ordered Arrow interval."""
    precursor_mzs = table["precursor_mz"].to_numpy(zero_copy_only=False)
    rts_orig = table["retention_time"].to_numpy(zero_copy_only=False)
    n_spectra = len(precursor_mzs)
    if n_spectra == 0:
        return ClusterResult(np.empty(0, dtype=np.int32), [])
    if n_spectra == 1:
        reps: list[ConsensusTuple] = []
        if options.compute_representatives:
            specs = _extract_spec_tuples(
                table, precursor_mzs, *extract_arrow_csr(table)
            )
            reps = [
                ConsensusTuple.from_spectrum(specs[0], float(rts_orig[0]), 0)
            ]
        return ClusterResult(np.zeros(1, dtype=np.int32), reps)

    clustering = options.clustering
    owns_pdist = pdist is None
    csr = (
        extract_arrow_csr(table)
        if owns_pdist or options.compute_representatives
        else None
    )
    if owns_pdist and csr is not None:
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
        refined_labels, label_order, _ = refine_clusters(
            labels, precursor_mzs, rts_orig, clustering, MIN_CLUSTER_SAMPLES
        )
        cluster_labels = np.empty(n_spectra, dtype=np.int32)
        cluster_labels[label_order] = refined_labels

        reps = []
        if options.compute_representatives and csr is not None:
            specs = _extract_spec_tuples(table, precursor_mzs, *csr)
            refined_order = np.argsort(refined_labels, kind="stable")
            order_map = label_order[refined_order]
            reps = get_representative_spectra(
                specs,
                refined_labels[refined_order],
                rts_orig[order_map],
                order_map,
                options.consensus,
                pdist=pdist,
                min_mz=options.min_mz,
                max_mz=options.max_mz,
                fragment_tol=clustering.fragment_tol,
            )
    finally:
        if owns_pdist and isinstance(pdist, np.memmap):
            close_memmap(pdist)
            pdist = None
            gc.collect()
    return ClusterResult(cluster_labels, reps)


def _run_chunk(
    bucket_id: int,
    dataset: lance.LanceDataset,
    intervals: Sequence[Interval],
    options: _ClusterOptions,
) -> tuple[str, int, float, float, list[TaggedResult]]:
    columns = list(
        SORTED_INTERVAL_COLUMNS
        if options.compute_representatives
        else _CHUNK_NO_REP_COLUMNS
    )
    read_batch = options.clustering.batch_size
    results: list[TaggedResult] = []
    group: list[Interval] = []
    group_size = 0

    def _flush() -> None:
        row_ids = np.concatenate([ids for _, ids in group])
        table = dataset.take(indices=pa.array(row_ids), columns=columns)
        offset = 0
        for task_id, ids in group:
            n_rows = len(ids)
            results.append(
                (
                    task_id,
                    cluster_arrow_interval(
                        table.slice(offset, n_rows), options
                    ),
                )
            )
            offset += n_rows

    for interval in intervals:
        n_rows = len(interval[1])
        if group and group_size + n_rows > read_batch:
            _flush()
            group, group_size = [], 0
        group.append(interval)
        group_size += n_rows
    if group:
        _flush()

    cost = float(sum(len(ids) ** 2 for _, ids in intervals))
    n_spec = float(sum(len(ids) for _, ids in intervals))
    return ("chunk", bucket_id, cost, n_spec, results)


def _run_tile(
    dataset: lance.LanceDataset,
    row_ids: np.ndarray,
    r0: int,
    r1: int,
    memmap_path: str,
    fragment_tol: float,
    min_matched_peaks: int,
    tile_cost: float,
    tile_spectra: float,
) -> tuple[str, float, float]:
    table = dataset.take(
        indices=pa.array(row_ids), columns=list(_TILE_PEAK_COLUMNS)
    )
    mz_flat, intensity_flat, offsets = extract_arrow_csr(table)
    condensed = np.lib.format.open_memmap(memmap_path, mode="r+")
    try:
        fill_condensed_rows_csr(
            condensed,
            mz_flat,
            intensity_flat,
            offsets,
            r0,
            r1,
            fragment_tol,
            min_matched_peaks,
        )
        condensed.flush()
    finally:
        close_memmap(condensed)
    return ("tile", tile_cost, tile_spectra)


def _finalize_giant(
    giant_idx: int,
    bucket_id: int,
    task_id: int,
    dataset: lance.LanceDataset,
    row_ids: np.ndarray,
    memmap_path: str,
    options: _ClusterOptions,
) -> tuple[int, int, int, ClusterResult]:
    columns = list(
        SORTED_INTERVAL_COLUMNS
        if options.compute_representatives
        else _GIANT_METADATA_COLUMNS
    )
    table = dataset.take(indices=pa.array(row_ids), columns=columns)
    pdist = np.lib.format.open_memmap(memmap_path, mode="r")
    try:
        result = cluster_arrow_interval(table, options, pdist=pdist)
    finally:
        close_memmap(pdist)
    return (giant_idx, bucket_id, task_id, result)


# ----------------------------------------------------------------------
# Pass 1 / Pass 2 orchestration and bucket assembly
# ----------------------------------------------------------------------


def _format_hms(seconds: float) -> str:
    total = int(max(0.0, seconds)) if math.isfinite(seconds) else 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _pass2_worker_limit(giant_sizes: Sequence[int], n_workers: int) -> int:
    max_workers = min(max(1, n_workers), len(giant_sizes))
    if max_workers <= 1:
        return 1
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return max_workers
    if page_size <= 0 or phys_pages <= 0:
        return max_workers
    total_ram = int(page_size) * int(phys_pages)
    for cg_path in _CGROUP_MEMORY_PATHS:
        try:
            raw = Path(cg_path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) > 0:
            total_ram = min(total_ram, int(raw))
    budget = int(total_ram * _PASS2_RAM_BUDGET_FRACTION)
    max_n = max(giant_sizes)
    per_giant = max(1, _PASS2_BYTES_PER_PAIR * max_n * (max_n - 1) // 2)
    return min(max_workers, max(1, budget // per_giant))


def _plan_buckets(
    bucket_datasets: Sequence[lance.LanceDataset],
    labels_list: Sequence[str],
    options: _ClusterOptions,
    n_workers: int,
) -> tuple[
    list[tuple[int, np.ndarray]],
    list[tuple[float, Any]],
    list[tuple[int, lance.LanceDataset, Interval]],
]:
    clustering = options.clustering
    num_chunks = CHUNKS_PER_WORKER * n_workers
    threshold = _effective_tile_threshold(clustering.batch_size, n_workers)
    bucket_meta: list[tuple[int, np.ndarray]] = []
    pass1_jobs: list[tuple[float, Any]] = []
    raw_giants: list[tuple[int, lance.LanceDataset, Interval]] = []

    for bucket_id, (dataset, label) in enumerate(
        zip(bucket_datasets, labels_list, strict=True)
    ):
        data = read_canonical_metadata(
            dataset, CANONICAL_SORT_COLUMNS, with_row_index=True
        )
        n = data.shape[0]
        logger.info("Cluster %d spectra with charge %s", n, label)
        idx = data["index"].to_numpy(dtype=np.int64)
        splits = precursor_mz_splits(
            data["precursor_mz"].to_numpy(),
            clustering.precursor_tol.mass,
            clustering.precursor_tol.mode,
            clustering.batch_size,
        )
        bucket_meta.append((n, splits))

        normal: list[Interval] = []
        for task_id in range(len(splits) - 1):
            row_ids = idx[
                int(splits[task_id]) : int(splits[task_id + 1])
            ].copy()
            if len(row_ids) < threshold:
                normal.append((task_id, row_ids))
            else:
                raw_giants.append((bucket_id, dataset, (task_id, row_ids)))

        for chunk in _cost_based_chunking(normal, num_chunks):
            cost = float(sum(len(ids) ** 2 for _, ids in chunk))
            pass1_jobs.append(
                (
                    cost,
                    joblib.delayed(_run_chunk)(
                        bucket_id, dataset, chunk, options
                    ),
                )
            )
    return bucket_meta, pass1_jobs, raw_giants


def _run_pass1(
    pass1_jobs: Sequence[tuple[float, Any]],
    *,
    n_buckets: int,
    n_giants: int,
    n_workers: int,
    total_spectra: int,
) -> dict[int, list[TaggedResult]]:
    results_by_bucket: dict[int, list[TaggedResult]] = {}
    n_pass1 = len(pass1_jobs)
    if n_pass1 == 0:
        return results_by_bucket

    pool_workers = min(n_workers, n_pass1)
    total_cost = sum(cost for cost, _ in pass1_jobs)
    logger.info(
        "Clustering %d spectra across %d charge bucket(s) in %d task(s) "
        "(%d giant interval(s) tiled) on %d worker(s)...",
        total_spectra,
        n_buckets,
        n_pass1,
        n_giants,
        pool_workers,
    )

    ordered: Iterator[Any] = (
        job for _, job in sorted(pass1_jobs, key=lambda x: x[0], reverse=True)
    )
    done_tasks = 0
    done_cost = 0.0
    done_spectra = 0.0
    next_log_pct = 5
    start_time = time.monotonic()

    done_gen: Any = joblib.Parallel(
        n_jobs=pool_workers,
        backend="loky",
        return_as="generator_unordered",
    )(ordered)
    with contextlib.closing(done_gen) as done_iter:
        for done in done_iter:
            done_tasks += 1
            if done[0] == "chunk":
                _, bucket_id, cost, n_spec, chunk_res = done
                results_by_bucket.setdefault(bucket_id, []).extend(chunk_res)
                done_cost += cost
                done_spectra += n_spec
            else:
                _, tile_cost, tile_spec = done
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
                    _format_hms(elapsed),
                    _format_hms(eta),
                )
                next_log_pct = (int(pct) // 5 + 1) * 5
    return results_by_bucket


def _run_pass2(
    giants: Sequence[_GiantJob],
    giant_paths: dict[int, str],
    results_by_bucket: dict[int, list[TaggedResult]],
    options: _ClusterOptions,
    n_workers: int,
) -> None:
    pass2_workers = _pass2_worker_limit([len(g[4]) for g in giants], n_workers)
    logger.info("Finalizing %d giant interval(s)...", len(giants))
    finalize_ctx: contextlib.AbstractContextManager[
        Iterable[tuple[int, int, int, ClusterResult]]
    ]
    if pass2_workers <= 1:
        finalize_ctx = contextlib.nullcontext(
            _finalize_giant(*g, options) for g in giants
        )
    else:
        finalize_gen: Any = joblib.Parallel(
            n_jobs=pass2_workers,
            backend="loky",
            return_as="generator_unordered",
        )(joblib.delayed(_finalize_giant)(*g, options) for g in giants)
        finalize_ctx = contextlib.closing(finalize_gen)
    with finalize_ctx as finalize_results:
        for giant_idx, bucket_id, task_id, result in finalize_results:
            results_by_bucket.setdefault(bucket_id, []).append(
                (task_id, result)
            )
            path = giant_paths.pop(giant_idx, None)
            if path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(path)


def _finalize_bucket(
    n: int,
    splits: np.ndarray,
    interval_results: Sequence[TaggedResult],
    bucket_label: str,
) -> ClusterResult:
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


def generate_clusters_multi(
    bucket_datasets: Sequence[lance.LanceDataset],
    clustering: ClusteringConfig,
    consensus: ConsensusConfig | None = None,
    *,
    min_mz: float = 101.0,
    max_mz: float = 1500.0,
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
            _bucket_label_from_uri(ds.uri, str(i))
            for i, ds in enumerate(bucket_datasets)
        ]
    )
    temp_dir_str = os.path.abspath(temp_dir) if temp_dir is not None else None
    options = _ClusterOptions(
        clustering=clustering,
        consensus=ConsensusConfig() if consensus is None else consensus,
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

    n_workers = max(1, int(joblib.effective_n_jobs(-1)))
    bucket_meta, pass1_jobs, raw_giants = _plan_buckets(
        bucket_datasets, labels_list, options, n_workers
    )

    giant_paths: dict[int, str] = {}
    try:
        giants: list[_GiantJob] = []
        for giant_idx, (bucket_id, dataset, (task_id, row_ids)) in enumerate(
            raw_giants
        ):
            n = len(row_ids)
            path, mm = open_shared_condensed(n, temp_dir=temp_dir_str)
            giant_paths[giant_idx] = path
            close_memmap(mm)
            giants.append(
                (giant_idx, bucket_id, task_id, dataset, row_ids, path)
            )
            for r0, r1, tile_cost in _plan_tile_bounds(n, n_workers):
                pass1_jobs.append(
                    (
                        tile_cost,
                        joblib.delayed(_run_tile)(
                            dataset,
                            row_ids,
                            r0,
                            r1,
                            path,
                            clustering.fragment_tol,
                            clustering.min_matched_peaks,
                            tile_cost,
                            tile_cost / n,
                        ),
                    )
                )
        giants.sort(key=lambda g: len(g[4]), reverse=True)

        results_by_bucket = _run_pass1(
            pass1_jobs,
            n_buckets=len(bucket_datasets),
            n_giants=len(giants),
            n_workers=n_workers,
            total_spectra=sum(n for n, _ in bucket_meta),
        )
        if giants:
            _run_pass2(
                giants, giant_paths, results_by_bucket, options, n_workers
            )
    finally:
        for path in giant_paths.values():
            with contextlib.suppress(OSError):
                os.unlink(path)

    return [
        _finalize_bucket(
            n,
            splits,
            results_by_bucket.get(bucket_id, []),
            labels_list[bucket_id],
        )
        for bucket_id, (n, splits) in enumerate(bucket_meta)
    ]
