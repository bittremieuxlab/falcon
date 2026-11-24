"""Hierarchical spectrum clustering engine and worker coordinator."""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fastcluster
import joblib
import numpy as np
import pyarrow as pa

from ..config.models import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    MedoidConsensusConfig,
)
from ..dataset import (
    CANONICAL_SORT_COLUMNS,
    SORTED_INTERVAL_COLUMNS,
    bucket_label_from_uri,
)
from ..types import (
    ClusterResult,
    ConsensusTuple,
    SpectrumTuple,
    offset_cluster_result,
)
from ..utils import effective_n_workers, format_hms
from . import scheduling
from .consensus import get_representative_spectra
from .distance import close_memmap, compute_condensed_distance_matrix
from .refine import cut_linkage, refine_clusters
from .scheduling import (
    CHUNKS_PER_WORKER,
    prepare_bucket_chunks,
    tile_threshold,
)
from .tiling import (
    GiantMemmapStore,
    TileTask,
    allocate_giant_tiles,
    tile_distance,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import lance
    import pandas as pd

    from ..config.models import ClusteringConfig, ConsensusConfig
    from .scheduling import BucketPlan, IntervalSpec

logger = logging.getLogger(__name__)

TaggedIntervalResult = tuple[int, ClusterResult]

# Minimum spectra a refined cluster needs to keep a shared label;
# smaller groups become individually labeled singletons.
_MIN_CLUSTER_SAMPLES = 2

# Run explicit gc.collect() only after intervals larger than this size;
# smaller intervals do not allocate enough to justify GC overhead.
_GC_COLLECT_MIN_SPECTRA = 2**11

_GIANT_METADATA_COLUMNS: tuple[str, ...] = (
    "precursor_mz",
    "retention_time",
)

# Estimated bytes per condensed pair during Pass 2 giant linkage:
# memmap page cache (4 B) plus fastcluster's float32 copy (4 B) and
# float64 working copy (8 B). Reserve 30% headroom for OS/workers.
_PASS2_BYTES_PER_PAIR = 16
_PASS2_RAM_BUDGET_FRACTION = 0.7
_PROC_CGROUP_PATH = "/proc/self/cgroup"
_CGROUP_V2_ROOT = "/sys/fs/cgroup"
_CGROUP_V1_MEMORY_ROOT = "/sys/fs/cgroup/memory"
_CGROUP_MEMORY_PATHS: tuple[str, ...] = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)

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
class ClusteringOptions:
    """Bundled clustering, consensus, and m/z-range options."""

    clustering: ClusteringConfig
    consensus: ConsensusConfig = field(default_factory=MedoidConsensusConfig)
    min_mz: float = DEFAULT_MIN_MZ
    max_mz: float = DEFAULT_MAX_MZ
    compute_representatives: bool = True
    temp_dir: str | None = None


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


def _parallel_pool(n_jobs: int) -> joblib.Parallel:
    """Create a loky pool of ``n_jobs`` returning unordered results."""
    return joblib.Parallel(
        n_jobs=max(1, n_jobs),
        backend="loky",
        verbose=0,
        return_as="generator_unordered",
    )


def _effective_tile_threshold(batch_size: int, n_workers: int) -> int:
    """Return minimum interval size that splits into >= 2 tiles."""
    if n_workers <= 1:
        return batch_size + 1
    # Exact smallest n with n * (n - 1) // 2 >= 2 * MIN_PAIRS_PER_TILE.
    min_two_tile = (math.isqrt(16 * scheduling.MIN_PAIRS_PER_TILE) + 3) // 2
    return max(tile_threshold(batch_size), min_two_tile)


def _read_positive_int(path: Path) -> int | None:
    """Read a positive decimal integer from ``path``, or ``None``."""
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if raw.isdigit():
        value = int(raw)
        if value > 0:
            return value
    return None


def _walk_cgroup_limits(root: Path, rel_path: str, filename: str) -> list[int]:
    """Collect finite limits from ``root / rel_path`` up to ``root``."""
    parts = [p for p in rel_path.split("/") if p and p not in (".", "..")]
    current = root.joinpath(*parts)
    limits: list[int] = []
    while True:
        limit = _read_positive_int(current / filename)
        if limit is not None:
            limits.append(limit)
        if current == root:
            break
        current = current.parent
    return limits


def _cgroup_memory_limit() -> int | None:
    """Return minimum finite Linux cgroup memory limit in bytes."""
    limits: list[int] = []
    try:
        proc_text = Path(_PROC_CGROUP_PATH).read_text(
            encoding="utf-8", errors="surrogateescape"
        )
    except (OSError, UnicodeError):
        proc_text = ""
    for line in proc_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, rel_path = fields
        if hierarchy == "0" and controllers == "":
            limits.extend(
                _walk_cgroup_limits(
                    Path(_CGROUP_V2_ROOT), rel_path, "memory.max"
                )
            )
        elif "memory" in controllers.split(","):
            limits.extend(
                _walk_cgroup_limits(
                    Path(_CGROUP_V1_MEMORY_ROOT),
                    rel_path,
                    "memory.limit_in_bytes",
                )
            )
    for fallback in _CGROUP_MEMORY_PATHS:
        limit = _read_positive_int(Path(fallback))
        if limit is not None:
            limits.append(limit)
    return min(limits) if limits else None


def _pass2_worker_limit(
    giants: Sequence[GiantInterval], n_workers: int
) -> int:
    """Cap concurrent Pass 2 giant workers against effective RAM."""
    max_workers = min(max(1, n_workers), len(giants))
    if max_workers <= 1:
        return max(1, max_workers)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return max_workers
    if page_size <= 0 or phys_pages <= 0:
        return max_workers
    total_ram = int(page_size) * int(phys_pages)
    cgroup_limit = _cgroup_memory_limit()
    if cgroup_limit is not None:
        total_ram = min(total_ram, cgroup_limit)
    budget_bytes = int(total_ram * _PASS2_RAM_BUDGET_FRACTION)
    max_n = max(giant.spec.n_spectra for giant in giants)
    per_giant_bytes = max(1, _PASS2_BYTES_PER_PAIR * max_n * (max_n - 1) // 2)
    return min(max_workers, max(1, budget_bytes // per_giant_bytes))


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
    n_spectra = len(precursor_mzs)

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
        pdist = compute_condensed_distance_matrix(
            spec_tuples,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
            temp_dir=options.temp_dir,
        )
    else:
        expected_pairs = n_spectra * (n_spectra - 1) // 2
        if pdist.ndim != 1 or pdist.shape[0] != expected_pairs:
            raise ValueError(
                f"pdist must be 1D of length {expected_pairs} for "
                f"{n_spectra} spectra, got shape {pdist.shape}"
            )
    try:
        labels = cut_linkage(
            fastcluster.linkage(pdist, clustering.linkage),
            clustering.distance_threshold,
        )

        refined_labels, label_order, n_clusters = refine_clusters(
            labels,
            precursor_mzs,
            rts_orig,
            clustering,
            _MIN_CLUSTER_SAMPLES,
        )

        cluster_labels = np.empty(n_spectra, dtype=np.int32)
        if not options.compute_representatives:
            cluster_labels[label_order] = refined_labels
            rep_spectra: list[ConsensusTuple] = []
        elif n_clusters < n_spectra:
            refined_order = np.argsort(refined_labels, kind="stable")
            sorted_labels = refined_labels[refined_order]
            order_map = label_order[refined_order]
            rts_ref = rts_orig[order_map]
            rep_spectra = get_representative_spectra(
                spec_tuples,
                sorted_labels,
                rts_ref,
                order_map,
                options.consensus,
                pdist=pdist,
                min_mz=options.min_mz,
                max_mz=options.max_mz,
                fragment_tol=clustering.fragment_tol,
            )
            cluster_labels[order_map] = sorted_labels
        else:
            # Every spectrum became a singleton; refined_labels in
            # label_order are 0..n_spectra-1, so iterating pos emits
            # representatives ordered by ascending cluster_id.
            cluster_labels[label_order] = refined_labels
            rep_spectra = [
                ConsensusTuple.from_spectrum(
                    spec_tuples[int(idx)],
                    float(rts_orig[int(idx)]),
                    int(refined_labels[pos]),
                )
                for pos, idx in enumerate(label_order)
            ]
    finally:
        # Release owned distance storage eagerly on large intervals.
        if owns_pdist:
            if isinstance(pdist, np.memmap):
                close_memmap(pdist)
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
    columns = list(SORTED_INTERVAL_COLUMNS)

    def _cluster_group(
        group: list[IntervalSpec],
    ) -> list[TaggedIntervalResult]:
        # Batch Lance reads across small intervals; `take` preserves
        # row_ids order (already canonical) so slices stay contiguous.
        group_row_ids = np.concatenate([spec.row_ids for spec in group])
        group_spectra = dataset.take(
            indices=pa.array(group_row_ids), columns=columns
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
    spectra = giant.dataset.take(
        indices=pa.array(giant.spec.row_ids), columns=columns
    ).to_pandas()
    pdist = np.lib.format.open_memmap(giant.memmap_path, mode="r")
    try:
        result = cluster_mz_interval(
            spectra,
            giant.options,
            pdist=pdist,
            already_sorted=True,
        )
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

    Validates split coverage and computes cumulative label offsets in a
    single sorted pass so labels and representatives are unique across
    the bucket.

    Parameters
    ----------
    n : int
        Total number of spectra in the bucket.
    splits : Sequence[int] | np.ndarray
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

    Raises
    ------
    ValueError
        If ``splits`` bounds do not match ``n`` or ``interval_results``
        task IDs do not cover ``0..len(splits) - 2`` without gaps or
        duplicates.
    """
    if len(splits) == 0 or int(splits[0]) != 0 or int(splits[-1]) != n:
        raise ValueError(
            f"Invalid splits for n={n}: expected bounds starting at 0 "
            f"and ending at {n}"
        )
    n_splits = len(splits) - 1
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

    def _jobs() -> Iterator[Any]:
        for tile in tile_tasks:
            yield joblib.delayed(tile_distance)(tile)
        for task in chunk_tasks:
            yield joblib.delayed(run_chunk)(task)

    done_cost = 0.0
    done_spectra = 0.0
    next_log_pct = 5
    start = time.monotonic()

    done_gen: Any = _parallel_pool(pool_workers)(_jobs())
    with contextlib.closing(done_gen) as done_iter:
        for done_tasks, done in enumerate(done_iter, start=1):
            if isinstance(done, ChunkDone):
                results_by_bucket.setdefault(done.bucket_id, []).extend(
                    done.results
                )
                done_cost += done.cost
                done_spectra += done.n_spectra
            else:
                done_cost += done.tile_cost
                done_spectra += done.tile_spectra
            pct = (
                min(100.0, 100.0 * done_cost / total_cost)
                if total_cost
                else 100.0
            )
            if pct >= next_log_pct or done_tasks == n_pass1:
                elapsed = time.monotonic() - start
                remaining_cost = max(0.0, total_cost - done_cost)
                eta = (
                    elapsed * remaining_cost / done_cost if done_cost else 0.0
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
    temp_dir: os.PathLike[str] | str | None = None,
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
    temp_dir : os.PathLike[str] | str | None, default=None
        Optional directory for temporary distance memmaps; defaults to
        Python's system temporary directory.

    Returns
    -------
    list[ClusterResult]
        Per-dataset ``ClusterResult(labels, representatives)`` results
        in the same order as ``bucket_datasets``.
    """
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
    threshold = _effective_tile_threshold(clustering.batch_size, n_workers)

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

    # Schedule heaviest chunks first so small buckets backfill cleanly.
    chunk_tasks.sort(key=lambda task: task.cost, reverse=True)

    with GiantMemmapStore(temp_dir=temp_dir_str) as memmap_store:
        memmap_paths, tile_tasks = allocate_giant_tiles(
            raw_giants,
            clustering.fragment_tol,
            clustering.min_matched_peaks,
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
        # LPT ordering for giant tiles and Pass 2 giant finalization.
        tile_tasks.sort(key=lambda tile: tile.tile_cost, reverse=True)
        giants.sort(key=lambda giant: giant.spec.cost, reverse=True)

        total_spectra = sum(plan.n_spectra for plan in plans)
        results_by_bucket = _run_pass1(
            chunk_tasks,
            tile_tasks,
            options,
            n_buckets=len(bucket_datasets),
            n_workers=n_workers,
            total_spectra=total_spectra,
        )

        if giants:
            pass2_workers = _pass2_worker_limit(giants, n_workers)
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_ctx: contextlib.AbstractContextManager[
                Iterable[GiantDone]
            ]
            if len(giants) == 1 or pass2_workers <= 1:
                finalize_ctx = contextlib.nullcontext(
                    finalize_giant(giant) for giant in giants
                )
            else:
                finalize_gen: Any = _parallel_pool(pass2_workers)(
                    joblib.delayed(finalize_giant)(giant) for giant in giants
                )
                finalize_ctx = contextlib.closing(finalize_gen)
            with finalize_ctx as finalize_results:
                for done in finalize_results:
                    results_by_bucket.setdefault(done.bucket_id, []).append(
                        (done.task_id, done.result)
                    )
                    memmap_store.release(done.giant_idx)

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
    temp_dir: os.PathLike[str] | str | None = None,
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
    temp_dir : os.PathLike[str] | str | None, default=None
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
