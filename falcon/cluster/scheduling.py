"""Precursor m/z interval splitting, LPT scheduling, and tiling."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba as nb
import numba.typed
import numpy as np

from ..dataset import CANONICAL_SORT_COLUMNS, read_canonical_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance

    from ..config.models import ClusteringConfig

logger = logging.getLogger(__name__)

# Number of cost-balanced m/z-split chunks to create per CPU worker.
# Having several times more tasks than workers lets joblib load-balance
# dynamically instead of stranding cores on the tail of a fixed split.
CHUNKS_PER_WORKER = 8

# Minimum pairs per distance tile. Giant intervals split into at most
# worker_count tiles, never finer than this pair budget so each tile
# amortizes its Lance read over enough cosine distance work.
MIN_PAIRS_PER_TILE = 4_000_000

__all__ = [
    "CHUNKS_PER_WORKER",
    "MIN_PAIRS_PER_TILE",
    "ChunkTask",
    "GiantInterval",
    "IntervalSpec",
    "TileTask",
    "cost_based_chunking",
    "interval_row_bounds",
    "precursor_mz_splits",
    "prepare_bucket_chunks",
    "tile_threshold",
]


@dataclass(frozen=True, slots=True)
class IntervalSpec:
    """One disjoint precursor m/z interval within a charge bucket."""

    task_id: int
    row_ids: list[int]
    precursor_mzs: np.ndarray

    @property
    def n_spectra(self) -> int:
        """Return the number of spectra in this interval."""
        return len(self.row_ids)

    @property
    def cost(self) -> int:
        """Return estimated quadratic pairwise distance cost."""
        n = len(self.row_ids)
        return n * n


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """Cost-balanced bundle of normal intervals for one worker."""

    bucket_id: int
    cost: int
    n_spectra: int
    dataset: lance.LanceDataset
    intervals: tuple[IntervalSpec, ...]


@dataclass(frozen=True, slots=True)
class GiantInterval:
    """Large interval whose distance matrix is built via tiling."""

    bucket_id: int
    dataset: lance.LanceDataset
    spec: IntervalSpec
    memmap_path: str


@dataclass(frozen=True, slots=True)
class TileTask:
    """Single row-band distance tile for a ``GiantInterval``."""

    giant_idx: int
    dataset: lance.LanceDataset
    row_ids: list[int]
    r0: int
    r1: int
    memmap_path: str
    tile_cost: float


def tile_threshold(batch_size: int) -> int:
    """
    Return the minimum interval size that triggers tiled distance work.

    Tracking ``batch_size // 2`` self-scales with ``batch_size`` and
    catches forced-split giants in ``[batch_size / 2, batch_size]``.

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


@nb.njit(cache=True)
def _append_even_splits(
    splits: nb.typed.List, block_end: int, batch_size: int
) -> None:
    """Split ``[splits[-1], block_end)`` evenly into <= batch_size."""
    block_start = splits[-1]
    block_size = block_end - block_start
    if block_size <= batch_size:
        splits.append(block_end)
        return
    n_chunks = math.ceil(block_size / batch_size)
    chunk_size = block_size // n_chunks
    rem = block_size % n_chunks
    pos = block_start
    for k in range(n_chunks):
        pos += chunk_size + (1 if k < rem else 0)
        splits.append(pos)


@nb.njit(cache=True)
def precursor_mz_splits(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
) -> nb.typed.List:
    """
    Split sorted precursor m/z values into disjoint clustering blocks.

    First partitions at adjacent precursor m/z gaps strictly exceeding
    ``precursor_tol_mass``, then divides any contiguous block larger
    than ``batch_size`` into evenly sized intervals of at most
    ``batch_size`` spectra (differing in size by at most 1).

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
    n = len(precursor_mzs)
    splits = nb.typed.List([0])
    mode_is_da = precursor_tol_mode == "Da"
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
    return splits


def cost_based_chunking(
    intervals: Sequence[IntervalSpec], num_chunks: int
) -> list[list[IntervalSpec]]:
    """
    Distribute interval tasks across chunks by quadratic cost.

    Uses greedy longest-processing-time-first (LPT) scheduling with a
    min-heap where each interval's estimated cost is ``n_spectra ** 2``.

    Parameters
    ----------
    intervals : Sequence[IntervalSpec]
        Intervals to distribute across worker chunks.
    num_chunks : int
        Maximum number of chunks to produce.

    Returns
    -------
    list[list[IntervalSpec]]
        Non-empty chunks of ``IntervalSpec`` objects.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
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
    """
    Partition upper-triangle rows ``[0, n)`` into balanced tiles.

    Row ``i`` contributes ``n - 1 - i`` pairs, so splitting on
    cumulative pair count gives roughly equal pair work per tile.

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
    cum = np.cumsum(n - 1 - np.arange(n))
    edges = [
        int(np.searchsorted(cum, cum[-1] * p / n_tiles))
        for p in range(1, n_tiles)
    ]
    bounds = sorted({0, *edges, n})
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i] < bounds[i + 1]
    ]


def prepare_bucket_chunks(
    dataset: lance.LanceDataset,
    bucket_label: str,
    clustering: ClusteringConfig,
    num_chunks: int,
    threshold: int,
) -> tuple[int, nb.typed.List, list[list[IntervalSpec]], list[IntervalSpec]]:
    """
    Load one bucket's metadata and partition it into m/z-split tasks.

    Separates giant intervals before running LPT chunking so giant
    costs never skew the load balance of normal-interval chunks.

    Parameters
    ----------
    dataset : lance.LanceDataset
        Staged per-charge Lance dataset.
    bucket_label : str
        Human-readable charge label for logging.
    clustering : ClusteringConfig
        Validated clustering configuration controlling precursor m/z
        splitting and maximum interval size.
    num_chunks : int
        Target number of cost-balanced chunks for normal intervals.
    threshold : int
        Intervals with at least this many spectra are returned
        separately as giants for tiled distance computation.

    Returns
    -------
    tuple
        ``(n_spectra, splits, normal_chunks, giant_specs)``.
    """
    data = read_canonical_metadata(
        dataset, CANONICAL_SORT_COLUMNS, with_row_index=True
    )
    n = data.shape[0]
    logger.info("Cluster %d spectra with charge %s", n, bucket_label)
    idx = data["index"].to_numpy()
    mzs = data["precursor_mz"].to_numpy()
    splits = precursor_mz_splits(
        mzs,
        clustering.precursor_tol.mass,
        clustering.precursor_tol.mode,
        clustering.batch_size,
    )

    all_specs = [
        IntervalSpec(
            task_id=task_id,
            row_ids=idx[splits[task_id] : splits[task_id + 1]].tolist(),
            precursor_mzs=mzs[splits[task_id] : splits[task_id + 1]],
        )
        for task_id in range(len(splits) - 1)
    ]
    normal_specs = [spec for spec in all_specs if spec.n_spectra < threshold]
    giant_specs = [spec for spec in all_specs if spec.n_spectra >= threshold]
    normal_chunks = cost_based_chunking(normal_specs, num_chunks)
    return n, splits, normal_chunks, giant_specs
