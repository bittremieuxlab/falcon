"""Precursor m/z interval splitting, LPT chunking, and tile planning."""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

import numba as nb
import numba.typed
import numpy as np

from ..dataset import CANONICAL_SORT_COLUMNS, read_canonical_metadata
from ..types import PrecursorTolMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance

    from ..config import ClusteringConfig

logger = logging.getLogger(__name__)

# Number of cost-balanced m/z-split chunks to create per CPU worker.
# Having several times more tasks than workers lets joblib load-balance
# dynamically instead of stranding cores on the tail of a fixed split.
CHUNKS_PER_WORKER = 8

# Minimum pairs per distance tile. Giant intervals split into at most
# worker_count tiles, never finer than this pair budget so each tile
# amortizes its Lance read over enough cosine distance work.
MIN_PAIRS_PER_TILE = 4_000_000

_VALID_PRECURSOR_TOL_MODES = frozenset(get_args(PrecursorTolMode))

__all__ = [
    "CHUNKS_PER_WORKER",
    "MIN_PAIRS_PER_TILE",
    "BucketPlan",
    "IntervalSpec",
    "cost_based_chunking",
    "effective_tile_threshold",
    "interval_row_bounds",
    "plan_tile_bounds",
    "precursor_mz_splits",
    "prepare_bucket_chunks",
    "tile_threshold",
]


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
class BucketPlan:
    """Partitioned scheduling plan for one charge bucket."""

    n_spectra: int
    splits: np.ndarray
    normal_chunks: list[list[IntervalSpec]]
    giant_specs: list[IntervalSpec]


def tile_threshold(batch_size: int) -> int:
    """
    Return the minimum interval size that triggers tiled distance work.

    Tracking ``batch_size // 2`` self-scales with ``batch_size`` and
    catches forced-split giants in ``[batch_size / 2, batch_size]``.

    Parameters
    ----------
    batch_size : int
        Configured maximum interval size (must be at least 1).

    Returns
    -------
    int
        Minimum spectrum count for tiling a distance matrix build.

    Raises
    ------
    ValueError
        If ``batch_size < 1``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    return max(2, batch_size // 2)


def effective_tile_threshold(batch_size: int, n_workers: int) -> int:
    """
    Return the minimum interval size that splits into at least 2 tiles.

    Parameters
    ----------
    batch_size : int
        Configured maximum interval size (must be at least 1).
    n_workers : int
        Available parallel worker count.

    Returns
    -------
    int
        Minimum spectrum count that justifies tiled distance work
        (``batch_size + 1`` when ``n_workers <= 1`` to disable tiling).
    """
    if n_workers <= 1:
        return batch_size + 1
    # Exact smallest n with n * (n - 1) // 2 >= 2 * MIN_PAIRS_PER_TILE.
    min_two_tile = (math.isqrt(16 * MIN_PAIRS_PER_TILE) + 3) // 2
    return max(tile_threshold(batch_size), min_two_tile)


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
    """Compute monotonic split indices into an int64 NumPy array."""
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

    First partitions at adjacent precursor m/z gaps strictly exceeding
    ``precursor_tol_mass``, then divides any contiguous block larger
    than ``batch_size`` into evenly sized intervals of at most
    ``batch_size`` spectra (differing in size by at most 1).

    Parameters
    ----------
    precursor_mzs : np.ndarray
        1D precursor m/z array sorted ascending (evaluated in float64).
    precursor_tol_mass : float
        Maximum gap between adjacent spectra within a block.
    precursor_tol_mode : {"Da", "ppm"}
        Unit of ``precursor_tol_mass``.
    batch_size : int
        Maximum number of spectra allowed in a single interval.

    Returns
    -------
    np.ndarray
        Monotonic int64 split-point indices starting at ``0`` and ending
        at ``len(precursor_mzs)`` (or ``[0]`` when empty).

    Raises
    ------
    ValueError
        If ``batch_size < 1``, ``precursor_tol_mode`` is invalid,
        ``precursor_mzs`` is not 1D, or ``precursor_tol_mode == "ppm"``
        with non-finite or non-positive m/z values.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if precursor_tol_mode not in _VALID_PRECURSOR_TOL_MODES:
        raise ValueError(
            "precursor_tol_mode must be 'Da' or 'ppm', "
            f"got {precursor_tol_mode!r}"
        )
    mzs = np.asarray(precursor_mzs, dtype=np.float64)
    if mzs.ndim != 1:
        raise ValueError(f"precursor_mzs must be 1D, got shape {mzs.shape}")
    mode_is_da = precursor_tol_mode == "Da"
    if not mode_is_da and not bool(np.all(np.isfinite(mzs) & (mzs > 0.0))):
        raise ValueError(
            "precursor_tol_mode='ppm' requires finite strictly positive "
            "m/z values"
        )
    return _precursor_mz_splits_nb(
        mzs, precursor_tol_mass, mode_is_da, batch_size
    )


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

    Row ``i`` contributes ``n - 1 - i`` pairs, so half-open prefix sums
    of pair counts produce roughly equal pair work per tile while
    guaranteeing every tile performs strictly positive pair work.

    Parameters
    ----------
    n : int
        Number of spectra in the interval (must be at least 2).
    n_tiles : int
        Target number of work-balanced row bands (must be at least 1).

    Returns
    -------
    list[tuple[int, int]]
        Non-empty contiguous ``(r0, r1)`` half-open row ranges covering
        ``[0, n)``.

    Raises
    ------
    ValueError
        If ``n < 2`` or ``n_tiles < 1``.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if n_tiles < 1:
        raise ValueError(f"n_tiles must be at least 1, got {n_tiles}")

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
    """
    Plan work-balanced row-band tiles and quadratic costs for a giant.

    Picks ``min(n_workers, max(1, total_pairs // MIN_PAIRS_PER_TILE))``
    tiles and assigns each ``(r0, r1)`` band a proportional share of the
    interval's ``n ** 2`` scheduling cost.

    Parameters
    ----------
    n : int
        Number of spectra in the giant interval (must be at least 2).
    n_workers : int
        Available worker count (must be at least 1).

    Returns
    -------
    list[tuple[int, int, float]]
        List of ``(r0, r1, tile_cost)`` tuples covering ``[0, n)``.

    Raises
    ------
    ValueError
        If ``n < 2`` or ``n_workers < 1``.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")

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
) -> BucketPlan:
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
    BucketPlan
        Typed plan containing spectrum count, split boundaries,
        cost-balanced normal chunks, and giant interval specs.
    """
    data = read_canonical_metadata(
        dataset, CANONICAL_SORT_COLUMNS, with_row_index=True
    )
    n = data.shape[0]
    logger.info("Cluster %d spectra with charge %s", n, bucket_label)
    idx = data["index"].to_numpy(dtype=np.int64)
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
            row_ids=idx[
                int(splits[task_id]) : int(splits[task_id + 1])
            ].copy(),
        )
        for task_id in range(len(splits) - 1)
    ]
    normal_specs = [spec for spec in all_specs if spec.n_spectra < threshold]
    giant_specs = [spec for spec in all_specs if spec.n_spectra >= threshold]
    normal_chunks = cost_based_chunking(normal_specs, num_chunks)
    return BucketPlan(
        n_spectra=n,
        splits=splits,
        normal_chunks=normal_chunks,
        giant_specs=giant_specs,
    )
