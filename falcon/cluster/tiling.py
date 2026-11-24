"""Giant-interval memmap lifecycle and tiled distance computation."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import pyarrow as pa

from .distance import (
    close_memmap,
    fill_condensed_rows_csr,
    open_shared_condensed,
)
from .scheduling import plan_tile_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance

    from .scheduling import IntervalSpec

_TILE_PEAK_COLUMNS: tuple[str, ...] = ("mz", "intensity")

__all__ = [
    "GiantMemmapStore",
    "TileDone",
    "TileTask",
    "allocate_giant_tiles",
    "tile_distance",
]


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
class TileDone:
    """Completion record emitted by a Pass 1 distance tile worker."""

    giant_idx: int
    tile_cost: float
    tile_spectra: float


class GiantMemmapStore:
    """Manage lifecycle and eager unlinking of giant memmap files."""

    def __init__(self, temp_dir: os.PathLike[str] | str | None = None) -> None:
        self._temp_dir = temp_dir
        self._paths: dict[int, str] = {}

    def allocate(self, giant_idx: int, n_spectra: int) -> str:
        """
        Allocate a shared condensed memmap for ``giant_idx``.

        Parameters
        ----------
        giant_idx : int
            Unique index identifying the giant interval.
        n_spectra : int
            Number of spectra in the giant interval (at least 2).

        Returns
        -------
        str
            Path to the backing ``.npy`` file on disk.

        Raises
        ------
        ValueError
            If ``giant_idx`` is already allocated or ``n_spectra < 2``.
        """
        if giant_idx in self._paths:
            raise ValueError(
                f"Memmap already allocated for giant_idx={giant_idx}"
            )
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

    def release_all(self) -> None:
        """Unlink all remaining tracked memmap files."""
        for giant_idx in list(self._paths):
            self.release(giant_idx)

    def __enter__(self) -> Self:
        """Enter the managed memmap store context."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Ensure all remaining tracked memmaps are unlinked."""
        self.release_all()


def _extract_list_csr(
    column: pa.ChunkedArray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one PyArrow list column into ``(values, offsets)``."""
    col = (
        column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    )
    if not isinstance(col, (pa.ListArray, pa.LargeListArray)):
        raise TypeError(
            "Expected PyArrow list arrays for 'mz' and 'intensity' "
            f"columns, got {type(col).__name__} for {name!r}"
        )
    if col.null_count:
        raise ValueError(f"Null lists are not allowed in {name!r} column")
    offsets = col.offsets.to_numpy().astype(np.int64, copy=False)
    start, stop = int(offsets[0]), int(offsets[-1])
    if start != 0:
        offsets = offsets - start
    values = col.values.slice(start, stop - start)
    if values.null_count:
        raise ValueError(f"Null values are not allowed in {name!r} column")
    flat = np.asarray(values.to_numpy(), dtype=np.float32)
    return flat, offsets


def _extract_arrow_csr(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract float32 CSR peak buffers from a PyArrow table."""
    mz_flat, mz_offsets = _extract_list_csr(table["mz"], "mz")
    intensity_flat, int_offsets = _extract_list_csr(
        table["intensity"], "intensity"
    )
    if not np.array_equal(mz_offsets, int_offsets):
        raise ValueError(
            "Mismatched PyArrow list offsets between 'mz' and 'intensity'"
        )
    return mz_flat, intensity_flat, mz_offsets


def allocate_giant_tiles(
    raw_giants: Sequence[tuple[int, lance.LanceDataset, IntervalSpec]],
    fragment_tol: float,
    min_matched_peaks: int,
    n_workers: int,
    memmap_store: GiantMemmapStore,
) -> tuple[list[str], list[TileTask]]:
    """
    Allocate shared condensed memmaps and balanced row-band tiles.

    Parameters
    ----------
    raw_giants : Sequence[tuple[int, lance.LanceDataset, IntervalSpec]]
        Giant intervals as ``(bucket_id, dataset, spec)`` triples.
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched fragment peaks required for nonzero similarity.
    n_workers : int
        Available worker count bounding tiles per giant.
    memmap_store : GiantMemmapStore
        Store managing the lifecycle of allocated ``.npy`` memmaps.

    Returns
    -------
    tuple[list[str], list[TileTask]]
        Backing memmap paths aligned with ``raw_giants`` and Pass 1
        distance tile tasks.
    """
    memmap_paths: list[str] = []
    tile_tasks: list[TileTask] = []
    for giant_idx, (_bucket_id, dataset, spec) in enumerate(raw_giants):
        n = spec.n_spectra
        path = memmap_store.allocate(giant_idx, n)
        memmap_paths.append(path)
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
                    fragment_tol=fragment_tol,
                    min_matched_peaks=min_matched_peaks,
                )
            )
    return memmap_paths, tile_tasks


def tile_distance(task: TileTask) -> TileDone:
    """
    Compute rows ``[task.r0, task.r1)`` of a giant condensed memmap.

    Reads only the ``mz`` and ``intensity`` columns and streams their
    PyArrow ``ListArray`` buffers directly into
    :func:`~falcon.cluster.distance.fill_condensed_rows_csr`.

    Parameters
    ----------
    task : TileTask
        Row-band distance tile descriptor.

    Returns
    -------
    TileDone
        Typed completion record carrying cost and spectrum progress.
    """
    table = task.dataset.take(
        indices=pa.array(task.row_ids), columns=list(_TILE_PEAK_COLUMNS)
    )
    mz_flat, intensity_flat, offsets = _extract_arrow_csr(table)
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
    return TileDone(
        giant_idx=task.giant_idx,
        tile_cost=task.tile_cost,
        tile_spectra=task.tile_spectra,
    )
