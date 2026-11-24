"""Greedy cosine similarity and condensed distance matrix kernels."""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

    from ..config import PathLike

# Allocate small condensed matrices (< ~8 MiB in float32) in RAM to
# avoid per-interval temporary-file churn.
MEMMAP_MIN_SPECTRA = 2**11


@nb.njit(cache=True)
def cosine_fast_arrays(
    mz1: np.ndarray,
    intensity1: np.ndarray,
    mz2: np.ndarray,
    intensity2: np.ndarray,
    fragment_tol: float,
) -> tuple[float, int]:
    """
    Compute greedy one-to-one cosine similarity over peak arrays.

    Assumes both ``mz`` arrays are sorted ascending and both
    ``intensity`` arrays are L2-normalized with matching length.
    """
    n1 = len(mz1)
    n2 = len(mz2)
    if n1 == 0 or n2 == 0:
        return 0.0, 0

    other_idx = 0
    peak_matches = []
    for i in range(n1):
        peak_mz = mz1[i]
        peak_intensity = intensity1[i]
        while other_idx < n2 and peak_mz - mz2[other_idx] > fragment_tol:
            other_idx += 1
        if other_idx == n2:
            break
        j = other_idx
        while j < n2 and mz2[j] - peak_mz <= fragment_tol:
            # Negate score so plain tuple sort orders by descending
            # score with ascending (i, j) tie-breaking and stays
            # Numba-cacheable.
            peak_matches.append((-(peak_intensity * intensity2[j]), i, j))
            j += 1

    if len(peak_matches) == 0:
        return 0.0, 0

    score = 0.0
    n_matches = 0
    peak_matches.sort()
    used1 = np.zeros(n1, np.bool_)
    used2 = np.zeros(n2, np.bool_)
    for neg_score, i, j in peak_matches:
        if not used1[i] and not used2[j]:
            score -= neg_score
            used1[i] = True
            used2[j] = True
            n_matches += 1

    return max(0.0, min(score, 1.0)), n_matches


def close_memmap(mm: np.ndarray) -> None:
    """Close the underlying OS mapping of a NumPy memmap immediately."""
    mmap_obj = getattr(mm, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _extract_list_csr(
    column: pa.ChunkedArray,
) -> tuple[np.ndarray, np.ndarray]:
    col = (
        column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    )
    offsets = col.offsets.to_numpy().astype(np.int64, copy=False)
    start, stop = int(offsets[0]), int(offsets[-1])
    if start != 0:
        offsets = offsets - start
    values = col.values.slice(start, stop - start)
    return np.asarray(values.to_numpy(), dtype=np.float32), offsets


def extract_arrow_csr(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract contiguous float32 CSR peaks from a PyArrow table."""
    mz_flat, mz_offsets = _extract_list_csr(table["mz"])
    intensity_flat, int_offsets = _extract_list_csr(table["intensity"])
    if not np.array_equal(mz_offsets, int_offsets):
        raise ValueError(
            "Mismatched PyArrow list offsets between 'mz' and 'intensity'"
        )
    return mz_flat, intensity_flat, mz_offsets


def open_shared_condensed(
    n: int,
    *,
    temp_dir: PathLike | None = None,
) -> tuple[str, np.memmap]:
    """Create a named float32 memmap for a condensed distance matrix."""
    target_dir = os.fspath(temp_dir) if temp_dir is not None else None
    with tempfile.NamedTemporaryFile(
        suffix=".npy", dir=target_dir, delete=False
    ) as pdist_file:
        pdist_filename = pdist_file.name
    try:
        mm = np.lib.format.open_memmap(
            pdist_filename,
            mode="w+",
            dtype=np.float32,
            shape=(n * (n - 1) // 2,),
        )
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(pdist_filename)
        raise
    return pdist_filename, mm


def compute_condensed_distance_matrix_csr(
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
    fragment_tol: float,
    min_matched_peaks: int,
    *,
    temp_dir: PathLike | None = None,
) -> np.ndarray:
    """
    Compute the float32 condensed pairwise cosine distance matrix.

    Small intervals (``< MEMMAP_MIN_SPECTRA``) stay in RAM; larger
    intervals allocate a memmap in ``temp_dir`` and unlink its path
    immediately on POSIX so disk space is reclaimed on release.
    """
    n = int(offsets.shape[0]) - 1
    if n < 2:
        return np.empty(0, dtype=np.float32)

    is_memmap = n >= MEMMAP_MIN_SPECTRA
    if not is_memmap:
        condensed: np.ndarray = np.empty(n * (n - 1) // 2, dtype=np.float32)
    else:
        pdist_filename, condensed = open_shared_condensed(n, temp_dir=temp_dir)
        os.unlink(pdist_filename)

    try:
        if nb.get_num_threads() <= 1:
            fill_condensed_rows_csr(
                condensed,
                mz_flat,
                intensity_flat,
                offsets,
                0,
                n - 1,
                fragment_tol,
                min_matched_peaks,
            )
        else:
            _condensed_distance_matrix_parallel(
                condensed,
                mz_flat,
                intensity_flat,
                offsets,
                fragment_tol,
                min_matched_peaks,
            )
    except BaseException:
        if is_memmap:
            close_memmap(condensed)
        raise
    return condensed


@nb.njit(cache=True, boundscheck=False)
def _fill_distance_row(
    condensed_dist_matrix: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
    i: int,
    n: int,
    fragment_tol: float,
    min_matched_peaks: int,
) -> None:
    s1, e1 = offsets[i], offsets[i + 1]
    mz1 = mz_flat[s1:e1]
    int1 = intensity_flat[s1:e1]
    row_base = n * i - ((i + 2) * (i + 1)) // 2
    for j in range(i + 1, n):
        s2, e2 = offsets[j], offsets[j + 1]
        sim, n_match = cosine_fast_arrays(
            mz1,
            int1,
            mz_flat[s2:e2],
            intensity_flat[s2:e2],
            fragment_tol,
        )
        if n_match < min_matched_peaks:
            sim = 0.0
        condensed_dist_matrix[row_base + j] = 1.0 - sim


@nb.njit(cache=True, boundscheck=False)
def fill_condensed_rows_csr(
    condensed_dist_matrix: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
    r0: int,
    r1: int,
    fragment_tol: float,
    min_matched_peaks: int,
) -> None:
    """Fill condensed distance rows ``[r0, r1)`` serially from CSR."""
    n = len(offsets) - 1
    stop = min(r1, n - 1)
    for i in range(r0, stop):
        _fill_distance_row(
            condensed_dist_matrix,
            mz_flat,
            intensity_flat,
            offsets,
            i,
            n,
            fragment_tol,
            min_matched_peaks,
        )


@nb.njit(cache=True, parallel=True, boundscheck=False)
def _condensed_distance_matrix_parallel(
    condensed_dist_matrix: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
    fragment_tol: float,
    min_matched_peaks: int,
) -> None:
    n = len(offsets) - 1
    n_rows = n - 1
    half = (n_rows + 1) // 2
    for k in nb.prange(half):
        _fill_distance_row(
            condensed_dist_matrix,
            mz_flat,
            intensity_flat,
            offsets,
            k,
            n,
            fragment_tol,
            min_matched_peaks,
        )
        mirror = n_rows - 1 - k
        if mirror != k:
            _fill_distance_row(
                condensed_dist_matrix,
                mz_flat,
                intensity_flat,
                offsets,
                mirror,
                n,
                fragment_tol,
                min_matched_peaks,
            )


@nb.njit(cache=True, boundscheck=False)
def condensed_index(i: int, j: int, n: int) -> int:
    """Map an index pair ``(i, j)`` to its condensed matrix offset."""
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
