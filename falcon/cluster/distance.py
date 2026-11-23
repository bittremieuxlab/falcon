"""Memory-mapped condensed pairwise cosine distance matrix kernels."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .similarity import cosine_fast_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..types import SpectrumTuple

__all__ = [
    "compute_condensed_distance_matrix",
    "condensed_index",
    "fill_condensed_rows",
    "open_shared_condensed",
    "pack_spectra_csr",
]


def pack_spectra_csr(
    spec_tuples: Sequence[SpectrumTuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack variable-length spectrum peaks into contiguous CSR arrays.

    Packing avoids Numba's deprecated reflected-list boxing and improves
    cache locality across quadratic pairwise distance evaluations.

    Parameters
    ----------
    spec_tuples : Sequence[SpectrumTuple]
        Preprocessed spectra to pack.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(mz_flat, intensity_flat, offsets)`` where spectrum ``i``
        occupies slice ``offsets[i]:offsets[i + 1]``.
    """
    n = len(spec_tuples)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i, spec in enumerate(spec_tuples):
        offsets[i + 1] = offsets[i] + len(spec.mz)
    total_peaks = int(offsets[n])
    if total_peaks == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, offsets
    mz_flat = np.concatenate(
        [np.asarray(s.mz, dtype=np.float32) for s in spec_tuples]
    )
    intensity_flat = np.concatenate(
        [np.asarray(s.intensity, dtype=np.float32) for s in spec_tuples]
    )
    return mz_flat, intensity_flat, offsets


def open_shared_condensed(n: int) -> tuple[str, np.memmap]:
    """
    Create a named memmapped condensed distance matrix for tiling.

    The backing ``.npy`` file stays named on disk so worker processes
    can reopen it by path; the caller must unlink the file when done.

    Parameters
    ----------
    n : int
        Number of spectra in the interval.

    Returns
    -------
    tuple[str, np.memmap]
        Backing ``.npy`` file path and writable float32 memmap of shape
        ``(n * (n - 1) // 2,)``.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".npy", delete=False
    ) as pdist_file:
        pdist_filename = pdist_file.name
    condensed_dist_matrix = np.lib.format.open_memmap(
        pdist_filename,
        mode="w+",
        dtype=np.float32,
        shape=(n * (n - 1) // 2,),
    )
    return pdist_filename, condensed_dist_matrix


def compute_condensed_distance_matrix(
    spec_tuples: Sequence[SpectrumTuple],
    fragment_tol: float,
    min_matched_peaks: int,
) -> np.ndarray:
    """
    Compute the float32 condensed pairwise cosine distance matrix.

    Allocates an on-disk NumPy memmap in the system temp directory,
    unlinks the path immediately on POSIX so space is reclaimed when the
    memmap is garbage-collected, and fills it in parallel via Numba.

    Parameters
    ----------
    spec_tuples : Sequence[SpectrumTuple]
        Preprocessed spectra to compare pairwise.
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched fragment peaks required; pairs below this gate
        receive distance ``1.0``.

    Returns
    -------
    np.ndarray
        Float32 condensed upper-triangle distance vector of length
        ``n * (n - 1) // 2``.
    """
    n = len(spec_tuples)
    if n < 2:
        return np.empty(0, dtype=np.float32)

    pdist_filename, condensed_dist_matrix = open_shared_condensed(n)
    # Unlink immediately after mapping: on POSIX the inode lives until
    # the memmap is released, preventing temp-file leaks across splits.
    os.unlink(pdist_filename)

    mz_flat, intensity_flat, offsets = pack_spectra_csr(spec_tuples)
    _condensed_distance_matrix_parallel(
        condensed_dist_matrix,
        mz_flat,
        intensity_flat,
        offsets,
        fragment_tol,
        min_matched_peaks,
    )
    return condensed_dist_matrix


def fill_condensed_rows(
    condensed_dist_matrix: np.ndarray,
    spec_tuples: Sequence[SpectrumTuple],
    r0: int,
    r1: int,
    fragment_tol: float,
    min_matched_peaks: int,
) -> None:
    """
    Fill condensed distance entries for rows ``[r0, r1)`` serially.

    Computes pairs ``(i, j)`` with ``r0 <= i < r1`` and ``i < j < n``
    single-threaded for multiprocess tiling. Results are bit-for-bit
    identical to ``compute_condensed_distance_matrix``.

    Parameters
    ----------
    condensed_dist_matrix : np.ndarray
        Shared float32 condensed distance buffer written in place.
    spec_tuples : Sequence[SpectrumTuple]
        All spectra in the interval in canonical order.
    r0 : int
        Start row index (inclusive).
    r1 : int
        Stop row index (exclusive).
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched peaks required for nonzero similarity.
    """
    if r0 >= r1:
        return
    mz_flat, intensity_flat, offsets = pack_spectra_csr(spec_tuples)
    _condensed_rows_csr(
        condensed_dist_matrix,
        mz_flat,
        intensity_flat,
        offsets,
        r0,
        r1,
        fragment_tol,
        min_matched_peaks,
    )


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
    """Fill condensed upper-triangle row ``i`` from CSR peak arrays."""
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
def _condensed_rows_csr(
    condensed_dist_matrix: np.ndarray,
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
    r0: int,
    r1: int,
    fragment_tol: float,
    min_matched_peaks: int,
) -> None:
    """Fill condensed upper-triangle rows ``[r0, r1)`` from CSR data."""
    n = len(offsets) - 1
    for i in range(r0, r1):
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
    """Fill the full condensed pairwise distance matrix in parallel."""
    n = len(offsets) - 1
    for i in nb.prange(n - 1):
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


@nb.njit(cache=True, boundscheck=False)
def condensed_index(i: int, j: int, n: int) -> int:
    """
    Map a pair ``(i, j)`` to its SciPy condensed distance matrix index.

    Symmetric in ``i`` and ``j``; indices are swapped when ``i > j``.
    Hot-loop helper with no bounds checks; callers must guarantee
    ``0 <= i, j < n`` and ``i != j``.

    Parameters
    ----------
    i : int
        First spectrum index (``0 <= i < n``).
    j : int
        Second spectrum index (``0 <= j < n``, ``j != i``).
    n : int
        Total number of spectra (``n > 1``).

    Returns
    -------
    int
        Flat 0-based index into the condensed upper-triangle array.
    """
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
