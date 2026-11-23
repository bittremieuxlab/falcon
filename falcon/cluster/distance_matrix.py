"""Memory-mapped condensed pairwise cosine distance matrix kernels."""

import os
import tempfile

import numba as nb
import numpy as np

from . import similarity


def compute_condensed_distance_matrix(
    spec_tuples: list[similarity.SpectrumTuple],
    fragment_mz_tol: float,
    min_matches: int,
) -> np.ndarray:
    """
    Compute the float32 condensed pairwise cosine distance matrix.

    Allocates an on-disk NumPy memmap in the system temp directory,
    unlinks the path immediately on POSIX so space is reclaimed when the
    memmap is garbage-collected, and fills it in parallel via Numba.

    Parameters
    ----------
    spec_tuples : list[SpectrumTuple]
        Preprocessed spectra to compare pairwise.
    fragment_mz_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matches : int
        Minimum matched fragment peaks required; pairs below this gate
        receive distance ``1.0``.

    Returns
    -------
    np.ndarray
        Float32 condensed upper-triangle distance vector of length
        ``n * (n - 1) // 2``.
    """
    n = len(spec_tuples)
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
    # Unlink immediately after mapping: on POSIX the inode lives until
    # the memmap is released, preventing temp-file leaks across splits.
    os.unlink(pdist_filename)

    _condensed_distance_matrix_parallel(
        condensed_dist_matrix,
        spec_tuples,
        fragment_mz_tol,
        min_matches,
    )

    return condensed_dist_matrix


def open_shared_condensed(n: int) -> tuple[str, np.memmap]:
    """
    Create a named memmapped condensed distance matrix for tiling.

    Unlike ``compute_condensed_distance_matrix``, the backing file stays
    named on disk so multiple worker processes can open it by path; the
    caller must unlink the file after all workers finish.

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


@nb.njit
def _condensed_rows(
    condensed_dist_matrix: np.ndarray,
    spec_tuples: list[similarity.SpectrumTuple],
    r0: int,
    r1: int,
    fragment_mz_tol: float,
    min_matches: int,
) -> None:
    """
    Fill condensed distance entries for rows ``[r0, r1)``.

    Computes pairs ``(i, j)`` with ``r0 <= i < r1`` and ``i < j < n``
    serially (no ``prange``) because parallelism comes from running
    disjoint row-band tiles across separate worker processes. Results
    are bit-for-bit identical to ``_condensed_distance_matrix_parallel``
    in full-matrix mode.

    Parameters
    ----------
    condensed_dist_matrix : np.ndarray
        Shared float32 condensed distance buffer written in place.
    spec_tuples : list[SpectrumTuple]
        All spectra in the interval in canonical order.
    r0 : int
        Start row index (inclusive).
    r1 : int
        Stop row index (exclusive).
    fragment_mz_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matches : int
        Minimum matched peaks required for nonzero similarity.
    """
    n = len(spec_tuples)
    for i in range(r0, r1):
        spec_tup1 = spec_tuples[i]
        for j in range(i + 1, n):
            sim, n_match = similarity.cosine_fast(
                spec_tup1, spec_tuples[j], fragment_mz_tol
            )
            if n_match < min_matches:
                sim = 0.0
            condensed_dist_matrix[condensed_index(i, j, n)] = 1.0 - sim


@nb.njit(parallel=True)
def _condensed_distance_matrix_parallel(
    condensed_dist_matrix: np.ndarray,
    spec_tuples: list[similarity.SpectrumTuple],
    fragment_mz_tol: float,
    min_matches: int,
) -> None:
    """
    Fill the full condensed pairwise distance matrix in parallel.

    Parameters
    ----------
    condensed_dist_matrix : np.ndarray
        Preallocated float32 condensed distance buffer written in place.
    spec_tuples : list[SpectrumTuple]
        Spectra to compare pairwise.
    fragment_mz_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matches : int
        Minimum matched peaks required for nonzero similarity.
    """
    n = len(spec_tuples)

    for i in nb.prange(n - 1):
        for j in range(i + 1, n):
            spec_tup1 = spec_tuples[i]
            spec_tup2 = spec_tuples[j]
            sim, n_match = similarity.cosine_fast(
                spec_tup1, spec_tup2, fragment_mz_tol
            )
            if n_match < min_matches:
                sim = 0.0
            distance = 1.0 - sim
            idx = condensed_index(i, j, n)
            condensed_dist_matrix[idx] = distance


@nb.njit
def condensed_index(i: int, j: int, n: int) -> int:
    """
    Map a pair ``(i, j)`` to its SciPy condensed distance matrix index.

    Symmetric in ``i`` and ``j``; indices are swapped when ``i > j``.

    Parameters
    ----------
    i : int
        First spectrum index (``0 <= i < n``).
    j : int
        Second spectrum index (``0 <= j < n``, ``j != i``).
    n : int
        Total number of spectra (``n > 0``).

    Returns
    -------
    int
        Flat 0-based index into the condensed upper-triangle array.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``i == j``, or either index is out of ``[0, n)``.
    """
    if n <= 0:
        raise ValueError("Number of spectra must be greater than 0")
    elif i == j:
        raise ValueError("No diagonal elements in condensed matrix")
    elif i < 0 or j < 0:
        raise ValueError("Indices must be non-negative")
    elif i >= n or j >= n:
        raise ValueError("Index out of bounds")
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
