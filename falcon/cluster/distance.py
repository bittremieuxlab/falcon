"""Condensed pairwise cosine distance matrix kernels and memmaps."""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .similarity import cosine_fast_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from ..types import PathLike, SpectrumTuple

# Allocate small condensed matrices (< ~8 MiB in float32) directly in
# RAM to avoid per-interval temporary-file churn in /tmp.
MEMMAP_MIN_SPECTRA = 2**11

__all__ = [
    "MEMMAP_MIN_SPECTRA",
    "close_memmap",
    "compute_condensed_distance_matrix",
    "compute_condensed_distance_matrix_csr",
    "condensed_index",
    "extract_arrow_csr",
    "fill_condensed_rows_csr",
    "open_shared_condensed",
    "pack_spectra_csr",
]


def close_memmap(mm: np.ndarray) -> None:
    """Close the underlying OS mapping of a NumPy memmap immediately."""
    mmap_obj = getattr(mm, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _extract_list_csr(
    column: pa.ChunkedArray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one PyArrow list column into ``(values, offsets)``."""
    import pyarrow as pa

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


def extract_arrow_csr(
    table: pa.Table,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract contiguous float32 CSR peak buffers from a PyArrow table.

    Parameters
    ----------
    table : pa.Table
        Arrow table containing non-null ``mz`` and ``intensity`` float
        list columns with matching per-row lengths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(mz_flat, intensity_flat, offsets)`` where spectrum ``i``
        occupies slice ``offsets[i]:offsets[i + 1]``.

    Raises
    ------
    TypeError
        If ``mz`` or ``intensity`` is not a PyArrow list array.
    ValueError
        If either column contains null lists/values or mismatched
        offsets.
    """
    mz_flat, mz_offsets = _extract_list_csr(table["mz"], "mz")
    intensity_flat, int_offsets = _extract_list_csr(
        table["intensity"], "intensity"
    )
    if not np.array_equal(mz_offsets, int_offsets):
        raise ValueError(
            "Mismatched PyArrow list offsets between 'mz' and 'intensity'"
        )
    return mz_flat, intensity_flat, mz_offsets


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

    Raises
    ------
    ValueError
        If any spectrum has mismatched ``mz`` and ``intensity`` lengths.
    """
    n = len(spec_tuples)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i, spec in enumerate(spec_tuples):
        n_mz = len(spec.mz)
        if n_mz != len(spec.intensity):
            raise ValueError(
                f"Spectrum {i} has mismatched mz ({n_mz}) and intensity "
                f"({len(spec.intensity)}) lengths"
            )
        offsets[i + 1] = offsets[i] + n_mz
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


def open_shared_condensed(
    n: int,
    *,
    temp_dir: PathLike | None = None,
) -> tuple[str, np.memmap]:
    """
    Create a named memmapped condensed distance matrix for tiling.

    The backing ``.npy`` file stays named on disk so worker processes
    can reopen it by path; the caller must unlink the file when done.

    Parameters
    ----------
    n : int
        Number of spectra in the interval (must be at least 2).
    temp_dir : PathLike | None, default=None
        Optional directory for the backing ``.npy`` file; defaults to
        Python's system temporary directory.

    Returns
    -------
    tuple[str, np.memmap]
        Backing ``.npy`` file path and writable float32 memmap of shape
        ``(n * (n - 1) // 2,)``.

    Raises
    ------
    ValueError
        If ``n < 2``.
    """
    if n < 2:
        raise ValueError(
            f"open_shared_condensed requires at least 2 spectra, got {n}"
        )
    target_dir = os.fspath(temp_dir) if temp_dir is not None else None
    with tempfile.NamedTemporaryFile(
        suffix=".npy", dir=target_dir, delete=False
    ) as pdist_file:
        pdist_filename = pdist_file.name
    try:
        condensed_dist_matrix = np.lib.format.open_memmap(
            pdist_filename,
            mode="w+",
            dtype=np.float32,
            shape=(n * (n - 1) // 2,),
        )
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(pdist_filename)
        raise
    return pdist_filename, condensed_dist_matrix


def _validate_csr_buffers(
    mz_flat: np.ndarray,
    intensity_flat: np.ndarray,
    offsets: np.ndarray,
) -> int:
    """Validate 1D CSR peak arrays and offsets; return count."""
    if offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError(
            f"offsets must be 1D with length >= 1, got shape {offsets.shape}"
        )
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(
            f"offsets must have an integer dtype, got {offsets.dtype}"
        )
    if bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("offsets must be monotonically non-decreasing")
    if (
        mz_flat.ndim != 1
        or intensity_flat.ndim != 1
        or mz_flat.shape[0] != intensity_flat.shape[0]
    ):
        raise ValueError(
            "mz_flat and intensity_flat must be 1D arrays of equal length"
        )
    if int(offsets[0]) < 0 or int(offsets[-1]) > mz_flat.shape[0]:
        raise ValueError(
            f"CSR offsets [{int(offsets[0])}, {int(offsets[-1])}] exceed "
            f"peak buffer length {mz_flat.shape[0]}"
        )
    return int(offsets.shape[0]) - 1


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
    Compute the float32 condensed pairwise distance matrix from CSR.

    Validates CSR shapes, dtypes, monotonicity, and bounds before
    invoking the unchecked Numba kernels. Small intervals (fewer than
    ``MEMMAP_MIN_SPECTRA`` spectra) use an in-memory float32 array;
    larger intervals allocate a NumPy memmap in ``temp_dir`` (or the
    system temp directory) and unlink its path immediately on POSIX so
    space is reclaimed when the memmap is released.

    Parameters
    ----------
    mz_flat, intensity_flat : np.ndarray
        Contiguous 1D fragment m/z and unit-norm intensity buffers.
    offsets : np.ndarray
        Monotonic 1D integer CSR offsets of length ``n + 1``.
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched fragment peaks required; pairs below this gate
        receive distance ``1.0``.
    temp_dir : PathLike | None, default=None
        Optional directory for large interval distance memmaps.

    Returns
    -------
    np.ndarray
        Float32 condensed upper-triangle distance vector of length
        ``n * (n - 1) // 2``.

    Raises
    ------
    ValueError
        If CSR arrays or offsets have invalid shape, dtype, order, or
        bounds.
    """
    n = _validate_csr_buffers(mz_flat, intensity_flat, offsets)
    if n < 2:
        return np.empty(0, dtype=np.float32)

    is_memmap = n >= MEMMAP_MIN_SPECTRA
    if not is_memmap:
        condensed_dist_matrix: np.ndarray = np.empty(
            n * (n - 1) // 2, dtype=np.float32
        )
    else:
        pdist_filename, condensed_dist_matrix = open_shared_condensed(
            n, temp_dir=temp_dir
        )
        try:
            # Unlink immediately after mapping: on POSIX the inode lives
            # until the memmap is released, preventing temp-file leaks.
            os.unlink(pdist_filename)
        except BaseException:
            close_memmap(condensed_dist_matrix)
            raise

    try:
        if nb.get_num_threads() <= 1:
            _condensed_rows_csr(
                condensed_dist_matrix,
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
                condensed_dist_matrix,
                mz_flat,
                intensity_flat,
                offsets,
                fragment_tol,
                min_matched_peaks,
            )
    except BaseException:
        if is_memmap:
            close_memmap(condensed_dist_matrix)
        raise
    return condensed_dist_matrix


def compute_condensed_distance_matrix(
    spec_tuples: Sequence[SpectrumTuple],
    fragment_tol: float,
    min_matched_peaks: int,
    *,
    temp_dir: PathLike | None = None,
) -> np.ndarray:
    """
    Compute the float32 condensed pairwise cosine distance matrix.

    Packs ``spec_tuples`` into CSR buffers and delegates to
    :func:`compute_condensed_distance_matrix_csr`.

    Parameters
    ----------
    spec_tuples : Sequence[SpectrumTuple]
        Preprocessed spectra to compare pairwise.
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched fragment peaks required; pairs below this gate
        receive distance ``1.0``.
    temp_dir : PathLike | None, default=None
        Optional directory for large interval distance memmaps.

    Returns
    -------
    np.ndarray
        Float32 condensed upper-triangle distance vector of length
        ``n * (n - 1) // 2``.
    """
    if len(spec_tuples) < 2:
        return np.empty(0, dtype=np.float32)
    mz_flat, intensity_flat, offsets = pack_spectra_csr(spec_tuples)
    return compute_condensed_distance_matrix_csr(
        mz_flat,
        intensity_flat,
        offsets,
        fragment_tol,
        min_matched_peaks,
        temp_dir=temp_dir,
    )


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
    """
    Fill condensed distance rows ``[r0, r1)`` serially from CSR arrays.

    Validates array shapes, dtypes, and bounds in Python before invoking
    the unchecked Numba kernel. Enables tile workers to stream directly
    from PyArrow ``ListArray`` CSR buffers without constructing
    intermediate ``SpectrumTuple`` objects.

    Parameters
    ----------
    condensed_dist_matrix : np.ndarray
        Shared floating-point condensed distance buffer written in
        place.
    mz_flat, intensity_flat : np.ndarray
        Contiguous 1D fragment m/z and unit-norm intensity buffers.
    offsets : np.ndarray
        Monotonic 1D integer CSR offsets of length ``n + 1``.
    r0 : int
        Start row index (inclusive, ``0 <= r0 <= r1 <= n``).
    r1 : int
        Stop row index (exclusive, ``0 <= r0 <= r1 <= n``).
    fragment_tol : float
        Fragment m/z matching tolerance (in Da).
    min_matched_peaks : int
        Minimum matched peaks required for nonzero similarity.

    Raises
    ------
    ValueError
        If row bounds, condensed buffer dtype/size, or CSR arrays are
        invalid.
    """
    n = _validate_csr_buffers(mz_flat, intensity_flat, offsets)
    if r0 < 0 or r1 < 0 or r0 > n or r1 > n or r0 > r1:
        raise ValueError(
            f"Row band [{r0}, {r1}) out of bounds for n={n} spectra"
        )
    required = n * (n - 1) // 2
    if not np.issubdtype(condensed_dist_matrix.dtype, np.floating):
        raise ValueError(
            "condensed_dist_matrix must have a floating dtype, got "
            f"{condensed_dist_matrix.dtype}"
        )
    if (
        condensed_dist_matrix.ndim != 1
        or condensed_dist_matrix.shape[0] < required
    ):
        raise ValueError(
            f"condensed_dist_matrix must be 1D with length >= {required}, "
            f"got shape {condensed_dist_matrix.shape}"
        )
    if r0 == r1:
        return
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
    """Fill the condensed pairwise distance matrix with paired rows."""
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
