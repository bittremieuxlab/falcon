"""Spectrum quality filtering, normalization, and m/z bin bounds."""

import math
from typing import TypedDict

import numba as nb
import numpy as np
import spectrum_utils.spectrum as sus


class ProcessedSpectrum(TypedDict):
    """Preprocessed spectrum record staged into a Lance dataset."""

    identifier: str
    precursor_mz: float
    precursor_charge: int | None
    mz: np.ndarray
    intensity: np.ndarray
    retention_time: float | None


@nb.njit(cache=True)
def _check_spectrum_valid(
    spectrum_mz: np.ndarray, min_peaks: int, min_mz_range: float
) -> bool:
    """
    Check whether a spectrum meets minimum peak count and m/z span.

    Assumes ``spectrum_mz`` is sorted ascending.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        Sorted fragment m/z peaks of the spectrum.
    min_peaks : int
        Minimum number of peaks required.
    min_mz_range : float
        Minimum m/z span (``max - min``) required.

    Returns
    -------
    bool
        ``True`` if the spectrum passes both quality gates, ``False``
        otherwise.
    """
    return (
        len(spectrum_mz) >= min_peaks
        and spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range
    )


@nb.njit(cache=True)
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    L2-normalize spectrum peak intensities to unit Euclidean norm.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        Peak intensities with nonzero L2 norm.

    Returns
    -------
    np.ndarray
        Unit-norm peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(
    spectrum: sus.MsmsSpectrum,
    min_peaks: int,
    min_mz_range: float,
    mz_min: float | None = None,
    mz_max: float | None = None,
    remove_precursor_tolerance: float | None = None,
    min_intensity: float | None = None,
    max_peaks_used: int | None = None,
    scaling: str | None = None,
) -> ProcessedSpectrum | None:
    """
    Filter, scale, and L2-normalize a single MS/MS spectrum.

    Processing steps (with quality-gate checks after each filter):
    - Restrict fragment peaks to ``[mz_min, mz_max]``.
    - Remove peaks around the precursor m/z.
    - Remove peaks below ``min_intensity`` of the base peak and retain
      at most ``max_peaks_used`` strongest peaks.
    - Apply intensity scaling and L2 normalization.

    Note that the pinned ``spectrum_utils`` methods mutate ``spectrum``
    in place.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        Input spectrum to preprocess.
    min_peaks : int
        Minimum number of peaks required after each filtering stage.
    min_mz_range : float
        Minimum m/z span required after each filtering stage.
    mz_min : float | None, default=None
        Minimum fragment m/z (inclusive). No lower bound when ``None``.
    mz_max : float | None, default=None
        Maximum fragment m/z (inclusive). No upper bound when ``None``.
    remove_precursor_tolerance : float | None, default=None
        Window (in Da) around the precursor m/z within which peaks are
        removed. Skipped when ``None``.
    min_intensity : float | None, default=None
        Minimum peak intensity relative to the base peak.
    max_peaks_used : int | None, default=None
        Maximum number of most-intense peaks to retain.
    scaling : {"root", "log", "rank"} | None, default=None
        Peak intensity transformation applied before L2 normalization:
        ``"root"`` (square root), ``"log"`` (log2 of ``intensity + 1``),
        ``"rank"`` (rank up to ``max_peaks_used``), or ``None`` (off).

    Returns
    -------
    ProcessedSpectrum | None
        Dictionary ready for Lance staging, or ``None`` if the spectrum
        fails a quality gate.
    """
    none_charge = spectrum.precursor_charge is None
    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        return None

    if remove_precursor_tolerance is not None:
        # Temporarily assign charge 1 so spectrum_utils can compute the
        # precursor window, then restore missing charge afterward.
        if none_charge:
            spectrum.precursor_charge = 1
        spectrum = spectrum.remove_precursor_peak(
            remove_precursor_tolerance, "Da", 0
        )
        if none_charge:
            spectrum.precursor_charge = None
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    if min_intensity is not None or max_peaks_used is not None:
        min_intensity = 0.0 if min_intensity is None else min_intensity
        spectrum = spectrum.filter_intensity(min_intensity, max_peaks_used)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks_used)
    intensity = _norm_intensity(spectrum.intensity)

    return {
        "identifier": spectrum.identifier,
        "precursor_mz": spectrum.precursor_mz,
        "precursor_charge": spectrum.precursor_charge,
        "mz": spectrum.mz,
        "intensity": intensity,
        "retention_time": spectrum.retention_time,
    }


@nb.njit("Tuple((u4, f4, f4))(f4, f4, f4)", cache=True)
def get_dim(
    min_mz: float, max_mz: float, bin_size: float
) -> tuple[int, float, float]:
    """
    Compute aligned m/z bin boundaries and total bin count.

    Parameters
    ----------
    min_mz : float
        Lower bound of the mass range (inclusive).
    max_mz : float
        Upper bound of the mass range (inclusive).
    bin_size : float
        Bin width in Da.

    Returns
    -------
    tuple[int, float, float]
        ``(n_bins, start_dim, end_dim)`` where ``start_dim`` is the
        largest multiple of ``bin_size`` at or below ``min_mz`` and
        ``end_dim`` is the smallest multiple strictly above ``max_mz``.
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim
