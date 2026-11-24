"""Spectrum filtering, normalization, and preprocessing factory."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, get_args

import numpy as np

from .types import INT8_MAX, INT8_MIN, ScalingMethod, mz_bin_bounds

if TYPE_CHECKING:
    from collections.abc import Callable

    import spectrum_utils.spectrum as sus

    from .config import FalconConfig
    from .types import ProcessedSpectrum

# Match spectrum_utils==0.3.5 proton mass constant (Da) for
# charge-reduced precursor peak removal.
_PROTON_MASS = 1.0072766

_F32_INFO = np.finfo(np.float32)
_F32_MAX = float(_F32_INFO.max)
_F32_SUBNORMAL_MIN = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
_VALID_SCALINGS = frozenset(get_args(ScalingMethod))

__all__ = ["build_process_spectrum", "process_spectrum"]


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
    n_peaks = len(spectrum_mz)
    return (
        n_peaks > 0
        and n_peaks >= min_peaks
        and float(spectrum_mz[-1]) - float(spectrum_mz[0]) >= min_mz_range
    )


def _has_valid_intensities(intensity: np.ndarray) -> bool:
    """Return True if non-empty intensities are finite and non-zero.

    Assumes ``intensity`` is non-empty (guaranteed by prior
    :func:`_check_spectrum_valid`).
    """
    min_i = float(intensity.min())
    max_i = float(intensity.max())
    return min_i >= 0.0 and max_i > 0.0 and math.isfinite(max_i)


def _has_valid_metadata(spectrum: sus.MsmsSpectrum) -> bool:
    """Return True if precursor m/z, RT, and charge fit schema types."""
    if not (_F32_SUBNORMAL_MIN <= spectrum.precursor_mz <= _F32_MAX):
        return False
    rt = spectrum.retention_time
    if rt is not None and not math.isnan(rt) and not (abs(rt) <= _F32_MAX):
        return False
    charge = spectrum.precursor_charge
    return charge is None or (
        not isinstance(charge, bool)
        and isinstance(charge, (int, np.integer))
        and INT8_MIN <= charge <= INT8_MAX
    )


def _restrict_mz_range(
    spectrum: sus.MsmsSpectrum,
    min_mz: float | None,
    max_mz: float | None,
) -> sus.MsmsSpectrum:
    """Restrict fragment peaks inclusively to ``[min_mz, max_mz]``."""
    mz_f64 = spectrum.mz.astype(np.float64, copy=False)
    n_peaks = len(mz_f64)
    lo = 0 if min_mz is None else int(np.searchsorted(mz_f64, min_mz, "left"))
    hi = (
        n_peaks
        if max_mz is None
        else int(np.searchsorted(mz_f64, max_mz, "right"))
    )
    if lo == 0 and hi == n_peaks:
        return spectrum
    spectrum.mz = spectrum.mz[lo:hi]
    spectrum.intensity = spectrum.intensity[lo:hi]
    if spectrum.annotation is not None:
        spectrum.annotation = spectrum.annotation[lo:hi]
    return spectrum


def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    L2-normalize spectrum peak intensities to unit Euclidean norm.

    Computes the Euclidean norm in float64 to avoid float32 overflow on
    large intensities and underflow on tiny intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        Peak intensities with nonzero L2 norm.

    Returns
    -------
    np.ndarray
        Unit-norm peak intensities as float32.
    """
    intensity_f64 = spectrum_intensity.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(intensity_f64))
    return (intensity_f64 / norm).astype(np.float32)


def _remove_precursor_peak(
    spectrum: sus.MsmsSpectrum,
    remove_precursor_tol: float,
) -> sus.MsmsSpectrum:
    """Remove precursor and charge-reduced precursor peaks in Da."""
    charge = spectrum.precursor_charge
    precursor_mz_f64 = float(spectrum.precursor_mz)
    mz_f64 = spectrum.mz.astype(np.float64, copy=False)
    abs_charge = 0 if charge is None else abs(int(charge))

    if abs_charge <= 1:
        targets: tuple[float, ...] = (precursor_mz_f64,)
    else:
        sign = 1.0 if charge is not None and charge > 0 else -1.0
        neutral_mass = (precursor_mz_f64 - sign * _PROTON_MASS) * abs_charge
        targets = (
            precursor_mz_f64,
            *(
                neutral_mass / sub_charge + sign * _PROTON_MASS
                for sub_charge in range(abs_charge - 1, 0, -1)
            ),
        )

    keep = np.ones(len(spectrum.mz), dtype=bool)
    for target in targets:
        keep &= np.abs(mz_f64 - target) > remove_precursor_tol
    spectrum.mz = spectrum.mz[keep]
    spectrum.intensity = spectrum.intensity[keep]
    if spectrum.annotation is not None:
        spectrum.annotation = spectrum.annotation[keep]
    return spectrum


def process_spectrum(
    spectrum: sus.MsmsSpectrum,
    min_peaks: int,
    min_mz_range: float,
    min_mz: float | None = None,
    max_mz: float | None = None,
    remove_precursor_tol: float | None = None,
    min_intensity: float | None = None,
    max_peaks_used: int | None = None,
    scaling: ScalingMethod | None = "off",
) -> ProcessedSpectrum | None:
    """
    Filter, scale, and L2-normalize a single MS/MS spectrum.

    Processing steps (with quality-gate checks after each filter):
    - Restrict fragment peaks to ``[min_mz, max_mz]``.
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
    min_mz : float | None, default=None
        Minimum fragment m/z (inclusive). No lower bound when ``None``.
    max_mz : float | None, default=None
        Maximum fragment m/z (inclusive). No upper bound when ``None``.
    remove_precursor_tol : float | None, default=None
        Window (in Da) around the precursor m/z within which peaks are
        removed. Skipped when ``None``.
    min_intensity : float | None, default=None
        Minimum peak intensity relative to the base peak.
    max_peaks_used : int | None, default=None
        Maximum number of most-intense peaks to retain.
    scaling : {"off", "root", "log", "rank"} | None, default="off"
        Peak intensity transformation applied before L2 normalization:
        ``"root"`` (square root), ``"log"`` (log2 of ``intensity + 1``),
        ``"rank"`` (rank up to ``max_peaks_used``), or ``"off"`` /
        ``None`` (no scaling).

    Returns
    -------
    ProcessedSpectrum | None
        Dictionary ready for Lance staging, or ``None`` if the spectrum
        fails a quality gate.

    Raises
    ------
    ValueError
        If ``scaling`` is not ``None`` or one of ``"off"``, ``"root"``,
        ``"log"``, ``"rank"``.
    """
    if scaling is not None and scaling not in _VALID_SCALINGS:
        raise ValueError(
            f"Unsupported intensity scaling {scaling!r}; "
            "expected one of 'off', 'root', 'log', 'rank', or None"
        )
    if (
        not _has_valid_metadata(spectrum)
        or not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range)
        or not np.isfinite(spectrum.mz).all()
    ):
        return None

    if min_mz is not None or max_mz is not None:
        spectrum = _restrict_mz_range(spectrum, min_mz, max_mz)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    if remove_precursor_tol is not None:
        spectrum = _remove_precursor_peak(spectrum, remove_precursor_tol)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    if not _has_valid_intensities(spectrum.intensity):
        return None

    if min_intensity is not None or max_peaks_used is not None:
        eff_min_intensity = 0.0 if min_intensity is None else min_intensity
        spectrum = spectrum.filter_intensity(eff_min_intensity, max_peaks_used)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    if scaling not in (None, "off"):
        # spectrum_utils 0.3.5 treats an explicit max_rank=None as
        # invalid for rank scaling (uses kwargs.get('max_rank', ...)).
        if max_peaks_used is not None:
            spectrum = spectrum.scale_intensity(
                scaling, max_rank=max_peaks_used
            )
        else:
            spectrum = spectrum.scale_intensity(scaling)
        if not _has_valid_intensities(spectrum.intensity):
            return None

    intensity = _norm_intensity(spectrum.intensity)
    charge = spectrum.precursor_charge
    rt = spectrum.retention_time

    return {
        "identifier": spectrum.identifier,
        "precursor_mz": float(spectrum.precursor_mz),
        "precursor_charge": None if charge is None else int(charge),
        "mz": spectrum.mz,
        "intensity": intensity,
        "retention_time": None if rt is None else float(rt),
    }


def build_process_spectrum(
    config: FalconConfig,
) -> Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None]:
    """
    Build the configured single-spectrum preprocessing callable.

    Aligns ``config.preprocessing.min_mz`` and ``max_mz`` outward to the
    ``config.clustering.fragment_tol`` grid via ``mz_bin_bounds``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.

    Returns
    -------
    Callable[[MsmsSpectrum], ProcessedSpectrum | None]
        Configured preprocessing function ready for joblib workers.
    """
    prep = config.preprocessing
    _, min_mz, max_mz = mz_bin_bounds(
        prep.min_mz, prep.max_mz, config.clustering.fragment_tol
    )
    return functools.partial(
        process_spectrum,
        min_peaks=prep.min_peaks,
        min_mz_range=prep.min_mz_range,
        min_mz=min_mz,
        max_mz=max_mz,
        remove_precursor_tol=prep.remove_precursor_tol,
        min_intensity=prep.min_intensity,
        max_peaks_used=prep.max_peaks_used,
        scaling=prep.scaling,
    )
