"""Spectrum filtering, normalization, and preprocessing factory."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

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

__all__ = ["build_process_spectrum", "process_spectrum"]


def _is_valid_spectrum(
    mz: np.ndarray, min_peaks: int, min_mz_range: float
) -> bool:
    """Return True if ``mz`` meets peak count and m/z span gates."""
    n = len(mz)
    return (
        n > 0
        and n >= min_peaks
        and float(mz[-1]) - float(mz[0]) >= min_mz_range
    )


def _has_valid_intensities(intensity: np.ndarray) -> bool:
    """Return True if non-empty intensities are finite and non-zero."""
    min_i = float(intensity.min())
    max_i = float(intensity.max())
    return min_i >= 0.0 and max_i > 0.0 and math.isfinite(max_i)


def _subset_spectrum(
    spectrum: sus.MsmsSpectrum, selector: slice | np.ndarray
) -> sus.MsmsSpectrum:
    spectrum.mz = spectrum.mz[selector]
    spectrum.intensity = spectrum.intensity[selector]
    if spectrum.annotation is not None:
        spectrum.annotation = spectrum.annotation[selector]
    return spectrum


def _restrict_mz_range(
    spectrum: sus.MsmsSpectrum,
    min_mz: float | None,
    max_mz: float | None,
) -> sus.MsmsSpectrum:
    """Restrict sorted peaks inclusively to ``[min_mz, max_mz]``."""
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
    return _subset_spectrum(spectrum, slice(lo, hi))


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
    return spectrum if keep.all() else _subset_spectrum(spectrum, keep)


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

    Applies m/z restriction, precursor removal, intensity filtering,
    optional scaling, and float64-accumulated L2 normalization, checking
    ``min_peaks`` and ``min_mz_range`` after each filter. Note that
    ``spectrum_utils`` mutates ``spectrum`` in place.

    Returns
    -------
    ProcessedSpectrum | None
        Staged spectrum record, or ``None`` if rejected by gates.
    """
    precursor_mz = float(spectrum.precursor_mz)
    rt = spectrum.retention_time
    charge = spectrum.precursor_charge
    if (
        not (math.isfinite(precursor_mz) and precursor_mz > 0.0)
        or (rt is not None and math.isinf(rt))
        or (
            charge is not None
            and not (
                isinstance(charge, (int, np.integer))
                and INT8_MIN <= charge <= INT8_MAX
            )
        )
        or not _is_valid_spectrum(spectrum.mz, min_peaks, min_mz_range)
        or not np.isfinite(spectrum.mz).all()
    ):
        return None

    if min_mz is not None or max_mz is not None:
        spectrum = _restrict_mz_range(spectrum, min_mz, max_mz)
        if not _is_valid_spectrum(spectrum.mz, min_peaks, min_mz_range):
            return None

    if remove_precursor_tol is not None:
        spectrum = _remove_precursor_peak(spectrum, remove_precursor_tol)
        if not _is_valid_spectrum(spectrum.mz, min_peaks, min_mz_range):
            return None

    if not _has_valid_intensities(spectrum.intensity):
        return None

    if min_intensity is not None or max_peaks_used is not None:
        eff_min = 0.0 if min_intensity is None else min_intensity
        spectrum = spectrum.filter_intensity(eff_min, max_peaks_used)
        if not _is_valid_spectrum(spectrum.mz, min_peaks, min_mz_range):
            return None

    if scaling not in (None, "off"):
        # spectrum_utils 0.3.5 rejects max_rank=None for rank scaling.
        if max_peaks_used is not None:
            spectrum = spectrum.scale_intensity(
                scaling, max_rank=max_peaks_used
            )
        else:
            spectrum = spectrum.scale_intensity(scaling)

    intensity_f64 = spectrum.intensity.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(intensity_f64))
    intensity = (intensity_f64 / norm).astype(np.float32)

    return {
        "identifier": spectrum.identifier,
        "precursor_mz": precursor_mz,
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

    Aligns ``min_mz`` and ``max_mz`` outward to the ``fragment_tol``
    grid via ``mz_bin_bounds``.
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
