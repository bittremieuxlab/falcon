"""MGF spectrum reader and representative-spectrum writer."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pyteomics.mgf

from . import reader_utils

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

    import spectrum_utils.spectrum as sus

    from ..types import ConsensusTuple

# Matches standard Universal Spectrum Identifiers (USIs).
USI_PATTERN = re.compile(r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$")

__all__ = ["USI_PATTERN", "get_spectra", "write_spectra"]


def _resolve_identifier(
    params: dict[str, Any], base: str, spectrum_i: int
) -> str:
    """Resolve a spectrum identifier without mutating ``params``."""
    # Prefer embedded FILENAME param over the on-disk stem; avoid
    # re-stripping `base` so dotted filenames stay intact.
    raw_filename = params.get("filename")
    filename = Path(raw_filename).stem if raw_filename else base

    title = params.get("title")
    if isinstance(title, str) and (
        USI_PATTERN.match(title) or title.startswith(f"{filename}:cluster:")
    ):
        return title
    if "scans" in params:
        return f"{filename}:scan:{params['scans']}"
    if "scan" in params:
        return f"{filename}:scan:{params['scan']}"
    return f"{filename}:index:{spectrum_i}"


def get_spectra(
    source: IO[Any] | str | os.PathLike[str],
) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra from an MGF file or stream.

    Preserves USI titles and ``<filename>:cluster:`` titles; otherwise
    constructs an identifier from ``SCANS``/``SCAN`` or the 0-based
    spectrum index.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Path or open file object for the MGF source.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra; malformed entries are logged and skipped.
    """
    mgf_source = reader_utils.normalize_source(source)
    with pyteomics.mgf.MGF(mgf_source) as f_in:
        base = reader_utils.base_filename(source, f_in)

        for spectrum_i, spectrum_dict in enumerate(f_in):
            params = spectrum_dict.get("params", {})
            identifier = _resolve_identifier(params, base, spectrum_i)
            try:
                yield _parse_spectrum(spectrum_dict, identifier)
            except reader_utils.PARSE_EXCEPTIONS as exc:
                reader_utils.log_skipped_spectrum(source, identifier, exc)


def _parse_spectrum(
    spectrum_dict: dict[str, Any], identifier: str
) -> sus.MsmsSpectrum:
    """
    Parse a Pyteomics MGF spectrum dictionary into an ``MsmsSpectrum``.

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics MGF entry containing ``params``, ``m/z array``, and
        ``intensity array``.
    identifier : str
        Spectrum identifier assigned to the parsed spectrum.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing charge is ``None`` and missing RT is
        ``NaN``.
    """
    params = spectrum_dict["params"]
    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    retention_time = float(params.get("rtinseconds", float("nan")))

    precursor_mz = float(params["pepmass"][0])
    precursor_charge = int(params["charge"][0]) if "charge" in params else None

    return reader_utils.build_msms_spectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )


def write_spectra(
    filename: str | os.PathLike[str], spectra: Iterable[ConsensusTuple]
) -> None:
    """
    Write representative spectra to an MGF file.

    Parameters
    ----------
    filename : str | os.PathLike[str]
        Destination MGF file path.
    spectra : Iterable[ConsensusTuple]
        Representative spectra to serialize.
    """
    with Path(filename).open("w", encoding="utf-8") as f_out:
        pyteomics.mgf.write(
            _spectra_to_dicts(spectra),
            f_out,
            use_numpy=True,
        )


def _is_present(value: Any) -> bool:
    """Return True if a numeric metadata value is not None or NaN."""
    return value is not None and not math.isnan(float(value))


def _spectra_to_dicts(
    spectra: Iterable[ConsensusTuple],
) -> Iterator[dict[str, Any]]:
    """
    Convert representative spectra to Pyteomics MGF dictionaries.

    Parameters
    ----------
    spectra : Iterable[ConsensusTuple]
        Representative spectra to convert.

    Yields
    ------
    dict[str, Any]
        Pyteomics MGF dictionaries with intensities scaled to
        ``[0, 1000]`` and NaN charge/RT fields omitted.
    """
    for i, spectrum in enumerate(spectra):
        params: dict[str, Any] = {
            "title": f"falcon:cluster:{spectrum.cluster_id}",
            "scans": i + 1,
            "pepmass": float(spectrum.precursor_mz),
        }
        if _is_present(spectrum.precursor_charge):
            params["charge"] = int(spectrum.precursor_charge)
        if _is_present(spectrum.retention_time):
            params["rtinseconds"] = float(spectrum.retention_time)
        yield {
            "params": params,
            "m/z array": spectrum.mz,
            "intensity array": _scale_intensities(spectrum.intensity),
        }


def _scale_intensities(intensity: np.ndarray) -> np.ndarray:
    """
    Rescale positive peak intensities so the base peak equals 1000.

    Parameters
    ----------
    intensity : np.ndarray
        Float32 fragment peak intensities.

    Returns
    -------
    np.ndarray
        Rescaled intensities, or ``intensity`` unchanged if empty or
        non-positive.
    """
    if intensity.size == 0:
        return intensity
    max_i = float(np.max(intensity))
    if not math.isfinite(max_i) or max_i <= 0.0:
        return intensity
    return intensity * np.float32(1000.0 / max_i)
