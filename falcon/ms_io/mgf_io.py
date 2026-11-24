"""MGF spectrum reader and representative-spectrum writer."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pyteomics.mgf
from pyteomics.auxiliary import PyteomicsError

from . import reader_utils

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

    import spectrum_utils.spectrum as sus

    from ..types import ConsensusTuple

logger = logging.getLogger(__name__)

# Matches standard Universal Spectrum Identifiers (USIs).
USI_PATTERN = re.compile(r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$")

# Flush buffered serialized MGF blocks every ~1 MiB.
_WRITE_BUFFER_CHARS = 1 << 20

__all__ = ["USI_PATTERN", "get_spectra", "write_spectra"]


def _non_blank_scan(value: Any) -> str | None:
    """Return a stripped scan token, or ``None`` if absent or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_identifier(
    params: dict[str, Any], base: str, spectrum_i: int
) -> str:
    """Resolve a spectrum identifier without mutating ``params``."""
    # Prefer embedded FILENAME param over the on-disk stem; avoid
    # re-stripping `base` so dotted filenames stay intact.
    filename = reader_utils.path_stem(params.get("filename")) or base

    title = params.get("title")
    if isinstance(title, str) and (
        USI_PATTERN.match(title) or title.startswith(f"{filename}:cluster:")
    ):
        return title
    for key in ("scans", "scan"):
        scan = _non_blank_scan(params.get(key))
        if scan is not None:
            return f"{filename}:scan:{scan}"
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
    try:
        with pyteomics.mgf.MGF(
            mgf_source,
            read_charges=False,
            convert_arrays=1,
            dtype=np.float32,
        ) as f_in:
            base = reader_utils.base_filename(source, f_in)
            for spectrum_i, spectrum_dict in enumerate(f_in):
                identifier: Any = f"{base}:index:{spectrum_i}"
                try:
                    params = spectrum_dict["params"]
                    identifier = _resolve_identifier(params, base, spectrum_i)
                    spectrum = _parse_spectrum(spectrum_dict, identifier)
                except reader_utils.PARSE_EXCEPTIONS as exc:
                    reader_utils.log_skipped_spectrum(
                        source, identifier, exc, log=logger
                    )
                else:
                    yield spectrum
    except (PyteomicsError, ValueError) as exc:
        logger.warning("Failed to read file %s: %s", source, exc)


def _parse_retention_time(raw_rt: Any) -> float:
    """Parse an MGF ``RTINSECONDS`` scalar, range, or empty value."""
    if raw_rt is None or (isinstance(raw_rt, str) and not raw_rt.strip()):
        return float("nan")
    if isinstance(raw_rt, (tuple, list)):
        if not raw_rt:
            return float("nan")
        if len(raw_rt) >= 2:
            return (float(raw_rt[0]) + float(raw_rt[1])) / 2.0
        return float(raw_rt[0])
    return float(raw_rt)


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
    retention_time = _parse_retention_time(params.get("rtinseconds"))

    precursor_mz = float(params["pepmass"][0])
    raw_charge = params.get("charge")
    precursor_charge = int(raw_charge[0]) if raw_charge else None

    return reader_utils.build_msms_spectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )


def _is_present(value: Any) -> bool:
    """Return True if a numeric metadata value is present and finite."""
    return value is not None and math.isfinite(float(value))


def _format_spectrum(spectrum: ConsensusTuple, scan_nr: int) -> str:
    """
    Format a single representative spectrum as an MGF text block.

    Parameters
    ----------
    spectrum : ConsensusTuple
        Representative spectrum to serialize.
    scan_nr : int
        1-based sequential scan number assigned in the MGF output.

    Returns
    -------
    str
        Serialized MGF block with intensities scaled to ``[0, 1000]``
        and non-finite charge/RT fields omitted.
    """
    lines = [
        "BEGIN IONS",
        f"TITLE=falcon:cluster:{spectrum.cluster_id}",
        f"PEPMASS={float(spectrum.precursor_mz)}",
    ]
    if _is_present(spectrum.retention_time):
        lines.append(f"RTINSECONDS={float(spectrum.retention_time)}")
    if _is_present(spectrum.precursor_charge):
        charge = int(spectrum.precursor_charge)
        sign = "+" if charge >= 0 else "-"
        lines.append(f"CHARGE={abs(charge)}{sign}")
    lines.append(f"SCANS={scan_nr}")

    scaled_intensity = _scale_intensities(spectrum.intensity)
    lines.extend(
        f"{mz:.5f} {intensity:.1f}"
        for mz, intensity in zip(
            spectrum.mz.tolist(), scaled_intensity.tolist(), strict=True
        )
    )
    lines.append("END IONS")
    return "\n".join(lines) + "\n\n"


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
    buffer: list[str] = []
    buffered_chars = 0
    with Path(filename).open("w", encoding="utf-8") as f_out:
        for scan_nr, spectrum in enumerate(spectra, start=1):
            block = _format_spectrum(spectrum, scan_nr)
            buffer.append(block)
            buffered_chars += len(block)
            if buffered_chars >= _WRITE_BUFFER_CHARS:
                f_out.write("".join(buffer))
                buffer.clear()
                buffered_chars = 0
        if buffer:
            f_out.write("".join(buffer))


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
    return (intensity / np.float32(max_i)) * np.float32(1000.0)
