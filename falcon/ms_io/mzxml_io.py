"""mzXML MS/MS spectrum reader."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pyteomics.mzxml

from . import reader_utils

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

__all__ = ["get_spectra"]


def _open_mzxml(xml_source: IO[Any] | str) -> Any:
    """Open an mzXML source with non-indexed float32 settings."""
    return pyteomics.mzxml.MzXML(xml_source, use_index=False, dtype=np.float32)


def _extract_scan_nr(spectrum_dict: dict[str, Any]) -> str:
    """Extract the integer scan number from an mzXML spectrum dict."""
    return str(int(spectrum_dict["id"]))


def get_spectra(
    source: IO[Any] | str | os.PathLike[str],
) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra (MS level > 1) from an mzXML file or stream.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Path or open file object for the mzXML source.

    Returns
    -------
    Iterator[MsmsSpectrum]
        Parsed MS/MS spectra with ``<stem>:scan:<number>`` identifiers;
        malformed spectra are logged and skipped.
    """
    return reader_utils.iter_xml_spectra(
        source,
        _open_mzxml,
        "msLevel",
        _extract_scan_nr,
        _parse_spectrum,
        logger,
    )


def _parse_spectrum(
    spectrum_dict: dict[str, Any], identifier: str
) -> sus.MsmsSpectrum:
    """
    Parse a Pyteomics mzXML spectrum dict into an ``MsmsSpectrum``.

    Uses the first precursor entry, negates the charge magnitude for
    negative-mode scans (``polarity="-"``), and forwards Pyteomics's
    parsed ``retentionTime`` directly (or ``NaN`` when absent).

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics mzXML spectrum entry.
    identifier : str
        USI-style spectrum identifier assigned to the parsed spectrum.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing charge is ``None``.
    """
    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    raw_rt = spectrum_dict.get("retentionTime")
    retention_time = (
        float("nan")
        if raw_rt is None or (isinstance(raw_rt, str) and not raw_rt.strip())
        else raw_rt
    )

    precursor = spectrum_dict["precursorMz"][0]
    if isinstance(precursor, Mapping):
        precursor_mz = precursor["precursorMz"]
        raw_charge = precursor.get("precursorCharge")
        if raw_charge is None or (
            isinstance(raw_charge, str) and not raw_charge.strip()
        ):
            precursor_charge = None
        else:
            precursor_charge = int(raw_charge)
    else:
        precursor_mz = precursor
        precursor_charge = None
    if precursor_charge is not None and spectrum_dict.get("polarity") == "-":
        precursor_charge = -abs(precursor_charge)

    return reader_utils.build_msms_spectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
