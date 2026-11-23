"""mzXML MS/MS spectrum reader."""

from __future__ import annotations

import logging
from typing import IO, TYPE_CHECKING, Any

import pyteomics.mzxml

from . import reader_utils

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

__all__ = ["get_spectra"]


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

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra with ``<stem>:scan:<number>`` identifiers;
        malformed spectra are logged and skipped.
    """
    yield from reader_utils.iter_xml_spectra(
        source,
        pyteomics.mzxml.MzXML,
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

    Uses the first precursor entry and forwards Pyteomics's parsed
    ``retentionTime`` directly (or ``NaN`` when absent).

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
    retention_time = spectrum_dict.get("retentionTime", float("nan"))

    precursor = spectrum_dict["precursorMz"][0]
    precursor_mz = precursor["precursorMz"]
    precursor_charge = (
        int(precursor["precursorCharge"])
        if "precursorCharge" in precursor
        else None
    )

    return reader_utils.build_msms_spectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
