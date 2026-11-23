"""mzXML MS/MS spectrum reader."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import IO, Any

import pyteomics.mzxml
import spectrum_utils.spectrum as sus
from lxml.etree import LxmlError

from . import reader_utils

logger = logging.getLogger("falcon")


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
    mzxml_source: IO[Any] | str = (
        os.fspath(source) if isinstance(source, os.PathLike) else source
    )
    with pyteomics.mzxml.MzXML(mzxml_source) as f_in:
        filename = reader_utils.base_filename(source, f_in)
        try:
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get("msLevel", -1)) <= 1:
                        continue
                    scan_nr = int(spectrum_dict["id"])
                    spectrum_dict["id"] = f"{filename}:scan:{scan_nr}"
                    yield _parse_spectrum(spectrum_dict)
                except reader_utils.PARSE_EXCEPTIONS as exc:
                    reader_utils.log_skipped_spectrum(
                        source, spectrum_dict.get("id"), exc
                    )
        except LxmlError as exc:
            logger.warning("Failed to read file %s: %s", source, exc)


def _parse_spectrum(spectrum_dict: dict[str, Any]) -> sus.MsmsSpectrum:
    """
    Parse a Pyteomics mzXML spectrum dict into an ``MsmsSpectrum``.

    Uses the first precursor entry and forwards Pyteomics's parsed
    ``retentionTime`` directly (or ``NaN`` when absent).

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics mzXML spectrum entry.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing charge is ``None``.
    """
    spectrum_id = spectrum_dict["id"]
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
        spectrum_id,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
