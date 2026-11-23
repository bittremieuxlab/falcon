"""mzXML MS/MS spectrum reader."""

import logging
from collections.abc import Iterator
from typing import IO, Any, cast

import pyteomics.mzxml
import spectrum_utils.spectrum as sus
from lxml.etree import LxmlError

from . import reader_utils

logger = logging.getLogger("falcon")


def get_spectra(source: IO[Any] | str) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra (MS level > 1) from an mzXML file or stream.

    Parameters
    ----------
    source : IO[Any] | str
        Path or open file object for the mzXML source.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra with ``<stem>:scan:<number>`` identifiers;
        malformed spectra are logged and skipped.
    """
    with pyteomics.mzxml.MzXML(source) as f_in:
        filename = reader_utils.base_filename(source, f_in)
        try:
            for spectrum_dict in f_in:
                if int(spectrum_dict.get("msLevel", -1)) > 1:
                    # Build a USI-style identifier from the scan number.
                    scan_nr = int(spectrum_dict["id"])
                    spectrum_dict["id"] = f"{filename}:scan:{scan_nr}"
                    try:
                        yield _parse_spectrum(spectrum_dict)
                    except (ValueError, KeyError) as e:
                        reader_utils.log_skipped_spectrum(
                            source, spectrum_dict["id"], e
                        )
        except LxmlError as e:
            logger.warning("Failed to read file %s: %s", source, e)


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

    precursor_mz = spectrum_dict["precursorMz"][0]["precursorMz"]
    if "precursorCharge" in spectrum_dict["precursorMz"][0]:
        precursor_charge = int(
            spectrum_dict["precursorMz"][0]["precursorCharge"]
        )
    else:
        precursor_charge = None

    # spectrum_utils 0.3.5 types precursor_charge as int; None is valid.
    return sus.MsmsSpectrum(
        spectrum_id,
        precursor_mz,
        cast(int, precursor_charge),
        mz_array,
        intensity_array,
        None,
        retention_time,
    )
