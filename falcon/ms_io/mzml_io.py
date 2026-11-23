"""mzML MS/MS spectrum reader."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import IO, Any

import pyteomics.mzml
import spectrum_utils.spectrum as sus
from lxml.etree import LxmlError

from . import reader_utils

logger = logging.getLogger("falcon")


def get_spectra(
    source: IO[Any] | str | os.PathLike[str],
) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra (MS level > 1) from an mzML file or stream.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Path or open file object for the mzML source.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra with ``<stem>:scan:<number>`` identifiers;
        spectra lacking ``scan=`` or failing to parse are skipped.
    """
    mzml_source: IO[Any] | str = (
        os.fspath(source) if isinstance(source, os.PathLike) else source
    )
    with pyteomics.mzml.MzML(mzml_source) as f_in:
        filename = reader_utils.base_filename(source, f_in)
        try:
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get("ms level", -1)) <= 1:
                        continue
                    raw_id = spectrum_dict["id"]
                    scan_idx = raw_id.find("scan=")
                    if scan_idx == -1:
                        logger.warning(
                            "Skipping spectrum with unrecognized id "
                            "(no 'scan=' field) in %s: %s",
                            source,
                            raw_id,
                        )
                        continue
                    scan_nr = raw_id[scan_idx + 5 :]
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
    Parse a Pyteomics mzML spectrum dictionary into an ``MsmsSpectrum``.

    Uses the first precursor and selected ion, falling back from
    ``charge state`` to ``possible charge state`` before ``None``.

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics mzML spectrum entry.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing RT is stored as ``NaN``.
    """
    spectrum_id = spectrum_dict["id"]
    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    retention_time = spectrum_dict["scanList"]["scan"][0].get(
        "scan start time", float("nan")
    )

    precursor = spectrum_dict["precursorList"]["precursor"][0]
    precursor_ion = precursor["selectedIonList"]["selectedIon"][0]
    precursor_mz = precursor_ion["selected ion m/z"]
    if "charge state" in precursor_ion:
        precursor_charge: int | None = int(precursor_ion["charge state"])
    elif "possible charge state" in precursor_ion:
        precursor_charge = int(precursor_ion["possible charge state"])
    else:
        precursor_charge = None

    return reader_utils.build_msms_spectrum(
        spectrum_id,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
