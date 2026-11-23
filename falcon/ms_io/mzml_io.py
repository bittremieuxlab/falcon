"""mzML MS/MS spectrum reader."""

from __future__ import annotations

import logging
from typing import IO, TYPE_CHECKING, Any

import pyteomics.mzml

from . import reader_utils

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

__all__ = ["get_spectra"]


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

    def _extract_scan_nr(spectrum_dict: dict[str, Any]) -> str | None:
        raw_id = spectrum_dict["id"]
        scan_idx = raw_id.find("scan=")
        if scan_idx == -1:
            logger.warning(
                "Skipping spectrum with unrecognized id "
                "(no 'scan=' field) in %s: %s",
                source,
                raw_id,
            )
            return None
        return raw_id[scan_idx + 5 :]

    yield from reader_utils.iter_xml_spectra(
        source,
        pyteomics.mzml.MzML,
        "ms level",
        _extract_scan_nr,
        _parse_spectrum,
        logger,
    )


def _parse_spectrum(
    spectrum_dict: dict[str, Any], identifier: str
) -> sus.MsmsSpectrum:
    """
    Parse a Pyteomics mzML spectrum dictionary into an ``MsmsSpectrum``.

    Uses the first precursor and selected ion, falling back from
    ``charge state`` to ``possible charge state`` before ``None``.

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics mzML spectrum entry.
    identifier : str
        USI-style spectrum identifier assigned to the parsed spectrum.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing RT is stored as ``NaN``.
    """
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
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
