"""mzML MS/MS spectrum reader."""

from __future__ import annotations

import functools
import logging
import re
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pyteomics.mzml
import pyteomics.xml

from . import reader_utils

if TYPE_CHECKING:
    from collections.abc import Iterator

    import spectrum_utils.spectrum as sus

    from .reader_utils import SourceType

logger = logging.getLogger(__name__)

_PSI_MS_OBO_URI = "http://purl.obolibrary.org/obo/ms/psi-ms.obo"
_SCAN_NR_RE = re.compile(r"(?:^|\s)scan=(\S+)")

__all__ = ["get_spectra"]


@functools.lru_cache(maxsize=1)
def _load_offline_psims_cv() -> Any:
    """Load and cache the vendored PSI-MS CV without network access."""
    try:
        from psims.controlled_vocabulary.controlled_vocabulary import OBOCache

        return OBOCache(enabled=False, use_remote=False).load(_PSI_MS_OBO_URI)
    except Exception as exc:
        logger.debug("Could not preload offline PSI-MS CV: %s", exc)
        return None


def _open_mzml(xml_source: IO[Any] | str) -> Any:
    """Open an mzML source with non-indexed float32 offline settings."""
    kwargs: dict[str, Any] = {"use_index": False, "dtype": np.float32}
    if hasattr(pyteomics.xml, "CVParamParser"):
        cv = _load_offline_psims_cv()
        if cv is not None:
            kwargs["cv"] = cv
    return pyteomics.mzml.MzML(xml_source, **kwargs)


def _extract_scan_nr(spectrum_dict: dict[str, Any]) -> str | None:
    """Extract the ``scan=`` token from an mzML spectrum dictionary."""
    raw_id = spectrum_dict.get("id")
    if not isinstance(raw_id, str):
        return None
    match = _SCAN_NR_RE.search(raw_id)
    return match.group(1) if match is not None else None


def get_spectra(source: SourceType) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra (MS level > 1) from an mzML file or stream.

    Parameters
    ----------
    source : SourceType
        Path or open file object for the mzML source.

    Returns
    -------
    Iterator[MsmsSpectrum]
        Parsed MS/MS spectra with ``<stem>:scan:<number>`` identifiers;
        spectra lacking ``scan=`` or failing to parse are skipped.
    """
    warned_missing_scan = False

    def _extract_scan_nr_logged(spectrum_dict: dict[str, Any]) -> str | None:
        nonlocal warned_missing_scan
        scan_nr = _extract_scan_nr(spectrum_dict)
        if scan_nr is None:
            log_fn = logger.debug if warned_missing_scan else logger.warning
            warned_missing_scan = True
            log_fn(
                "Skipping spectrum with unrecognized id "
                "(no 'scan=' field) in %s: %s",
                source,
                spectrum_dict.get("id"),
            )
        return scan_nr

    return reader_utils.iter_xml_spectra(
        source,
        _open_mzml,
        "ms level",
        _extract_scan_nr_logged,
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
    Multi-valued charge sequences use the first entry, matching MGF
    behavior, and negative-mode scans negate the charge magnitude.

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics mzML spectrum entry.
    identifier : str
        USI-style spectrum identifier assigned to the parsed spectrum.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing or blank RT is stored as ``NaN``.
    """
    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    scans = (spectrum_dict.get("scanList") or {}).get("scan")
    raw_rt = scans[0].get("scan start time") if scans else None
    retention_time = reader_utils.parse_optional_rt(raw_rt)

    precursor = spectrum_dict["precursorList"]["precursor"][0]
    precursor_ion = precursor["selectedIonList"]["selectedIon"][0]
    precursor_mz = precursor_ion["selected ion m/z"]
    negative = "negative scan" in spectrum_dict
    precursor_charge = reader_utils.parse_optional_charge(
        precursor_ion.get("charge state"), negative=negative
    )
    if precursor_charge is None:
        precursor_charge = reader_utils.parse_optional_charge(
            precursor_ion.get("possible charge state"), negative=negative
        )

    return reader_utils.build_msms_spectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        retention_time,
    )
