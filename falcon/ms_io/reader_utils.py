"""Shared helpers for the per-format MS/MS peak-file readers.

These helpers live in a leaf module that imports only stdlib at module
load time so readers can share source normalization, identifier,
construction, and logging logic without circular or eager heavy imports.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

# Per-spectrum parsing exceptions recovered by skipping the entry.
PARSE_EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError)

__all__ = [
    "PARSE_EXCEPTIONS",
    "base_filename",
    "build_msms_spectrum",
    "iter_xml_spectra",
    "log_skipped_spectrum",
    "normalize_source",
]


def normalize_source(
    source: IO[Any] | str | os.PathLike[str],
) -> IO[Any] | str:
    """
    Convert ``os.PathLike`` inputs to ``str`` for Pyteomics readers.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Peak file path or open file object.

    Returns
    -------
    IO[Any] | str
        String path if ``source`` is ``os.PathLike``, otherwise
        ``source`` unchanged.
    """
    return os.fspath(source) if isinstance(source, os.PathLike) else source


def base_filename(
    source: IO[Any] | str | os.PathLike[str], reader: Any
) -> str:
    """
    Derive a base filename stem without directory or final extension.

    Used to build USI-style spectrum identifiers. Accepts path strings,
    ``PathLike`` objects, and open file objects; for file objects, falls
    back to the ``name`` attribute of ``source`` or of the Pyteomics
    ``reader``.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Peak file path or open file object passed to ``get_spectra``.
    reader : Any
        Open Pyteomics reader used as a fallback source for ``name``.

    Returns
    -------
    str
        Base filename stem, or ``"unknown"`` when no name is available.
    """
    if isinstance(source, (str, os.PathLike)):
        name: str | None = os.fspath(source)
    else:
        raw_name = getattr(source, "name", None) or getattr(
            reader, "name", None
        )
        name = os.fspath(raw_name) if raw_name else None
    if not name:
        return "unknown"
    return Path(name).stem


def build_msms_spectrum(
    identifier: str,
    precursor_mz: float,
    precursor_charge: int | None,
    mz: np.ndarray,
    intensity: np.ndarray,
    retention_time: float,
) -> sus.MsmsSpectrum:
    """
    Build a ``spectrum_utils`` ``MsmsSpectrum`` with nullable charge.

    Isolates the typing workaround required by pinned
    ``spectrum_utils==0.3.5``, whose signature annotates
    ``precursor_charge`` as ``int`` even though ``None`` is valid.

    Parameters
    ----------
    identifier : str
        Spectrum identifier.
    precursor_mz : float
        Precursor m/z value.
    precursor_charge : int | None
        Precursor charge state, or ``None`` when unknown.
    mz : np.ndarray
        Fragment m/z peaks.
    intensity : np.ndarray
        Fragment peak intensities.
    retention_time : float
        Retention time (or ``NaN`` when missing).

    Returns
    -------
    MsmsSpectrum
        Constructed ``spectrum_utils`` spectrum object.
    """
    import spectrum_utils.spectrum as sus

    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        cast("int", precursor_charge),
        mz,
        intensity,
        None,
        retention_time,
    )


def log_skipped_spectrum(
    source: IO[Any] | str | os.PathLike[str], identifier: Any, exc: Exception
) -> None:
    """
    Log a skipped unparsable spectrum at debug level.

    Debug logging avoids flooding stderr when large files contain many
    malformed spectra; whole-file read failures remain warnings in the
    individual format readers.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Peak file path or open file object being read.
    identifier : Any
        Spectrum identifier or title if available.
    exc : Exception
        Parsing exception that caused the spectrum to be skipped.
    """
    logger.debug(
        "Skipping unparsable spectrum %s in %s: %s", identifier, source, exc
    )


def iter_xml_spectra(
    source: IO[Any] | str | os.PathLike[str],
    reader_cls: type[Any],
    ms_level_key: str,
    extract_scan_nr: Callable[[dict[str, Any]], str | None],
    parse_spectrum: Callable[[dict[str, Any], str], sus.MsmsSpectrum],
    log: logging.Logger,
) -> Iterator[sus.MsmsSpectrum]:
    """
    Iterate MS/MS spectra (MS level > 1) from a Pyteomics XML reader.

    Shared driver for mzML and mzXML readers. Normalizes ``source``,
    filters out MS1 spectra, builds ``<stem>:scan:<number>`` identifiers
    without mutating the Pyteomics spectrum dictionaries, logs and skips
    unparsable spectra, and warns on XML parse errors.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Path or open file object for the XML peak source.
    reader_cls : type[Any]
        Pyteomics XML reader class (e.g. ``MzML`` or ``MzXML``).
    ms_level_key : str
        Dictionary key holding the MS level (``"ms level"`` for mzML,
        ``"msLevel"`` for mzXML).
    extract_scan_nr : Callable[[dict[str, Any]], str | None]
        Callback extracting the scan number string from a spectrum dict,
        or returning ``None`` to skip the spectrum.
    parse_spectrum : Callable[[dict[str, Any], str], sus.MsmsSpectrum]
        Callback converting ``(spectrum_dict, identifier)`` into an
        ``MsmsSpectrum``.
    log : logging.Logger
        Caller logger used to warn on whole-file XML read failures.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra from ``source``.
    """
    from lxml.etree import LxmlError

    xml_source = normalize_source(source)
    with reader_cls(xml_source) as f_in:
        filename = base_filename(source, f_in)
        try:
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get(ms_level_key, -1)) <= 1:
                        continue
                    scan_nr = extract_scan_nr(spectrum_dict)
                    if scan_nr is None:
                        continue
                    identifier = f"{filename}:scan:{scan_nr}"
                    yield parse_spectrum(spectrum_dict, identifier)
                except PARSE_EXCEPTIONS as exc:
                    log_skipped_spectrum(source, spectrum_dict.get("id"), exc)
        except LxmlError as exc:
            log.warning("Failed to read file %s: %s", source, exc)
