"""Shared helpers for the per-format MS/MS peak-file readers.

These helpers live in a leaf module that imports only stdlib at module
load time so readers can share source normalization, identifier,
construction, and logging logic without circular or eager heavy imports.
"""

from __future__ import annotations

import binascii
import functools
import logging
import math
import os
import zlib
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    import spectrum_utils.spectrum as sus

logger = logging.getLogger(__name__)

# Per-spectrum parsing exceptions recovered by skipping the entry.
PARSE_EXCEPTIONS = (
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    OverflowError,
    AttributeError,
)

__all__ = [
    "PARSE_EXCEPTIONS",
    "base_filename",
    "build_msms_spectrum",
    "iter_xml_spectra",
    "log_skipped_spectrum",
    "normalize_source",
    "path_stem",
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


def path_stem(value: Any) -> str | None:
    """
    Return the non-empty filename stem of a path-like value.

    Parameters
    ----------
    value : Any
        Candidate path (``str``, ``bytes``, ``os.PathLike``) or
        arbitrary object.

    Returns
    -------
    str | None
        Filename stem without its final suffix, or ``None`` if
        ``value`` is not path-like or yields an empty stem.
    """
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    return Path(os.fsdecode(os.fspath(value))).stem or None


def base_filename(
    source: IO[Any] | str | bytes | os.PathLike[Any], reader: Any
) -> str:
    """
    Derive a base filename stem without directory or final extension.

    Used to build USI-style spectrum identifiers. Accepts path strings,
    bytes, ``PathLike`` objects, and open file objects; for file
    objects, falls back to the ``name`` attribute of ``source`` or of
    the Pyteomics ``reader`` when it is path-like.

    Parameters
    ----------
    source : IO[Any] | str | bytes | os.PathLike[Any]
        Peak file path or open file object passed to ``get_spectra``.
    reader : Any
        Open Pyteomics reader used as a fallback source for ``name``.

    Returns
    -------
    str
        Base filename stem, or ``"unknown"`` when no non-empty stem is
        available.
    """
    candidates: tuple[Any, ...]
    if isinstance(source, (str, bytes, os.PathLike)):
        candidates = (source,)
    else:
        candidates = (
            getattr(source, "name", None),
            getattr(reader, "name", None),
        )
    for candidate in candidates:
        stem = path_stem(candidate)
        if stem is not None:
            return stem
    return "unknown"


@functools.lru_cache(maxsize=1)
def _msms_spectrum_cls() -> type[sus.MsmsSpectrum]:
    """Lazily import and cache ``spectrum_utils`` ``MsmsSpectrum``."""
    import spectrum_utils.spectrum as sus

    return sus.MsmsSpectrum


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

    Coerces ``precursor_mz`` and ``retention_time`` to plain ``float``
    (rejecting non-numeric values and non-finite ``precursor_mz`` while
    preserving ``NaN`` retention times) and isolates the typing
    workaround required by pinned ``spectrum_utils==0.3.5``, whose
    signature annotates ``precursor_charge`` as ``int`` even though
    ``None`` is valid.

    Parameters
    ----------
    identifier : str
        Spectrum identifier.
    precursor_mz : float
        Finite precursor m/z value.
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

    Raises
    ------
    ValueError
        If ``precursor_mz`` is non-finite, ``retention_time`` is
        infinite, or numeric conversion fails.
    """
    precursor_mz_val = float(precursor_mz)
    if not math.isfinite(precursor_mz_val):
        raise ValueError(
            f"Precursor m/z must be finite, got {precursor_mz_val!r}"
        )
    retention_time_val = float(retention_time)
    if math.isinf(retention_time_val):
        raise ValueError(
            f"Retention time must be finite or NaN, got {retention_time_val!r}"
        )
    return _msms_spectrum_cls()(
        identifier,
        precursor_mz_val,
        cast("int", precursor_charge),
        mz,
        intensity,
        None,
        retention_time_val,
    )


def log_skipped_spectrum(
    source: IO[Any] | str | os.PathLike[str],
    identifier: Any,
    exc: Exception,
    log: logging.Logger = logger,
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
    log : logging.Logger, optional
        Logger receiving the debug record. Defaults to this module's
        logger.
    """
    log.debug(
        "Skipping unparsable spectrum %s in %s: %s", identifier, source, exc
    )


def iter_xml_spectra(
    source: IO[Any] | str | os.PathLike[str],
    reader_cls: Callable[[IO[Any] | str], Any],
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
    unparsable spectra, and warns on XML or binary-decoding errors.

    Parameters
    ----------
    source : IO[Any] | str | os.PathLike[str]
        Path or open file object for the XML peak source.
    reader_cls : Callable[[IO[Any] | str], Any]
        Pyteomics XML reader class or opener callable (e.g. ``MzML``,
        ``MzXML``, or a thin wrapper configuring reader options).
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
        Caller logger used for debug skip logs and whole-file warnings.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra from ``source``.
    """
    from lxml.etree import LxmlError
    from pyteomics.auxiliary import PyteomicsError

    xml_source = normalize_source(source)
    try:
        with reader_cls(xml_source) as f_in:
            filename = base_filename(source, f_in)
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get(ms_level_key, -1)) <= 1:
                        continue
                    scan_nr = extract_scan_nr(spectrum_dict)
                    if scan_nr is None:
                        continue
                    identifier = f"{filename}:scan:{scan_nr}"
                    spectrum = parse_spectrum(spectrum_dict, identifier)
                except PARSE_EXCEPTIONS as exc:
                    log_skipped_spectrum(
                        source, spectrum_dict.get("id"), exc, log=log
                    )
                else:
                    yield spectrum
    except (
        LxmlError,
        PyteomicsError,
        KeyError,
        ValueError,
        binascii.Error,
        zlib.error,
    ) as exc:
        log.warning("Failed to read file %s: %s", source, exc)
