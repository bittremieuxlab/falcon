"""Shared helpers for the per-format MS/MS peak-file readers.

These helpers live in a leaf module that imports none of the format
readers so readers can share identifier, construction, and logging logic
without circular imports.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
import spectrum_utils.spectrum as sus

logger = logging.getLogger("falcon")

# Per-spectrum parsing exceptions recovered by skipping the entry.
PARSE_EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError)


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
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        cast(int, precursor_charge),
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
