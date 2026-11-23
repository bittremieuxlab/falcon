"""Shared helpers for the per-format spectrum readers.

These helpers live in a leaf module that imports none of the format
readers so readers can import them without creating a cycle with
``ms_io``.
"""

import logging
import os
from typing import IO, Any

logger = logging.getLogger("falcon")


def base_filename(source: IO[Any] | str, reader: Any) -> str:
    """
    Derive a base filename without directory or extension.

    Used to build USI-style spectrum identifiers. Accepts both path
    strings and open file objects; for file objects, falls back to the
    ``name`` attribute of ``source`` or of the Pyteomics ``reader``.

    Parameters
    ----------
    source : IO[Any] | str
        Peak file path or open file object passed to ``get_spectra``.
    reader : Any
        Open Pyteomics reader used as a fallback source for ``name``.

    Returns
    -------
    str
        Base filename stem, or ``"unknown"`` when no name is available.
    """
    if isinstance(source, str):
        name = source
    else:
        name = getattr(source, "name", None) or getattr(reader, "name", None)
    if not name:
        return "unknown"
    return os.path.splitext(os.path.basename(name))[0]


def log_skipped_spectrum(
    source: IO[Any] | str, identifier: Any, exc: Exception
) -> None:
    """
    Log a skipped unparsable spectrum at debug level.

    Debug logging avoids flooding stderr when large files contain many
    malformed spectra; whole-file read failures remain warnings in the
    individual format readers.

    Parameters
    ----------
    source : IO[Any] | str
        Peak file path or open file object being read.
    identifier : Any
        Spectrum identifier or title if available.
    exc : Exception
        Parsing exception that caused the spectrum to be skipped.
    """
    logger.debug(
        "Skipping unparsable spectrum %s in %s: %s", identifier, source, exc
    )
