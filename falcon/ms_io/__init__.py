"""Peak file readers and format-agnostic dispatch (MGF, mzML, mzXML)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from spectrum_utils.spectrum import MsmsSpectrum

from ..types import ConsensusTuple
from . import mgf_io, mzml_io, mzxml_io

_READERS = {
    ".mgf": mgf_io,
    ".mzml": mzml_io,
    ".mzxml": mzxml_io,
}

__all__ = ["get_spectra", "mgf_io", "mzml_io", "mzxml_io", "write_spectra"]


def get_spectra(filename: str | os.PathLike[str]) -> Iterator[MsmsSpectrum]:
    """
    Read MS/MS spectra from a peak file based on its extension.

    Supported formats are MGF (``.mgf``), mzML (``.mzml``), and mzXML
    (``.mzxml``), matched case-insensitively.

    Parameters
    ----------
    filename : str | os.PathLike[str]
        Path to the input peak file.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra from ``filename``.

    Raises
    ------
    ValueError
        If ``filename`` does not exist or has an unsupported extension.
    """
    path = Path(filename)
    if not path.is_file():
        raise ValueError(f"Non-existing peak file {filename}")

    ext = path.suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    yield from reader.get_spectra(path)


def write_spectra(
    filename: str | os.PathLike[str], spectra: Iterable[ConsensusTuple]
) -> None:
    """
    Write representative spectra to a peak file.

    Only MGF (``.mgf``) output is supported.

    Parameters
    ----------
    filename : str | os.PathLike[str]
        Destination peak file path.
    spectra : Iterable[ConsensusTuple]
        Representative spectra to serialize.

    Raises
    ------
    ValueError
        If ``filename`` does not have a ``.mgf`` extension.
    """
    path = Path(filename)
    if path.suffix.lower() != ".mgf":
        raise ValueError(
            "Unsupported peak file format (supported formats: MGF)"
        )

    mgf_io.write_spectra(path, spectra)
