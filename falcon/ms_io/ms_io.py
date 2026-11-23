"""Format-agnostic dispatch for reading and writing MS/MS peak files."""

import os
from collections.abc import Iterable, Iterator

from spectrum_utils.spectrum import MsmsSpectrum

from ..cluster.consensus import ConsensusTuple
from . import mgf_io, mzml_io, mzxml_io


def get_spectra(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read MS/MS spectra from a peak file based on its extension.

    Supported formats are MGF (``.mgf``), mzML (``.mzml``), and mzXML
    (``.mzxml``), matched case-insensitively.

    Parameters
    ----------
    filename : str
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
    if not os.path.isfile(filename):
        raise ValueError(f"Non-existing peak file {filename}")

    _, ext = os.path.splitext(filename.lower())
    if ext == ".mgf":
        spectrum_io = mgf_io
    elif ext == ".mzml":
        spectrum_io = mzml_io
    elif ext == ".mzxml":
        spectrum_io = mzxml_io
    else:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    yield from spectrum_io.get_spectra(filename)


def write_spectra(filename: str, spectra: Iterable[ConsensusTuple]) -> None:
    """
    Write representative spectra to a peak file.

    Only MGF (``.mgf``) output is supported.

    Parameters
    ----------
    filename : str
        Destination peak file path.
    spectra : Iterable[ConsensusTuple]
        Representative spectra to serialize.

    Raises
    ------
    ValueError
        If ``filename`` does not have a ``.mgf`` extension.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".mgf":
        spectrum_io = mgf_io
    else:
        raise ValueError(
            "Unsupported peak file format (supported formats: MGF)"
        )

    spectrum_io.write_spectra(filename, spectra)
