"""Peak file readers and format-agnostic dispatch (MGF, mzML, mzXML)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator

    from spectrum_utils.spectrum import MsmsSpectrum

    from ..types import ConsensusTuple
    from . import mgf_io, mzml_io, mzxml_io

_LAZY_SUBMODULES = frozenset({"mgf_io", "mzml_io", "mzxml_io"})

__all__ = ["get_spectra", "mgf_io", "mzml_io", "mzxml_io", "write_spectra"]


def __getattr__(name: str) -> Any:
    """Lazily load format-specific reader/writer submodules."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_spectra(filename: str | os.PathLike[str]) -> Iterator[MsmsSpectrum]:
    """
    Read MS/MS spectra from a peak file based on its extension.

    Supported formats are MGF (``.mgf``), mzML (``.mzml``), and mzXML
    (``.mzxml``), matched case-insensitively.

    Parameters
    ----------
    filename : str | os.PathLike[str]
        Path to the input peak file.

    Returns
    -------
    Iterator[MsmsSpectrum]
        Lazy iterator over parsed MS/MS spectra from ``filename``.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist or is not a regular file.
    ValueError
        If ``filename`` has an unsupported extension.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Non-existing peak file {filename}")

    ext = path.suffix.lower()
    if ext == ".mgf":
        from . import mgf_io

        return mgf_io.get_spectra(path)
    if ext == ".mzml":
        from . import mzml_io

        return mzml_io.get_spectra(path)
    if ext == ".mzxml":
        from . import mzxml_io

        return mzxml_io.get_spectra(path)
    raise ValueError(f'Unknown spectrum file type with extension "{ext}"')


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

    from . import mgf_io

    mgf_io.write_spectra(path, spectra)
