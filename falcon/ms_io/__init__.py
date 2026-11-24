"""Peak file readers and format-agnostic dispatch (MGF, mzML, mzXML)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import spectrum_utils.spectrum as sus

    from ..types import ConsensusTuple, PathLike
    from . import mgf_io, mzml_io, mzxml_io

_LAZY_SUBMODULES = frozenset({"mgf_io", "mzml_io", "mzxml_io"})

__all__ = ["get_spectra", "mgf_io", "mzml_io", "mzxml_io", "write_spectra"]


def __getattr__(name: str) -> Any:
    """Lazily load and cache format-specific submodules."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed submodules in ``dir(ms_io)``."""
    return sorted({*globals(), *__all__})


def get_spectra(filename: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra from a peak file based on its extension.

    Supported formats are MGF (``.mgf``), mzML (``.mzml``), and mzXML
    (``.mzxml``), matched case-insensitively.

    Parameters
    ----------
    filename : PathLike
        Path to the input peak file.

    Returns
    -------
    Iterator[MsmsSpectrum]
        Lazy iterator over parsed MS/MS spectra from ``filename``.

    Raises
    ------
    IsADirectoryError
        If ``filename`` refers to a directory.
    FileNotFoundError
        If ``filename`` does not exist or is not a regular file.
    ValueError
        If ``filename`` has an unsupported extension.
    """
    path = Path(filename)
    if path.is_dir():
        raise IsADirectoryError(
            f"Peak file path is a directory, not a file: {filename}"
        )
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
    filename: PathLike, spectra: Iterable[ConsensusTuple]
) -> None:
    """
    Write representative spectra to a peak file.

    Only MGF (``.mgf``) output is supported.

    Parameters
    ----------
    filename : PathLike
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
