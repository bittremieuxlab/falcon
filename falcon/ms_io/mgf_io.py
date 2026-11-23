"""MGF spectrum reader and representative-spectrum writer."""

import math
import os
import re
from collections.abc import Iterable, Iterator
from typing import IO, Any, cast

import numba as nb
import numpy as np
import pyteomics.mgf
import spectrum_utils.spectrum as sus

from ..cluster import cluster
from . import reader_utils

# Matches standard Universal Spectrum Identifiers (USIs).
USI_PATTERN = re.compile(r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$")


def get_spectra(source: IO[Any] | str) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra from an MGF file or stream.

    Preserves USI titles and ``<filename>:cluster:`` titles; otherwise
    constructs an identifier from ``SCANS``/``SCAN`` or the 0-based
    spectrum index.

    Parameters
    ----------
    source : IO[Any] | str
        Path or open file object for the MGF source.

    Yields
    ------
    MsmsSpectrum
        Parsed MS/MS spectra; malformed entries are logged and skipped.
    """
    with pyteomics.mgf.MGF(source) as f_in:
        base = reader_utils.base_filename(source, f_in)

        for spectrum_i, spectrum_dict in enumerate(f_in):
            params = spectrum_dict.get("params", {})

            # Prefer embedded FILENAME param over the on-disk filename.
            filename = os.path.splitext(
                os.path.basename(params.get("filename", base))
            )[0]

            if "title" not in params or not (
                USI_PATTERN.match(params["title"])
                or params["title"].startswith(f"{filename}:cluster:")
            ):
                if "scans" in params:
                    usi = f"{filename}:scan:{params['scans']}"
                elif "scan" in params:
                    usi = f"{filename}:scan:{params['scan']}"
                else:
                    usi = f"{filename}:index:{spectrum_i}"
                params["title"] = usi

            try:
                yield _parse_spectrum(spectrum_dict)
            except (ValueError, KeyError, TypeError, IndexError) as e:
                reader_utils.log_skipped_spectrum(
                    source, params.get("title"), e
                )
                continue


def _parse_spectrum(spectrum_dict: dict[str, Any]) -> sus.MsmsSpectrum:
    """
    Parse a Pyteomics MGF spectrum dictionary into an ``MsmsSpectrum``.

    Parameters
    ----------
    spectrum_dict : dict[str, Any]
        Pyteomics MGF entry containing ``params``, ``m/z array``, and
        ``intensity array``.

    Returns
    -------
    MsmsSpectrum
        Parsed spectrum; missing charge is ``None`` and missing RT is
        ``NaN``.
    """
    identifier = spectrum_dict["params"]["title"]

    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    retention_time = float(
        spectrum_dict["params"].get("rtinseconds", float("nan"))
    )

    precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
    if "charge" in spectrum_dict["params"]:
        precursor_charge = int(spectrum_dict["params"]["charge"][0])
    else:
        precursor_charge = None

    # spectrum_utils 0.3.5 types precursor_charge as int; None is valid.
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        cast(int, precursor_charge),
        mz_array,
        intensity_array,
        None,
        retention_time,
    )


def write_spectra(
    filename: str, spectra: Iterable[cluster.ConsensusTuple]
) -> None:
    """
    Write representative spectra to an MGF file.

    Parameters
    ----------
    filename : str
        Destination MGF file path.
    spectra : Iterable[ConsensusTuple]
        Representative spectra to serialize.
    """
    with open(filename, "w") as f_out:
        pyteomics.mgf.write(
            _spectra_to_dicts(spectra),
            f_out,
            use_numpy=True,
        )


def _spectra_to_dicts(
    spectra: Iterable[cluster.ConsensusTuple],
) -> Iterator[dict[str, Any]]:
    """
    Convert representative spectra to Pyteomics MGF dictionaries.

    Parameters
    ----------
    spectra : Iterable[ConsensusTuple]
        Representative spectra to convert.

    Yields
    ------
    dict[str, Any]
        Pyteomics MGF dictionaries with intensities scaled to
        ``[0, 1000]`` and NaN charge/RT fields omitted.
    """
    for i, spectrum in enumerate(spectra):
        params: dict[str, Any] = {
            "title": f"falcon:cluster:{spectrum.cluster_id}",
            "scans": i + 1,
            "pepmass": float(spectrum.precursor_mz),
        }
        if spectrum.precursor_charge is not None and not math.isnan(
            float(spectrum.precursor_charge)
        ):
            params["charge"] = int(spectrum.precursor_charge)
        if spectrum.retention_time is not None and not math.isnan(
            float(spectrum.retention_time)
        ):
            params["rtinseconds"] = float(spectrum.retention_time)
        yield {
            "params": params,
            "m/z array": spectrum.mz,
            "intensity array": _scale_intensities(spectrum.intensity),
        }


@nb.njit(cache=True, fastmath=True)
def _scale_intensities(intensity: np.ndarray) -> np.ndarray:
    """
    Rescale positive peak intensities so the base peak equals 1000.

    Parameters
    ----------
    intensity : np.ndarray
        Float32 fragment peak intensities.

    Returns
    -------
    np.ndarray
        Rescaled intensities, or ``intensity`` unchanged if empty or
        non-positive.
    """
    max_i = np.max(intensity) if intensity.size else np.float32(0.0)
    if max_i <= 0.0:
        return intensity
    return intensity * (np.float32(1000.0) / max_i)
