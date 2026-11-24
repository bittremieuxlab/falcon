"""Peak file readers (MGF, mzML, mzXML) and MGF writer."""

from __future__ import annotations

import binascii
import functools
import logging
import math
import os
import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from .types import PathLike
from .utils import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import numpy as np
    import spectrum_utils.spectrum as sus

    from .types import ConsensusTuple

logger = logging.getLogger(__name__)

SourceType = IO[Any] | PathLike

PARSE_EXCEPTIONS = (
    ValueError,
    KeyError,
    TypeError,
    IndexError,
    OverflowError,
    AttributeError,
)

_USI_PATTERN = re.compile(
    r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$"
)
_SCAN_NR_RE = re.compile(r"(?:^|\s)scan=(\S+)")
_PSI_MS_OBO_URI = "http://purl.obolibrary.org/obo/ms/psi-ms.obo"

__all__ = [
    "PARSE_EXCEPTIONS",
    "SourceType",
    "get_spectra",
    "read_mgf",
    "read_mzml",
    "read_mzxml",
    "write_spectra",
]


def _normalize_source(source: SourceType) -> IO[Any] | str:
    return os.fspath(source) if isinstance(source, os.PathLike) else source


def _path_stem(value: Any) -> str | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    stem = Path(value).stem
    return stem if stem.strip() else None


def _base_filename(source: SourceType, reader: Any) -> str:
    candidates: tuple[Any, ...] = (
        (source,)
        if isinstance(source, (str, os.PathLike))
        else (getattr(source, "name", None), getattr(reader, "name", None))
    )
    for candidate in candidates:
        stem = _path_stem(candidate)
        if stem is not None:
            return stem
    return "unknown"


def _parse_rt(raw_rt: Any) -> float:
    if isinstance(raw_rt, (tuple, list)):
        if not raw_rt:
            return float("nan")
        if len(raw_rt) == 2:
            lo, hi = float(raw_rt[0]), float(raw_rt[1])
            if math.isinf(lo) or math.isinf(hi):
                raise ValueError("Retention time range must be finite or NaN")
            return (lo + hi) / 2.0
        if len(raw_rt) != 1:
            raise ValueError(
                f"Expected 1 or 2 retention time values, got {len(raw_rt)}"
            )
        raw_rt = raw_rt[0]
    if raw_rt is None or (isinstance(raw_rt, str) and not raw_rt.strip()):
        return float("nan")
    val = float(raw_rt)
    if math.isinf(val):
        raise ValueError(f"Retention time must be finite or NaN, got {val!r}")
    return val


def _parse_charge(raw_charge: Any, *, negative: bool = False) -> int | None:
    if isinstance(raw_charge, (list, tuple)):
        if not raw_charge:
            return None
        raw_charge = raw_charge[0]
    if raw_charge is None or (
        isinstance(raw_charge, str) and not raw_charge.strip()
    ):
        return None
    if isinstance(raw_charge, float):
        if not math.isfinite(raw_charge) or not raw_charge.is_integer():
            raise ValueError(f"Non-integral precursor charge: {raw_charge!r}")
        charge = int(raw_charge)
    else:
        charge = int(raw_charge)
    return -abs(charge) if negative else charge


def _build_spectrum(
    identifier: str,
    precursor_mz: Any,
    precursor_charge: int | None,
    mz: np.ndarray,
    intensity: np.ndarray,
    retention_time: float,
) -> sus.MsmsSpectrum:
    import spectrum_utils.spectrum as sus

    mz_val = float(precursor_mz)
    if not math.isfinite(mz_val):
        raise ValueError(f"Precursor m/z must be finite, got {mz_val!r}")
    if math.isinf(retention_time):
        raise ValueError(
            f"Retention time must be finite or NaN, got {retention_time!r}"
        )
    return sus.MsmsSpectrum(
        identifier,
        mz_val,
        cast("int", precursor_charge),
        mz,
        intensity,
        None,
        retention_time,
    )


def _iter_xml(
    source: SourceType,
    opener: Callable[[IO[Any] | str], Any],
    ms_level_key: str,
    parse_entry: Callable[[dict[str, Any], str], sus.MsmsSpectrum | None],
) -> Iterator[sus.MsmsSpectrum]:
    from lxml.etree import LxmlError
    from pyteomics.auxiliary import PyteomicsError

    try:
        with opener(_normalize_source(source)) as f_in:
            filename = _base_filename(source, f_in)
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get(ms_level_key, -1)) <= 1:
                        continue
                    spec = parse_entry(spectrum_dict, filename)
                except PARSE_EXCEPTIONS as exc:
                    logger.debug(
                        "Skipping unparsable spectrum %s in %s: %s",
                        spectrum_dict.get("id")
                        if isinstance(spectrum_dict, Mapping)
                        else None,
                        source,
                        exc,
                    )
                else:
                    if spec is not None:
                        yield spec
    except (
        LxmlError,
        PyteomicsError,
        KeyError,
        ValueError,
        binascii.Error,
        zlib.error,
    ) as exc:
        logger.warning("Failed to read file %s: %s", source, exc)


# ----------------------------------------------------------------------
# MGF reader and writer
# ----------------------------------------------------------------------


def _resolve_mgf_id(params: dict[str, Any], base: str, idx: int) -> str:
    filename = _path_stem(params.get("filename")) or base
    title = params.get("title")
    if isinstance(title, str) and (
        _USI_PATTERN.match(title) or title.startswith(f"{filename}:cluster:")
    ):
        return title
    for key in ("scans", "scan"):
        raw = params.get(key)
        if raw is not None and str(raw).strip():
            return f"{filename}:scan:{str(raw).strip()}"
    return f"{filename}:index:{idx}"


def read_mgf(source: SourceType) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra from an MGF file or stream."""
    import numpy as np
    import pyteomics.mgf
    from pyteomics.auxiliary import PyteomicsError

    try:
        with pyteomics.mgf.MGF(
            _normalize_source(source),
            read_charges=False,
            convert_arrays=1,
            dtype=np.float32,
        ) as f_in:
            base = _base_filename(source, f_in)
            for idx, spectrum_dict in enumerate(f_in):
                identifier = f"{base}:index:{idx}"
                try:
                    params = spectrum_dict["params"]
                    identifier = _resolve_mgf_id(params, base, idx)
                    pepmass = params["pepmass"]
                    if isinstance(pepmass, (str, bytes)):
                        raise TypeError("Expected pepmass sequence")
                    spec = _build_spectrum(
                        identifier,
                        pepmass[0],
                        _parse_charge(params.get("charge")),
                        spectrum_dict["m/z array"],
                        spectrum_dict["intensity array"],
                        _parse_rt(params.get("rtinseconds")),
                    )
                except PARSE_EXCEPTIONS as exc:
                    logger.debug(
                        "Skipping unparsable spectrum %s in %s: %s",
                        identifier,
                        source,
                        exc,
                    )
                else:
                    yield spec
    except (PyteomicsError, ValueError) as exc:
        logger.warning("Failed to read file %s: %s", source, exc)


def _scale_intensities(intensity: np.ndarray) -> np.ndarray:
    import numpy as np

    if intensity.size == 0:
        return intensity
    max_i = float(np.max(intensity))
    if not math.isfinite(max_i) or max_i <= 0.0:
        return intensity
    return (intensity.astype(np.float64, copy=False) / max_i) * 1000.0


def _format_mgf_spectrum(spectrum: ConsensusTuple, scan_nr: int) -> str:
    import numpy as np

    precursor_mz = float(spectrum.precursor_mz)
    if not math.isfinite(precursor_mz):
        raise ValueError(
            f"Representative precursor_mz must be finite, got {precursor_mz!r}"
        )
    if not (
        np.all(np.isfinite(spectrum.mz))
        and np.all(np.isfinite(spectrum.intensity))
    ):
        raise ValueError("Representative peak arrays must be finite")

    lines = [
        "BEGIN IONS",
        f"TITLE=falcon:cluster:{spectrum.cluster_id}",
        f"PEPMASS={precursor_mz}",
    ]
    rt = float(spectrum.retention_time)
    if math.isfinite(rt):
        lines.append(f"RTINSECONDS={rt}")
    charge = spectrum.precursor_charge
    if charge is not None and math.isfinite(float(charge)):
        charge_int = int(charge)
        sign = "+" if charge_int >= 0 else "-"
        lines.append(f"CHARGE={abs(charge_int)}{sign}")
    lines.append(f"SCANS={scan_nr}")

    scaled = _scale_intensities(spectrum.intensity)
    lines.extend(
        f"{mz:.5f} {intensity:.1f}"
        for mz, intensity in zip(
            spectrum.mz.tolist(), scaled.tolist(), strict=True
        )
    )
    lines.append("END IONS")
    return "\n".join(lines) + "\n\n"


# ----------------------------------------------------------------------
# mzML and mzXML readers
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_offline_psims_cv() -> Any:
    """Load the vendored PSI-MS CV without network access."""
    try:
        from psims.controlled_vocabulary.controlled_vocabulary import OBOCache

        return OBOCache(enabled=False, use_remote=False).load(_PSI_MS_OBO_URI)
    except Exception as exc:
        logger.debug("Could not preload offline PSI-MS CV: %s", exc)
        return None


def read_mzml(source: SourceType) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra (MS level > 1) from an mzML file or stream."""
    import numpy as np
    import pyteomics.mzml
    import pyteomics.xml

    def _open(xml_source: IO[Any] | str) -> Any:
        kwargs: dict[str, Any] = {"use_index": False, "dtype": np.float32}
        if hasattr(pyteomics.xml, "CVParamParser"):
            cv = _load_offline_psims_cv()
            if cv is not None:
                kwargs["cv"] = cv
        return pyteomics.mzml.MzML(xml_source, **kwargs)

    warned_missing_scan = False

    def _parse(
        spectrum_dict: dict[str, Any], filename: str
    ) -> sus.MsmsSpectrum | None:
        nonlocal warned_missing_scan
        raw_id = spectrum_dict.get("id")
        match = _SCAN_NR_RE.search(raw_id) if isinstance(raw_id, str) else None
        if match is None:
            log_fn = logger.debug if warned_missing_scan else logger.warning
            warned_missing_scan = True
            log_fn(
                "Skipping spectrum with unrecognized id "
                "(no 'scan=' field) in %s: %s",
                source,
                raw_id,
            )
            return None

        scans = (spectrum_dict.get("scanList") or {}).get("scan")
        rt = _parse_rt(scans[0].get("scan start time") if scans else None)
        ion = spectrum_dict["precursorList"]["precursor"][0][
            "selectedIonList"
        ]["selectedIon"][0]
        negative = "negative scan" in spectrum_dict
        charge = _parse_charge(ion.get("charge state"), negative=negative)
        if charge is None:
            charge = _parse_charge(
                ion.get("possible charge state"), negative=negative
            )
        return _build_spectrum(
            f"{filename}:scan:{match.group(1)}",
            ion["selected ion m/z"],
            charge,
            spectrum_dict["m/z array"],
            spectrum_dict["intensity array"],
            rt,
        )

    return _iter_xml(source, _open, "ms level", _parse)


def read_mzxml(source: SourceType) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra (MS level > 1) from an mzXML source."""
    import numpy as np
    import pyteomics.mzxml

    def _open(xml_source: IO[Any] | str) -> Any:
        return pyteomics.mzxml.MzXML(
            xml_source, use_index=False, dtype=np.float32
        )

    def _parse(
        spectrum_dict: dict[str, Any], filename: str
    ) -> sus.MsmsSpectrum:
        scan_nr = str(int(spectrum_dict["id"]))
        rt = _parse_rt(spectrum_dict.get("retentionTime"))
        precursors = spectrum_dict["precursorMz"]
        if isinstance(precursors, (str, bytes)):
            raise TypeError("Expected precursorMz sequence")
        precursor = precursors[0]
        if isinstance(precursor, Mapping):
            precursor_mz = precursor["precursorMz"]
            charge = _parse_charge(
                precursor.get("precursorCharge"),
                negative=spectrum_dict.get("polarity") == "-",
            )
        else:
            precursor_mz = precursor
            charge = None
        return _build_spectrum(
            f"{filename}:scan:{scan_nr}",
            precursor_mz,
            charge,
            spectrum_dict["m/z array"],
            spectrum_dict["intensity array"],
            rt,
        )

    return _iter_xml(source, _open, "msLevel", _parse)


# ----------------------------------------------------------------------
# Extension-based dispatch
# ----------------------------------------------------------------------


def get_spectra(filename: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """
    Read MS/MS spectra from an MGF, mzML, or mzXML peak file.

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
        return read_mgf(path)
    if ext == ".mzml":
        return read_mzml(path)
    if ext == ".mzxml":
        return read_mzxml(path)
    raise ValueError(f'Unknown spectrum file type with extension "{ext}"')


def write_spectra(
    filename: PathLike, spectra: Iterable[ConsensusTuple]
) -> None:
    """Write representative spectra atomically to an MGF file."""
    path = Path(filename)
    if path.suffix.lower() != ".mgf":
        raise ValueError(
            "Unsupported peak file format (supported formats: MGF)"
        )
    with atomic_write_text(path, prefix=".mgf_") as f_out:
        for scan_nr, spectrum in enumerate(spectra, start=1):
            f_out.write(_format_mgf_spectrum(spectrum, scan_nr))
