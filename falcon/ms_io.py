"""Peak file readers (MGF, mzML, mzXML), MGF writer, and atomic I/O."""

from __future__ import annotations

import binascii
import contextlib
import functools
import logging
import math
import os
import re
import tempfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    import numpy as np
    import spectrum_utils.spectrum as sus

    from .cluster.consensus import ConsensusTuple
    from .config import PathLike

logger = logging.getLogger(__name__)

_PARSE_EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError)
_USI_PATTERN = re.compile(
    r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$"
)
_SCAN_NR_RE = re.compile(r"(?:^|\s)scan=(\S+)")
_PSI_MS_OBO_URI = "http://purl.obolibrary.org/obo/ms/psi-ms.obo"


@contextlib.contextmanager
def atomic_write_text(
    path: Path,
    *,
    prefix: str = ".falcon_",
    encoding: str = "utf-8",
) -> Generator[IO[str], None, None]:
    """Atomically write text to ``path`` via a sibling temp file."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=prefix, suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f_out:
            yield f_out
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _parse_rt(raw_rt: Any) -> float:
    if isinstance(raw_rt, (tuple, list)):
        if len(raw_rt) == 2:
            return (float(raw_rt[0]) + float(raw_rt[1])) / 2.0
        if len(raw_rt) == 1:
            raw_rt = raw_rt[0]
        elif not raw_rt:
            return float("nan")
        else:
            raise ValueError(
                f"Expected 1 or 2 retention time values, got {len(raw_rt)}"
            )
    if raw_rt is None or (isinstance(raw_rt, str) and not raw_rt.strip()):
        return float("nan")
    return float(raw_rt)


def _parse_charge(raw_charge: Any, *, negative: bool = False) -> int | None:
    if isinstance(raw_charge, (list, tuple)):
        if not raw_charge:
            return None
        raw_charge = raw_charge[0]
    if raw_charge is None or (
        isinstance(raw_charge, str) and not raw_charge.strip()
    ):
        return None
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


def _resolve_mgf_id(params: dict[str, Any], base: str, idx: int) -> str:
    raw_fn = params.get("filename")
    filename = (
        Path(raw_fn).stem
        if isinstance(raw_fn, str) and raw_fn.strip()
        else base
    )
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


def read_mgf(source: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra from an MGF file."""
    import numpy as np
    import pyteomics.mgf
    from pyteomics.auxiliary import PyteomicsError

    path = Path(source)
    base = path.stem
    try:
        with pyteomics.mgf.MGF(
            str(path), read_charges=False, convert_arrays=1, dtype=np.float32
        ) as f_in:
            for idx, spectrum_dict in enumerate(f_in):
                identifier = f"{base}:index:{idx}"
                try:
                    params = spectrum_dict["params"]
                    identifier = _resolve_mgf_id(params, base, idx)
                    pepmass = params["pepmass"]
                    spec = _build_spectrum(
                        identifier,
                        pepmass[0],
                        _parse_charge(params.get("charge")),
                        spectrum_dict["m/z array"],
                        spectrum_dict["intensity array"],
                        _parse_rt(params.get("rtinseconds")),
                    )
                except _PARSE_EXCEPTIONS as exc:
                    logger.debug(
                        "Skipping unparsable spectrum %s in %s: %s",
                        identifier,
                        path,
                        exc,
                    )
                else:
                    yield spec
    except (PyteomicsError, ValueError) as exc:
        logger.warning("Failed to read file %s: %s", path, exc)


@functools.lru_cache(maxsize=1)
def _load_offline_psims_cv() -> Any:
    """Load the vendored PSI-MS CV without network access."""
    try:
        from psims.controlled_vocabulary.controlled_vocabulary import OBOCache

        return OBOCache(enabled=False, use_remote=False).load(_PSI_MS_OBO_URI)
    except Exception as exc:
        logger.debug("Could not preload offline PSI-MS CV: %s", exc)
        return None


def read_mzml(source: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra (MS level > 1) from an mzML file."""
    import numpy as np
    import pyteomics.mzml
    import pyteomics.xml
    from lxml.etree import LxmlError
    from pyteomics.auxiliary import PyteomicsError

    path = Path(source)
    base = path.stem
    kwargs: dict[str, Any] = {"use_index": False, "dtype": np.float32}
    if hasattr(pyteomics.xml, "CVParamParser"):
        cv = _load_offline_psims_cv()
        if cv is not None:
            kwargs["cv"] = cv

    try:
        with pyteomics.mzml.MzML(str(path), **kwargs) as f_in:
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get("ms level", -1)) <= 1:
                        continue
                    raw_id = spectrum_dict.get("id")
                    match = (
                        _SCAN_NR_RE.search(raw_id)
                        if isinstance(raw_id, str)
                        else None
                    )
                    if match is None:
                        logger.warning(
                            "Skipping spectrum with unrecognized id "
                            "(no 'scan=' field) in %s: %s",
                            path,
                            raw_id,
                        )
                        continue
                    scans = (spectrum_dict.get("scanList") or {}).get("scan")
                    rt = _parse_rt(
                        scans[0].get("scan start time") if scans else None
                    )
                    ion = spectrum_dict["precursorList"]["precursor"][0][
                        "selectedIonList"
                    ]["selectedIon"][0]
                    negative = "negative scan" in spectrum_dict
                    charge = _parse_charge(
                        ion.get("charge state"), negative=negative
                    )
                    if charge is None:
                        charge = _parse_charge(
                            ion.get("possible charge state"), negative=negative
                        )
                    spec = _build_spectrum(
                        f"{base}:scan:{match.group(1)}",
                        ion["selected ion m/z"],
                        charge,
                        spectrum_dict["m/z array"],
                        spectrum_dict["intensity array"],
                        rt,
                    )
                except _PARSE_EXCEPTIONS as exc:
                    logger.debug(
                        "Skipping unparsable spectrum %s in %s: %s",
                        spectrum_dict.get("id"),
                        path,
                        exc,
                    )
                else:
                    yield spec
    except (
        LxmlError,
        PyteomicsError,
        KeyError,
        ValueError,
        binascii.Error,
        zlib.error,
    ) as exc:
        logger.warning("Failed to read file %s: %s", path, exc)


def read_mzxml(source: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra (MS level > 1) from an mzXML file."""
    import numpy as np
    import pyteomics.mzxml
    from lxml.etree import LxmlError
    from pyteomics.auxiliary import PyteomicsError

    path = Path(source)
    base = path.stem
    try:
        with pyteomics.mzxml.MzXML(
            str(path), use_index=False, dtype=np.float32
        ) as f_in:
            for spectrum_dict in f_in:
                try:
                    if int(spectrum_dict.get("msLevel", -1)) <= 1:
                        continue
                    scan_nr = str(int(spectrum_dict["id"]))
                    rt = _parse_rt(spectrum_dict.get("retentionTime"))
                    precursor = spectrum_dict["precursorMz"][0]
                    if isinstance(precursor, Mapping):
                        precursor_mz = precursor["precursorMz"]
                        charge = _parse_charge(
                            precursor.get("precursorCharge"),
                            negative=spectrum_dict.get("polarity") == "-",
                        )
                    else:
                        precursor_mz = precursor
                        charge = None
                    spec = _build_spectrum(
                        f"{base}:scan:{scan_nr}",
                        precursor_mz,
                        charge,
                        spectrum_dict["m/z array"],
                        spectrum_dict["intensity array"],
                        rt,
                    )
                except _PARSE_EXCEPTIONS as exc:
                    logger.debug(
                        "Skipping unparsable spectrum %s in %s: %s",
                        spectrum_dict.get("id"),
                        path,
                        exc,
                    )
                else:
                    yield spec
    except (
        LxmlError,
        PyteomicsError,
        KeyError,
        ValueError,
        binascii.Error,
        zlib.error,
    ) as exc:
        logger.warning("Failed to read file %s: %s", path, exc)


def get_spectra(filename: PathLike) -> Iterator[sus.MsmsSpectrum]:
    """Read MS/MS spectra from an MGF, mzML, or mzXML peak file."""
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


def _format_mgf_spectrum(spectrum: ConsensusTuple, scan_nr: int) -> str:
    import numpy as np

    precursor_mz = float(spectrum.precursor_mz)
    if not (
        math.isfinite(precursor_mz)
        and np.all(np.isfinite(spectrum.mz))
        and np.all(np.isfinite(spectrum.intensity))
    ):
        raise ValueError("Representative spectrum values must be finite")

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

    intensity = spectrum.intensity
    max_i = float(np.max(intensity)) if intensity.size else 0.0
    scaled = (
        (intensity.astype(np.float64, copy=False) / max_i) * 1000.0
        if max_i > 0.0
        else intensity
    )
    lines.extend(
        f"{mz:.5f} {val:.1f}"
        for mz, val in zip(spectrum.mz.tolist(), scaled.tolist(), strict=True)
    )
    lines.append("END IONS")
    return "\n".join(lines) + "\n\n"


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
