"""Shared test fixtures and toy dataset builders for Falcon tests."""

import base64
import struct
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import spectrum_utils.spectrum as sus

from falcon.cluster import similarity
from falcon.config import config as _global_config

SPECTRUM_SCHEMA = pa.schema(
    [
        pa.field("identifier", pa.string()),
        pa.field("precursor_mz", pa.float32()),
        pa.field("precursor_charge", pa.int8()),
        pa.field("mz", pa.list_(pa.float32())),
        pa.field("intensity", pa.list_(pa.float32())),
        pa.field("retention_time", pa.float32()),
    ]
)


@pytest.fixture
def reset_global_config():
    """Restore global ``falcon.config.config`` state after a test."""
    prev_namespace = _global_config._namespace
    yield _global_config
    _global_config._namespace = prev_namespace
    _global_config.__dict__.pop("work_dir", None)


@pytest.fixture
def simple_spectrum():
    """A valid MsmsSpectrum for testing preprocessing."""
    return sus.MsmsSpectrum(
        "test:scan:1",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array(
            [110.0, 200.0, 300.0, 400.0, 450.0, 480.0, 490.0],
            dtype=np.float32,
        ),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 10.0],
            dtype=np.float32,
        ),
        retention_time=120.0,
    )


@pytest.fixture
def narrow_spectrum():
    """Spectrum with a narrow m/z range (rejected by min_mz_range)."""
    return sus.MsmsSpectrum(
        "test:scan:2",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array([200.0, 200.5, 201.0, 201.5, 202.0], dtype=np.float32),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0], dtype=np.float32
        ),
        retention_time=130.0,
    )


@pytest.fixture
def few_peaks_spectrum():
    """Spectrum with too few peaks (rejected by min_peaks)."""
    return sus.MsmsSpectrum(
        "test:scan:3",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array([200.0, 400.0], dtype=np.float32),
        intensity=np.array([100.0, 200.0], dtype=np.float32),
        retention_time=140.0,
    )


@pytest.fixture
def no_charge_spectrum():
    """A spectrum without a charge state."""
    return sus.MsmsSpectrum(
        "test:scan:4",
        precursor_mz=500.0,
        precursor_charge=None,
        mz=np.array(
            [110.0, 200.0, 300.0, 400.0, 450.0, 480.0, 490.0],
            dtype=np.float32,
        ),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 10.0],
            dtype=np.float32,
        ),
        retention_time=150.0,
    )


@pytest.fixture
def spectrum_tuple_pair():
    """A pair of identical L2-normalized SpectrumTuples."""
    int1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    int1 = int1 / np.linalg.norm(int1)
    int2 = int1.copy()
    mz = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    spec1 = similarity.SpectrumTuple(
        precursor_mz=100.0, precursor_charge=2, mz=mz, intensity=int1
    )
    spec2 = similarity.SpectrumTuple(
        precursor_mz=100.0, precursor_charge=2, mz=mz.copy(), intensity=int2
    )
    return spec1, spec2


@pytest.fixture
def orthogonal_spectrum_tuple_pair():
    """Pair of SpectrumTuples with disjoint peaks (distance 1.0)."""
    int1 = np.array([1.0, 0.0], dtype=np.float32)
    int2 = np.array([1.0, 0.0], dtype=np.float32)
    spec1 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([100.0, 101.0], dtype=np.float32),
        intensity=int1,
    )
    spec2 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([200.0, 201.0], dtype=np.float32),
        intensity=int2,
    )
    return spec1, spec2


@pytest.fixture
def mock_spectra():
    """Factory fixture creating identical SpectrumTuples."""

    def _make(n, mz_base=100.0, mz_step=1.0, n_peaks=3, charge=2):
        spectra = []
        for _ in range(n):
            mz = np.array(
                [mz_base + j * mz_step for j in range(n_peaks)],
                dtype=np.float32,
            )
            intensity = np.array(
                [0.5 + 0.1 * j for j in range(n_peaks)], dtype=np.float32
            )
            intensity = intensity / np.linalg.norm(intensity)
            spectra.append(
                similarity.SpectrumTuple(
                    precursor_mz=np.float32(mz_base),
                    precursor_charge=charge,
                    mz=mz,
                    intensity=intensity,
                )
            )
        return spectra

    return _make


def make_spectrum_row(
    identifier,
    precursor_mz,
    mz,
    intensity,
    precursor_charge=2,
    retention_time=0.0,
):
    """Build a single Lance row dict for a spectrum (L2-normalized)."""
    mz = np.asarray(mz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    norm = np.linalg.norm(intensity)
    if norm > 0:
        intensity = intensity / norm
    return {
        "identifier": identifier,
        "precursor_mz": np.float32(precursor_mz),
        "precursor_charge": precursor_charge,
        "mz": mz,
        "intensity": intensity,
        "retention_time": np.float32(retention_time),
    }


@pytest.fixture
def spectrum_schema():
    """Return the PyArrow schema for staged Lance spectrum tables."""
    return SPECTRUM_SCHEMA


@pytest.fixture
def spectrum_row():
    """Expose ``make_spectrum_row`` as a fixture."""
    return make_spectrum_row


@pytest.fixture
def mgf_writer():
    """Return ``(format_mgf_spectrum, write_mgf)`` helpers."""
    return format_mgf_spectrum, write_mgf


@pytest.fixture
def mzml_writer():
    """Expose ``write_toy_mzml`` as a fixture."""
    return write_toy_mzml


@pytest.fixture
def mzxml_writer():
    """Expose ``write_toy_mzxml`` as a fixture."""
    return write_toy_mzxml


@pytest.fixture
def lance_dataset(tmp_path):
    """Factory fixture writing rows to a Lance dataset.

    The dataset path mirrors the production layout
    (``spectra_charge_<charge>.lance``) so clustering derives the
    charge label from the URI.
    """

    def _make(rows, charge=2, name=None):
        filename = name or f"spectra_charge_{charge}.lance"
        path = str(tmp_path / filename)
        return lance.write_dataset(
            pa.Table.from_pylist(rows, SPECTRUM_SCHEMA),
            path,
            mode="overwrite",
            data_storage_version="stable",
        )

    return _make


def format_mgf_spectrum(
    peaks,
    precursor_mz,
    charge=2,
    scan=None,
    rt=None,
    title=None,
    extra_params=None,
):
    """Format a single MGF spectrum block as text."""
    lines = ["BEGIN IONS"]
    if title is not None:
        lines.append(f"TITLE={title}")
    if scan is not None:
        lines.append(f"SCANS={scan}")
    lines.append(f"PEPMASS={precursor_mz}")
    if charge is not None:
        sign = "+" if charge >= 0 else "-"
        lines.append(f"CHARGE={abs(charge)}{sign}")
    if rt is not None:
        lines.append(f"RTINSECONDS={rt}")
    if extra_params:
        for key, value in extra_params.items():
            lines.append(f"{key}={value}")
    for mz, intensity in peaks:
        lines.append(f"{mz:.4f} {intensity:.4f}")
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


def write_mgf(path: Path, entries: list[str]) -> Path:
    """Write formatted MGF spectrum blocks to ``path``."""
    path.write_text("".join(entries))
    return path


def write_toy_mzml(path: Path, spectra_specs: list[dict]) -> Path:
    """Write a minimal valid mzML file from spectrum descriptors.

    Each descriptor dict may include ``scan``, ``ms_level``,
    ``precursor_mz``, ``charge``, ``rt``, ``mz``, ``intensity``, and
    optional ``id_str``.
    """
    blocks = []
    for idx, spec in enumerate(spectra_specs):
        scan_nr = spec.get("scan", idx + 1)
        id_str = spec.get(
            "id_str",
            f"controllerType=0 controllerNumber=1 scan={scan_nr}",
        )
        ms_level = spec.get("ms_level", 2)
        mzs = [float(v) for v in spec["mz"]]
        intensities = [float(v) for v in spec["intensity"]]
        mz_b64 = base64.b64encode(struct.pack(f"<{len(mzs)}d", *mzs)).decode(
            "ascii"
        )
        int_b64 = base64.b64encode(
            struct.pack(f"<{len(intensities)}d", *intensities)
        ).decode("ascii")

        scan_xml = ""
        if "rt" in spec and spec["rt"] is not None:
            scan_xml = (
                '    <scanList count="1"><scan>'
                '<cvParam cvRef="MS" accession="MS:1000016" '
                f'name="scan start time" value="{spec["rt"]}" '
                'unitName="second"/>'
                "</scan></scanList>\n"
            )
        else:
            scan_xml = '    <scanList count="1"><scan/></scanList>\n'

        prec_xml = ""
        if (
            ms_level is not None
            and ms_level > 1
            and spec.get("precursor_mz") is not None
        ):
            charge_xml = ""
            if spec.get("charge") is not None:
                charge_xml = (
                    '<cvParam cvRef="MS" accession="MS:1000041" '
                    f'name="charge state" value="{spec["charge"]}"/>'
                )
            prec_xml = (
                '    <precursorList count="1"><precursor>'
                '<selectedIonList count="1"><selectedIon>'
                '<cvParam cvRef="MS" accession="MS:1000744" '
                f'name="selected ion m/z" value="{spec["precursor_mz"]}"/>'
                f"{charge_xml}"
                "</selectedIon></selectedIonList>"
                "</precursor></precursorList>\n"
            )

        ms_level_xml = ""
        if ms_level is not None:
            ms_level_xml = (
                '    <cvParam cvRef="MS" accession="MS:1000511" '
                f'name="ms level" value="{ms_level}"/>\n'
            )

        blocks.append(
            f'   <spectrum index="{idx}" id="{id_str}" '
            f'defaultArrayLength="{len(mzs)}">\n'
            f"{ms_level_xml}"
            f"{scan_xml}"
            f"{prec_xml}"
            '    <binaryDataArrayList count="2">\n'
            f'     <binaryDataArray encodedLength="{len(mz_b64)}">\n'
            '      <cvParam cvRef="MS" accession="MS:1000523" '
            'name="64-bit float"/>\n'
            '      <cvParam cvRef="MS" accession="MS:1000514" '
            'name="m/z array"/>\n'
            f"      <binary>{mz_b64}</binary>\n"
            "     </binaryDataArray>\n"
            f'     <binaryDataArray encodedLength="{len(int_b64)}">\n'
            '      <cvParam cvRef="MS" accession="MS:1000523" '
            'name="64-bit float"/>\n'
            '      <cvParam cvRef="MS" accession="MS:1000515" '
            'name="intensity array"/>\n'
            f"      <binary>{int_b64}</binary>\n"
            "     </binaryDataArray>\n"
            "    </binaryDataArrayList>\n"
            "   </spectrum>\n"
        )

    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">\n'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
        ' <run id="run1">\n'
        f'  <spectrumList count="{len(blocks)}">\n'
        f"{''.join(blocks)}"
        "  </spectrumList>\n"
        " </run>\n"
        "</mzML>\n"
        "</indexedmzML>\n"
    )
    path.write_text(xml)
    return path


def write_toy_mzxml(path: Path, spectra_specs: list[dict]) -> Path:
    """Write a minimal valid mzXML file from spectrum descriptors."""
    scans = []
    for idx, spec in enumerate(spectra_specs):
        scan_num = spec.get("scan", idx + 1)
        ms_level = spec.get("ms_level", 2)
        mzs = [float(v) for v in spec["mz"]]
        intensities = [float(v) for v in spec["intensity"]]
        interleaved = []
        for mz_val, int_val in zip(mzs, intensities, strict=True):
            interleaved.extend([mz_val, int_val])
        peaks_b64 = base64.b64encode(
            struct.pack(f">{len(interleaved)}f", *interleaved)
        ).decode("ascii")

        attrs = [f'num="{scan_num}"']
        if ms_level is not None:
            attrs.append(f'msLevel="{ms_level}"')
        attrs.append(f'peaksCount="{len(mzs)}"')
        if "rt" in spec and spec["rt"] is not None:
            attrs.append(f'retentionTime="PT{spec["rt"]}S"')

        prec_xml = ""
        if spec.get("precursor_mz") is not None:
            charge_attr = (
                f' precursorCharge="{spec["charge"]}"'
                if spec.get("charge") is not None
                else ""
            )
            prec_xml = (
                f"   <precursorMz{charge_attr}>"
                f"{spec['precursor_mz']}</precursorMz>\n"
            )

        scans.append(
            f"  <scan {' '.join(attrs)}>\n"
            f"{prec_xml}"
            '   <peaks compressionType="none" precision="32" '
            f'byteOrder="network" contentType="m/z-int">{peaks_b64}</peaks>\n'
            "  </scan>\n"
        )

    xml = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        '<mzXML xmlns="http://sashimi.sourceforge.net/'
        'schema_revision/mzXML_3.2">\n'
        f' <msRun scanCount="{len(scans)}">\n'
        f"{''.join(scans)}"
        " </msRun>\n"
        "</mzXML>\n"
    )
    path.write_text(xml)
    return path
