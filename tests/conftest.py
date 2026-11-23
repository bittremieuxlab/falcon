"""Shared test fixtures and toy dataset builders for Falcon tests."""

import base64
import struct
import tempfile
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from falcon.config import ClusteringConfig, PrecursorTolerance, load_config
from falcon.dataset import SPECTRUM_SCHEMA
from falcon.types import SpectrumTuple, make_consensus_tuple

DATA_DIR = Path(__file__).resolve().parent / "data"

TOY_PEAKS_5 = [
    (150.0, 100.0),
    (250.0, 200.0),
    (350.0, 150.0),
    (450.0, 300.0),
    (550.0, 250.0),
]
LOW_QUALITY_PEAKS = [(150.0, 100.0), (160.0, 200.0)]


def make_spectrum_tuple(
    mz,
    intensity,
    precursor_mz=500.0,
    charge=2,
    normalize=True,
):
    """Build a ``SpectrumTuple`` with float32 arrays and optional L2 norm."""
    mz_arr = np.asarray(mz, dtype=np.float32)
    int_arr = np.asarray(intensity, dtype=np.float32)
    if normalize and int_arr.size > 0:
        norm = np.linalg.norm(int_arr)
        if norm > 0:
            int_arr = int_arr / norm
    return SpectrumTuple(
        precursor_mz=np.float32(precursor_mz),
        precursor_charge=charge,
        mz=mz_arr,
        intensity=int_arr,
    )


def make_consensus_record(
    precursor_mz=500.0,
    charge=2,
    mz=(100.0, 200.0, 300.0),
    intensity=(0.5, 1.0, 0.25),
    rt=120.0,
    cluster_id=0,
):
    """Build a toy ``ConsensusTuple`` via ``make_consensus_tuple``."""
    return make_consensus_tuple(
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        retention_time=float("nan") if rt is None else rt,
        cluster_id=cluster_id,
    )


def make_falcon_config(
    input_filenames=("in.mgf",),
    output_filename="out",
    **sections,
):
    """Build a validated ``FalconConfig`` with optional section overrides."""
    io_section = {
        "input_filenames": list(input_filenames),
        "output_filename": str(output_filename),
        **sections.pop("io", {}),
    }
    return load_config(overrides={"io": io_section, **sections})


def make_da_clustering(**overrides):
    """Build a ``ClusteringConfig`` with 0.5 Da precursor tolerance."""
    defaults = {
        "precursor_tol": PrecursorTolerance(mass=0.5, mode="Da"),
        "fragment_tol": 0.05,
        "distance_threshold": 0.1,
    }
    defaults.update(overrides)
    return ClusteringConfig(**defaults)


def make_spectrum_row(
    identifier,
    precursor_mz,
    mz,
    intensity,
    precursor_charge=2,
    retention_time=0.0,
):
    """Build a single Lance row dict for a spectrum (L2-normalized)."""
    spec = make_spectrum_tuple(
        mz, intensity, precursor_mz=precursor_mz, charge=precursor_charge
    )
    return {
        "identifier": identifier,
        "precursor_mz": spec.precursor_mz,
        "precursor_charge": spec.precursor_charge,
        "mz": spec.mz,
        "intensity": spec.intensity,
        "retention_time": np.float32(retention_time),
    }


def format_mgf_spectrum(
    peaks,
    precursor_mz,
    charge=2,
    scan=None,
    rt=None,
    title=None,
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
    for mz, intensity in peaks:
        lines.append(f"{mz:.4f} {intensity:.4f}")
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


def write_mgf(path: Path, entries: list[str]) -> Path:
    """Write formatted MGF spectrum blocks to ``path``."""
    path.write_text("".join(entries), encoding="utf-8")
    return path


def write_toy_mzml(path: Path, spectra_specs: list[dict]) -> Path:
    """Write a minimal valid mzML file from spectrum descriptors."""
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
    path.write_text(xml, encoding="utf-8")
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
    path.write_text(xml, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def data_dir():
    """Return path to checked-in peak-file fixtures."""
    return DATA_DIR


@pytest.fixture(scope="session")
def spec_factory():
    """Expose ``make_spectrum_tuple`` as a session fixture."""
    return make_spectrum_tuple


@pytest.fixture(scope="session")
def consensus_factory():
    """Expose ``make_consensus_record`` as a session fixture."""
    return make_consensus_record


@pytest.fixture(scope="session")
def config_factory():
    """Expose ``make_falcon_config`` as a session fixture."""
    return make_falcon_config


@pytest.fixture(scope="session")
def da_clustering_factory():
    """Expose ``make_da_clustering`` as a session fixture."""
    return make_da_clustering


@pytest.fixture(scope="session")
def spectrum_row():
    """Expose ``make_spectrum_row`` as a session fixture."""
    return make_spectrum_row


@pytest.fixture(scope="session")
def mock_spectra():
    """Factory fixture creating identical L2-normalized SpectrumTuples."""

    def _make(n, n_peaks=3):
        mz = [100.0 + float(j) for j in range(n_peaks)]
        intensity = [0.5 + 0.1 * j for j in range(n_peaks)]
        return [
            make_spectrum_tuple(mz, intensity, precursor_mz=100.0, charge=2)
            for _ in range(n)
        ]

    return _make


@pytest.fixture(scope="session")
def mgf_block():
    """Expose ``format_mgf_spectrum`` as a session fixture."""
    return format_mgf_spectrum


@pytest.fixture(scope="session")
def mgf_writer():
    """Expose ``write_mgf`` as a session fixture."""
    return write_mgf


@pytest.fixture(scope="session")
def mzml_writer():
    """Expose ``write_toy_mzml`` as a session fixture."""
    return write_toy_mzml


@pytest.fixture(scope="session")
def mzxml_writer():
    """Expose ``write_toy_mzxml`` as a session fixture."""
    return write_toy_mzxml


@pytest.fixture
def lance_dataset(tmp_path):
    """Factory fixture writing rows to a Lance dataset."""

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


@pytest.fixture
def track_mkdtemp(monkeypatch):
    """Track directories allocated via ``tempfile.mkdtemp``."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)
    return created
