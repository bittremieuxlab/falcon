"""Synthetic data builders and constants shared across tests."""

from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Any

from falcon.config import ClusteringConfig, PrecursorTolerance, load_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from falcon.cluster.consensus import ConsensusTuple, SpectrumTuple
    from falcon.config import FalconConfig

TOY_PEAKS_5: list[tuple[float, float]] = [
    (150.0, 100.0),
    (250.0, 200.0),
    (350.0, 150.0),
    (450.0, 300.0),
    (550.0, 250.0),
]
LOW_QUALITY_PEAKS: list[tuple[float, float]] = [
    (150.0, 100.0),
    (160.0, 200.0),
]


def make_spectrum_tuple(
    mz: Any,
    intensity: Any,
    precursor_mz: float = 500.0,
    charge: Any = 2,
    normalize: bool = True,
) -> SpectrumTuple:
    """Build a ``SpectrumTuple`` with float32 arrays and optional L2 norm."""
    import numpy as np

    from falcon.cluster.consensus import SpectrumTuple

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
    precursor_mz: float = 500.0,
    charge: Any = 2,
    mz: Any = (100.0, 200.0, 300.0),
    intensity: Any = (0.5, 1.0, 0.25),
    rt: float | None = 120.0,
    cluster_id: int = 0,
) -> ConsensusTuple:
    """Build a toy ``ConsensusTuple`` via ``make_consensus_tuple``."""
    import numpy as np

    from falcon.cluster.consensus import make_consensus_tuple

    return make_consensus_tuple(
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        retention_time=float("nan") if rt is None else rt,
        cluster_id=cluster_id,
    )


def make_falcon_config(
    input_filenames: Sequence[str] = ("in.mgf",),
    output_filename: str | Path = "out",
    **sections: Any,
) -> FalconConfig:
    """Build a validated ``FalconConfig`` with optional section overrides."""
    io_section = {
        "input_filenames": list(input_filenames),
        "output_filename": str(output_filename),
        **sections.pop("io", {}),
    }
    return load_config(overrides={"io": io_section, **sections})


def make_da_clustering(**overrides: Any) -> ClusteringConfig:
    """Build a ``ClusteringConfig`` with 0.5 Da precursor tolerance."""
    defaults: dict[str, Any] = {
        "precursor_tol": PrecursorTolerance(mass=0.5, mode="Da"),
        "fragment_tol": 0.05,
        "distance_threshold": 0.1,
    }
    defaults.update(overrides)
    return ClusteringConfig.model_validate(defaults)


def make_spectrum_row(
    identifier: str,
    precursor_mz: float = 500.0,
    mz: Any = (100.0, 200.0, 300.0),
    intensity: Any = (1.0, 0.8, 0.6),
    precursor_charge: int | None = 2,
    retention_time: float = 0.0,
) -> dict[str, Any]:
    """Build a single Lance row dict for a spectrum (L2-normalized)."""
    import numpy as np

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
    peaks: list[tuple[float, float]],
    precursor_mz: float,
    charge: int | None = 2,
    scan: int | str | None = None,
    rt: float | None = None,
    title: str | None = None,
) -> str:
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


def write_toy_mzml(path: Path, spectra_specs: list[dict[str, Any]]) -> Path:
    """Write a minimal valid mzML file from spectrum descriptors."""
    blocks = []
    for idx, spec in enumerate(spectra_specs):
        scan_nr = spec.get("scan", idx + 1)
        id_str = spec.get(
            "id_str", f"controllerType=0 controllerNumber=1 scan={scan_nr}"
        )
        ms_level = spec.get("ms_level", 2)
        mzs = [float(v) for v in spec.get("mz", ())]
        intensities = [float(v) for v in spec.get("intensity", ())]
        mz_b64 = base64.b64encode(struct.pack(f"<{len(mzs)}d", *mzs)).decode(
            "ascii"
        )
        int_b64 = base64.b64encode(
            struct.pack(f"<{len(intensities)}d", *intensities)
        ).decode("ascii")

        if spec.get("omit_scan_list"):
            scan_xml = ""
        elif spec.get("rt") is not None:
            scan_xml = (
                '    <scanList count="1"><scan>'
                '<cvParam cvRef="MS" accession="MS:1000016" '
                f'name="scan start time" value="{spec["rt"]}" '
                'unitName="second"/></scan></scanList>\n'
            )
        else:
            scan_xml = '    <scanList count="1"><scan/></scanList>\n'

        prec_xml = ""
        if ms_level > 1 and spec.get("precursor_mz") is not None:
            charge_parts = []
            if spec.get("charge") is not None:
                charge_parts.append(
                    '<cvParam cvRef="MS" accession="MS:1000041" '
                    f'name="charge state" value="{spec["charge"]}"/>'
                )
            for possible in spec.get("possible_charges", ()):
                charge_parts.append(
                    '<cvParam cvRef="MS" accession="MS:1000633" '
                    f'name="possible charge state" value="{possible}"/>'
                )
            prec_xml = (
                '    <precursorList count="1"><precursor>'
                '<selectedIonList count="1"><selectedIon>'
                '<cvParam cvRef="MS" accession="MS:1000744" '
                f'name="selected ion m/z" value="{spec["precursor_mz"]}"/>'
                f"{''.join(charge_parts)}"
                "</selectedIon></selectedIonList>"
                "</precursor></precursorList>\n"
            )

        cv_parts = [
            '    <cvParam cvRef="MS" accession="MS:1000511" '
            f'name="ms level" value="{ms_level}"/>\n'
        ]
        if spec.get("negative_scan"):
            cv_parts.append(
                '    <cvParam cvRef="MS" accession="MS:1000129" '
                'name="negative scan"/>\n'
            )

        blocks.append(
            f'   <spectrum index="{idx}" id="{id_str}" '
            f'defaultArrayLength="{len(mzs)}">\n'
            f"{''.join(cv_parts)}{scan_xml}{prec_xml}"
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


def write_toy_mzxml(path: Path, spectra_specs: list[dict[str, Any]]) -> Path:
    """Write a minimal valid mzXML file from spectrum descriptors."""
    scans = []
    for idx, spec in enumerate(spectra_specs):
        scan_num = spec.get("scan", idx + 1)
        ms_level = spec.get("ms_level", 2)
        mzs = [float(v) for v in spec.get("mz", ())]
        intensities = [float(v) for v in spec.get("intensity", ())]
        interleaved = [
            v for pair in zip(mzs, intensities, strict=True) for v in pair
        ]
        peaks_b64 = base64.b64encode(
            struct.pack(f">{len(interleaved)}f", *interleaved)
        ).decode("ascii")

        attrs = [
            f'num="{scan_num}"',
            f'msLevel="{ms_level}"',
            f'peaksCount="{len(mzs)}"',
        ]
        if spec.get("polarity") is not None:
            attrs.append(f'polarity="{spec["polarity"]}"')
        if spec.get("rt") is not None:
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
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mzXML xmlns="http://sashimi.sourceforge.net/'
        'schema_revision/mzXML_3.2">\n'
        f' <msRun scanCount="{len(scans)}">\n'
        f"{''.join(scans)}"
        " </msRun>\n"
        "</mzXML>\n"
    )
    path.write_text(xml, encoding="utf-8")
    return path
