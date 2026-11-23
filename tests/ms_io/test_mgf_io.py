"""Unit tests for falcon.ms_io.mgf_io."""

import io
import math

import numpy as np
import pytest

from falcon.ms_io import mgf_io
from falcon.types import ConsensusTuple

_DEFAULT_CHARGE = np.int32(2)
_DEFAULT_RT = np.float32(120.0)


def _make_consensus(
    precursor_mz=500.0,
    charge=_DEFAULT_CHARGE,
    mz=(100.0, 200.0, 300.0),
    intensity=(0.5, 1.0, 0.25),
    rt=_DEFAULT_RT,
    cluster_id=3,
):
    return ConsensusTuple(
        precursor_mz=np.float32(precursor_mz),
        precursor_charge=charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        retention_time=rt,
        cluster_id=np.int32(cluster_id),
    )


# ----------------------------------------------------------------------
# Reading MGF spectra and identifier construction
# ----------------------------------------------------------------------


class TestMgfRead:
    def test_parses_peaks_pepmass_charge_and_rt(self, tmp_path):
        mgf = tmp_path / "basic.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.25 99999.0\n"
            "CHARGE=2+ and 3+\n"
            "RTINSECONDS=123.5\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "basic:scan:1"
        assert s.precursor_mz == pytest.approx(500.25)
        # First charge is chosen when multiple charges are listed.
        assert s.precursor_charge == 2
        assert s.retention_time == pytest.approx(123.5)
        np.testing.assert_allclose(s.mz, [100.0, 200.0])
        np.testing.assert_allclose(s.intensity, [50.0, 100.0])

    def test_dotted_filename_preserves_full_stem(self, tmp_path):
        mgf = tmp_path / "sample.v1.mgf"
        mgf.write_text(
            "BEGIN IONS\nSCANS=7\nPEPMASS=500.0\n100.0 50\nEND IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == ["sample.v1:scan:7"]

    def test_missing_charge_and_rt(self, tmp_path):
        mgf = tmp_path / "minimal.mgf"
        mgf.write_text(
            "BEGIN IONS\nPEPMASS=400.0\n100.0 50\n200.0 100\nEND IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge is None
        assert math.isnan(spectra[0].retention_time)

    def test_negative_charge(self, tmp_path):
        mgf = tmp_path / "neg.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=3-\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge == -3

    @pytest.mark.parametrize(
        "usi",
        [
            "mzspec:PXD000000:run:scan:9",
            "mzspec:MSV000001:file:42",
            "mzspec:PXD000000:run:scan:9:PEPTIDE/2",
        ],
    )
    def test_usi_titles_preserved(self, tmp_path, usi):
        mgf = tmp_path / "run.mgf"
        mgf.write_text(
            f"BEGIN IONS\n"
            f"TITLE={usi}\n"
            f"SCANS=99\n"
            f"PEPMASS=500.0\n"
            f"100.0 50\n"
            f"END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].identifier == usi

    def test_cluster_title_preserved_only_for_matching_stem(self, tmp_path):
        mgf = tmp_path / "myrun.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=myrun:cluster:7\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "TITLE=otherfile:cluster:8\n"
            "SCANS=2\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == [
            "myrun:cluster:7",
            "myrun:scan:2",
        ]

    def test_identifier_fallback_precedence_scans_scan_index(self, tmp_path):
        mgf = tmp_path / "prec.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=plain label\n"
            "SCANS=10\n"
            "SCAN=20\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCAN=77\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "PEPMASS=600.0\n"
            "100.0 50\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == [
            "prec:scan:10",
            "prec:scan:77",
            "prec:index:2",
        ]

    def test_filename_param_overrides_disk_stem(self, tmp_path):
        mgf = tmp_path / "ondisk.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "FILENAME=/nested/dir/original.raw\n"
            "SCANS=5\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert spectra[0].identifier == "original:scan:5"

    def test_stream_without_name_uses_unknown_stem(self):
        buf = io.StringIO(
            "BEGIN IONS\nSCANS=3\nPEPMASS=500.0\n100.0 50\nEND IONS\n"
        )
        spectra = list(mgf_io.get_spectra(buf))
        assert len(spectra) == 1
        assert spectra[0].identifier == "unknown:scan:3"

    def test_empty_file_and_zero_peak_spectrum(self, tmp_path):
        empty = tmp_path / "empty.mgf"
        empty.write_text("")
        assert list(mgf_io.get_spectra(str(empty))) == []

        zero_peaks = tmp_path / "zero_peaks.mgf"
        zero_peaks.write_text("BEGIN IONS\nSCANS=1\nPEPMASS=500.0\nEND IONS\n")
        spectra = list(mgf_io.get_spectra(str(zero_peaks)))
        assert len(spectra) == 1
        assert len(spectra[0].mz) == 0

    def test_malformed_spectra_skipped_without_aborting(self, tmp_path):
        mgf = tmp_path / "mixed.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "END IONS\n"
            # Missing PEPMASS -> KeyError -> skipped.
            "BEGIN IONS\n"
            "SCANS=2\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "END IONS\n"
            # Empty PEPMASS= -> TypeError in float(None) -> skipped.
            "BEGIN IONS\n"
            "SCANS=3\n"
            "PEPMASS=\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "END IONS\n"
            # Non-numeric RTINSECONDS -> ValueError -> skipped.
            "BEGIN IONS\n"
            "SCANS=4\n"
            "PEPMASS=550.0\n"
            "RTINSECONDS=not_a_number\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=5\n"
            "PEPMASS=600.0\n"
            "CHARGE=3+\n"
            "100.0 50\n"
            "END IONS\n"
        )
        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == [
            "mixed:scan:1",
            "mixed:scan:5",
        ]


# ----------------------------------------------------------------------
# Writing MGF representative spectra and intensity scaling
# ----------------------------------------------------------------------


class TestMgfWriteAndRoundtrip:
    def test_spectra_to_dicts_full_fields_and_sequential_scans(self):
        dicts = list(
            mgf_io._spectra_to_dicts(
                [
                    _make_consensus(cluster_id=10, charge=np.int32(2)),
                    _make_consensus(cluster_id=25, charge=np.int32(-1)),
                ]
            )
        )
        assert len(dicts) == 2
        assert dicts[0]["params"] == {
            "title": "falcon:cluster:10",
            "scans": 1,
            "pepmass": pytest.approx(500.0),
            "charge": 2,
            "rtinseconds": pytest.approx(120.0),
        }
        assert dicts[1]["params"]["title"] == "falcon:cluster:25"
        assert dicts[1]["params"]["scans"] == 2
        assert dicts[1]["params"]["charge"] == -1
        # Intensities [0.5, 1.0, 0.25] rescale to base peak 1000.
        np.testing.assert_allclose(
            dicts[0]["intensity array"], [500.0, 1000.0, 250.0], rtol=1e-5
        )

    @pytest.mark.parametrize("missing_charge", [np.nan, None])
    def test_missing_charge_omitted(self, missing_charge):
        dicts = list(
            mgf_io._spectra_to_dicts([_make_consensus(charge=missing_charge)])
        )
        params = dicts[0]["params"]
        assert "charge" not in params
        assert params["rtinseconds"] == pytest.approx(120.0)

    @pytest.mark.parametrize("missing_rt", [np.float32("nan"), None])
    def test_missing_rt_omitted(self, missing_rt):
        dicts = list(
            mgf_io._spectra_to_dicts([_make_consensus(rt=missing_rt)])
        )
        params = dicts[0]["params"]
        assert "rtinseconds" not in params
        assert params["charge"] == 2

    def test_roundtrip_positive_negative_and_missing_metadata(self, tmp_path):
        reps = [
            _make_consensus(
                precursor_mz=500.5,
                charge=np.int32(2),
                rt=np.float32(120.25),
                cluster_id=0,
            ),
            _make_consensus(
                precursor_mz=600.25,
                charge=np.int32(-2),
                rt=np.float32("nan"),
                cluster_id=1,
            ),
            _make_consensus(
                precursor_mz=700.0,
                charge=np.nan,
                rt=np.float32(45.0),
                cluster_id=2,
            ),
        ]
        out_path = str(tmp_path / "falcon.mgf")
        mgf_io.write_spectra(out_path, reps)

        read_back = list(mgf_io.get_spectra(out_path))
        assert len(read_back) == 3

        assert read_back[0].identifier == "falcon:cluster:0"
        assert read_back[0].precursor_mz == pytest.approx(500.5)
        assert read_back[0].precursor_charge == 2
        assert read_back[0].retention_time == pytest.approx(120.25)
        np.testing.assert_allclose(
            read_back[0].intensity, [500.0, 1000.0, 250.0], rtol=1e-4
        )

        assert read_back[1].identifier == "falcon:cluster:1"
        assert read_back[1].precursor_charge == -2
        assert math.isnan(read_back[1].retention_time)

        assert read_back[2].identifier == "falcon:cluster:2"
        assert read_back[2].precursor_charge is None
        assert read_back[2].retention_time == pytest.approx(45.0)

    def test_write_empty_spectra_list(self, tmp_path):
        out_path = tmp_path / "empty.mgf"
        mgf_io.write_spectra(str(out_path), [])
        assert out_path.is_file()
        assert list(mgf_io.get_spectra(str(out_path))) == []


class TestScaleIntensities:
    def test_scales_base_peak_to_1000_preserving_ratios(self):
        intensity = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        scaled = mgf_io._scale_intensities(intensity)
        np.testing.assert_allclose(
            scaled, np.array([500.0, 1000.0, 250.0], dtype=np.float32)
        )

    def test_empty_and_non_positive_arrays_unchanged(self):
        empty = np.array([], dtype=np.float32)
        assert len(mgf_io._scale_intensities(empty)) == 0

        zeros = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(mgf_io._scale_intensities(zeros), zeros)
