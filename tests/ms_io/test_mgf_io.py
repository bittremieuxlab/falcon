"""Unit tests for falcon.ms_io.mgf_io."""

import io
import math

import numpy as np
import pyteomics.mgf
import pytest

from falcon.ms_io import mgf_io

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
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "basic:scan:1"
        assert s.precursor_mz == pytest.approx(500.25)
        # First charge is chosen when multiple charges are listed.
        assert s.precursor_charge == 2
        assert s.retention_time == pytest.approx(123.5)
        np.testing.assert_allclose(s.mz, [100.0, 200.0])
        np.testing.assert_allclose(s.intensity, [50.0, 100.0])

    def test_dotted_filename_preserves_full_stem(
        self, tmp_path, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            tmp_path / "sample.v1.mgf",
            [mgf_block([(100.0, 50.0)], 500.0, charge=None, scan=7)],
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == ["sample.v1:scan:7"]

    def test_missing_charge_and_rt(self, tmp_path, mgf_block, mgf_writer):
        mgf = mgf_writer(
            tmp_path / "minimal.mgf",
            [mgf_block([(100.0, 50.0), (200.0, 100.0)], 400.0, charge=None)],
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge is None
        assert math.isnan(spectra[0].retention_time)

    def test_negative_charge(self, tmp_path, mgf_block, mgf_writer):
        mgf = mgf_writer(
            tmp_path / "neg.mgf",
            [
                mgf_block(
                    [(100.0, 50.0), (200.0, 100.0)], 500.0, charge=-3, scan=1
                )
            ],
        )
        spectra = list(mgf_io.get_spectra(mgf))
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
    def test_usi_titles_preserved(self, tmp_path, usi, mgf_block, mgf_writer):
        mgf = mgf_writer(
            tmp_path / "run.mgf",
            [
                mgf_block(
                    [(100.0, 50.0)], 500.0, charge=None, scan=99, title=usi
                )
            ],
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert len(spectra) == 1
        assert spectra[0].identifier == usi

    def test_cluster_title_preserved_only_for_matching_stem(
        self, tmp_path, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            tmp_path / "myrun.mgf",
            [
                mgf_block(
                    [(100.0, 50.0)],
                    500.0,
                    charge=None,
                    scan=1,
                    title="myrun:cluster:7",
                ),
                mgf_block(
                    [(100.0, 50.0)],
                    500.0,
                    charge=None,
                    scan=2,
                    title="otherfile:cluster:8",
                ),
            ],
        )
        spectra = list(mgf_io.get_spectra(mgf))
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
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
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
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert spectra[0].identifier == "original:scan:5"

    def test_stream_without_name_uses_unknown_stem(self, mgf_block):
        buf = io.StringIO(
            mgf_block([(100.0, 50.0)], 500.0, charge=None, scan=3)
        )
        spectra = list(mgf_io.get_spectra(buf))
        assert len(spectra) == 1
        assert spectra[0].identifier == "unknown:scan:3"

    def test_empty_file_and_zero_peak_spectrum(
        self, tmp_path, mgf_block, mgf_writer
    ):
        empty = mgf_writer(tmp_path / "empty.mgf", [])
        assert list(mgf_io.get_spectra(empty)) == []

        zero_peaks = mgf_writer(
            tmp_path / "zero_peaks.mgf",
            [mgf_block([], 500.0, charge=None, scan=1)],
        )
        spectra = list(mgf_io.get_spectra(zero_peaks))
        assert len(spectra) == 1
        assert len(spectra[0].mz) == 0

    def test_reading_does_not_mutate_input_params_title(self, monkeypatch):
        raw_entry = {
            "params": {
                "title": "original plain title",
                "scans": "4",
                "pepmass": (500.0, None),
                "charge": [2],
            },
            "m/z array": np.array([100.0, 200.0], dtype=np.float32),
            "intensity array": np.array([50.0, 100.0], dtype=np.float32),
        }
        original_params = dict(raw_entry["params"])

        class _FakeMGF:
            def __init__(self, _source):
                pass

            def __enter__(self):
                return iter([raw_entry])

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(pyteomics.mgf, "MGF", _FakeMGF)
        spectra = list(mgf_io.get_spectra("run.mgf"))
        assert [s.identifier for s in spectra] == ["run:scan:4"]
        assert raw_entry["params"] == original_params
        assert raw_entry["params"]["title"] == "original plain title"

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
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [
            "mixed:scan:1",
            "mixed:scan:5",
        ]


# ----------------------------------------------------------------------
# Writing MGF representative spectra and intensity scaling
# ----------------------------------------------------------------------


class TestMgfWriteAndRoundtrip:
    def test_spectra_to_dicts_full_fields_and_sequential_scans(
        self, consensus_factory
    ):
        dicts = list(
            mgf_io._spectra_to_dicts(
                [
                    consensus_factory(cluster_id=10, charge=2),
                    consensus_factory(cluster_id=25, charge=-1),
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
    def test_missing_charge_omitted(self, missing_charge, consensus_factory):
        spec = consensus_factory(charge=missing_charge)
        if missing_charge is None:
            spec = spec._replace(precursor_charge=None)
        dicts = list(mgf_io._spectra_to_dicts([spec]))
        params = dicts[0]["params"]
        assert "charge" not in params
        assert params["rtinseconds"] == pytest.approx(120.0)

    @pytest.mark.parametrize("missing_rt", [np.float32("nan"), None])
    def test_missing_rt_omitted(self, missing_rt, consensus_factory):
        spec = consensus_factory(rt=missing_rt)
        if missing_rt is None:
            spec = spec._replace(retention_time=None)
        dicts = list(mgf_io._spectra_to_dicts([spec]))
        params = dicts[0]["params"]
        assert "rtinseconds" not in params
        assert params["charge"] == 2

    def test_roundtrip_positive_negative_and_missing_metadata(
        self, tmp_path, consensus_factory
    ):
        reps = [
            consensus_factory(
                precursor_mz=500.5,
                charge=2,
                rt=120.25,
                cluster_id=0,
            ),
            consensus_factory(
                precursor_mz=600.25,
                charge=-2,
                rt=None,
                cluster_id=1,
            ),
            consensus_factory(
                precursor_mz=700.0,
                charge=None,
                rt=45.0,
                cluster_id=2,
            ),
        ]
        out_path = tmp_path / "falcon.mgf"
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
        assert read_back[1].retention_time is not None
        assert math.isnan(read_back[1].retention_time)

        assert read_back[2].identifier == "falcon:cluster:2"
        assert read_back[2].precursor_charge is None
        assert read_back[2].retention_time == pytest.approx(45.0)

    def test_write_empty_spectra_list(self, tmp_path):
        out_path = tmp_path / "empty.mgf"
        mgf_io.write_spectra(out_path, [])
        assert out_path.is_file()
        assert list(mgf_io.get_spectra(out_path)) == []


class TestScaleIntensities:
    def test_scales_base_peak_to_1000_preserving_ratios(self):
        intensity = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        scaled = mgf_io._scale_intensities(intensity)
        np.testing.assert_allclose(
            scaled, np.array([500.0, 1000.0, 250.0], dtype=np.float32)
        )

    def test_empty_non_positive_and_non_finite_arrays_unchanged(self):
        empty = np.array([], dtype=np.float32)
        assert len(mgf_io._scale_intensities(empty)) == 0

        zeros = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(mgf_io._scale_intensities(zeros), zeros)

        non_finite = np.array([1.0, np.inf], dtype=np.float32)
        np.testing.assert_array_equal(
            mgf_io._scale_intensities(non_finite), non_finite
        )
