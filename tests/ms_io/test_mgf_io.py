"""Unit tests for falcon.ms_io.mgf_io."""

import io
import logging
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
        assert s.mz.dtype == np.float32
        assert s.intensity.dtype == np.float32
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
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)

    def test_rtinseconds_range_and_empty_string(self, tmp_path):
        mgf = tmp_path / "rt_variants.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "RTINSECONDS=100.0-110.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=2\n"
            "PEPMASS=500.0\n"
            "RTINSECONDS=\n"
            "100.0 50\n"
            "END IONS\n",
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="Could not parse RTINSECONDS"):
            spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [
            "rt_variants:scan:1",
            "rt_variants:scan:2",
        ]
        assert spectra[0].retention_time == pytest.approx(105.0)
        assert spectra[1].retention_time is not None
        assert math.isnan(spectra[1].retention_time)

    def test_parse_retention_time_edge_tuples(self):
        assert mgf_io._parse_retention_time((42.5,)) == pytest.approx(42.5)
        assert math.isnan(mgf_io._parse_retention_time(()))
        assert math.isnan(mgf_io._parse_retention_time("   "))

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

    @pytest.mark.parametrize(
        "bad_usi",
        [
            "mzspec:PXD000000:run:scan:abc",
            "mzspec:PXD000000:run",
            "mzspec:PXD 000:run:scan:1",
            "mzspec:PXD000000:run:scan:9:PEP:EXTRA",
        ],
    )
    def test_near_miss_usi_titles_fall_back_to_scan_identifier(
        self, tmp_path, bad_usi, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            tmp_path / "run.mgf",
            [
                mgf_block(
                    [(100.0, 50.0)],
                    500.0,
                    charge=None,
                    scan=12,
                    title=bad_usi,
                )
            ],
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == ["run:scan:12"]

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

    def test_blank_scans_and_zero_scan_resolution(self, tmp_path):
        mgf = tmp_path / "scans_edge.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=\n"
            "SCAN=77\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=   \n"
            "SCAN=\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=0\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [
            "scans_edge:scan:77",
            "scans_edge:index:1",
            "scans_edge:scan:0",
        ]
        # Numeric 0 in params dict is also preserved.
        assert (
            mgf_io._resolve_identifier({"scans": 0}, "run", 5) == "run:scan:0"
        )

    def test_filename_param_overrides_disk_stem_unless_empty_stem(
        self, tmp_path
    ):
        mgf = tmp_path / "ondisk.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "FILENAME=/nested/dir/original.raw\n"
            "SCANS=5\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "FILENAME=/\n"
            "SCANS=6\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "FILENAME=.\n"
            "SCANS=7\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n",
            encoding="utf-8",
        )
        spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [
            "original:scan:5",
            "ondisk:scan:6",
            "ondisk:scan:7",
        ]

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

    def test_reading_does_not_mutate_input_params_and_passes_options(
        self, monkeypatch, fake_reader_factory
    ):
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
        received_kwargs = []

        monkeypatch.setattr(
            pyteomics.mgf,
            "MGF",
            fake_reader_factory([raw_entry], kwargs_log=received_kwargs),
        )
        spectra = list(mgf_io.get_spectra("run.mgf"))
        assert [s.identifier for s in spectra] == ["run:scan:4"]
        assert raw_entry["params"] == original_params
        assert raw_entry["params"]["title"] == "original plain title"
        assert received_kwargs == [
            {
                "read_charges": False,
                "convert_arrays": 1,
                "dtype": np.float32,
            }
        ]

    def test_missing_params_key_uses_fallback_identifier_in_skip_log(
        self, monkeypatch, caplog, fake_reader_factory
    ):
        monkeypatch.setattr(
            pyteomics.mgf,
            "MGF",
            fake_reader_factory(
                [{"m/z array": np.array([], dtype=np.float32)}]
            ),
        )
        with caplog.at_level(logging.DEBUG, logger="falcon.ms_io.mgf_io"):
            assert list(mgf_io.get_spectra("broken.mgf")) == []
        assert any(
            rec.levelno == logging.DEBUG
            and rec.name == "falcon.ms_io.mgf_io"
            and "broken:index:0" in rec.message
            for rec in caplog.records
        )

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
            # Non-finite PEPMASS=nan -> ValueError -> skipped.
            "BEGIN IONS\n"
            "SCANS=5\n"
            "PEPMASS=nan\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=6\n"
            "PEPMASS=600.0\n"
            "CHARGE=3+\n"
            "100.0 50\n"
            "END IONS\n",
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="Could not parse RTINSECONDS"):
            spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [
            "mixed:scan:1",
            "mixed:scan:6",
        ]

    @pytest.mark.parametrize(
        ("name", "bad_line"),
        [
            ("bad_pepmass.mgf", "PEPMASS=abc"),
            ("empty_charge.mgf", "PEPMASS=500.0\nCHARGE="),
        ],
    )
    def test_reader_level_parse_errors_warn_and_return_cleanly(
        self, tmp_path, caplog, name, bad_line
    ):
        mgf = tmp_path / name
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "100.0 50\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=2\n"
            f"{bad_line}\n"
            "100.0 50\n"
            "END IONS\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="falcon.ms_io.mgf_io"):
            spectra = list(mgf_io.get_spectra(mgf))
        assert [s.identifier for s in spectra] == [f"{mgf.stem}:scan:1"]
        assert any(
            rec.levelno == logging.WARNING
            and "Failed to read file" in rec.message
            for rec in caplog.records
        )

    def test_missing_file_propagates_os_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(mgf_io.get_spectra(tmp_path / "missing.mgf"))


# ----------------------------------------------------------------------
# Writing MGF representative spectra and intensity scaling
# ----------------------------------------------------------------------


class TestMgfWriteAndRoundtrip:
    def test_format_spectrum_full_fields_and_sequential_scans(
        self, consensus_factory
    ):
        pos_block = mgf_io._format_spectrum(
            consensus_factory(cluster_id=10, charge=2), scan_nr=1
        )
        neg_block = mgf_io._format_spectrum(
            consensus_factory(cluster_id=25, charge=-1), scan_nr=2
        )
        zero_block = mgf_io._format_spectrum(
            consensus_factory(cluster_id=30, charge=0), scan_nr=3
        )
        assert pos_block == (
            "BEGIN IONS\n"
            "TITLE=falcon:cluster:10\n"
            "PEPMASS=500.0\n"
            "RTINSECONDS=120.0\n"
            "CHARGE=2+\n"
            "SCANS=1\n"
            "100.00000 500.0\n"
            "200.00000 1000.0\n"
            "300.00000 250.0\n"
            "END IONS\n\n"
        )
        assert "TITLE=falcon:cluster:25\n" in neg_block
        assert "CHARGE=1-\n" in neg_block
        assert "SCANS=2\n" in neg_block
        assert "CHARGE=0+\n" in zero_block

        # Verify direct output parses cleanly with pyteomics.mgf.read.
        parsed = list(
            pyteomics.mgf.read(
                io.StringIO(pos_block + neg_block + zero_block),
                use_index=False,
            )
        )
        assert [int(p["params"]["charge"][0]) for p in parsed] == [2, -1, 0]

    @pytest.mark.parametrize(
        "missing_charge", [np.nan, None, float("inf"), float("-inf")]
    )
    def test_missing_or_non_finite_charge_omitted(
        self, missing_charge, consensus_factory
    ):
        spec = consensus_factory(charge=2)._replace(
            precursor_charge=missing_charge
        )
        block = mgf_io._format_spectrum(spec, scan_nr=1)
        assert "CHARGE=" not in block
        assert "RTINSECONDS=120.0\n" in block

    @pytest.mark.parametrize(
        "missing_rt",
        [
            np.float32("nan"),
            None,
            np.float32("inf"),
            np.float32("-inf"),
        ],
    )
    def test_missing_or_non_finite_rt_omitted(
        self, missing_rt, consensus_factory
    ):
        spec = consensus_factory(rt=120.0)._replace(retention_time=missing_rt)
        block = mgf_io._format_spectrum(spec, scan_nr=1)
        assert "RTINSECONDS=" not in block
        assert "CHARGE=2+\n" in block

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

    def test_buffered_flush_produces_identical_output(
        self, tmp_path, consensus_factory, monkeypatch
    ):
        reps = [
            consensus_factory(cluster_id=i, precursor_mz=500.0 + i)
            for i in range(4)
        ]
        unflushed = tmp_path / "large_buf.mgf"
        flushed = tmp_path / "small_buf.mgf"
        mgf_io.write_spectra(unflushed, reps)
        monkeypatch.setattr(mgf_io, "_WRITE_BUFFER_CHARS", 128)
        mgf_io.write_spectra(flushed, reps)
        assert flushed.read_text(encoding="utf-8") == unflushed.read_text(
            encoding="utf-8"
        )

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

    def test_small_positive_float32_intensities_do_not_overflow(self):
        intensity = np.array([0.5e-37, 1.0e-37, 0.0], dtype=np.float32)
        scaled = mgf_io._scale_intensities(intensity)
        assert scaled.dtype == np.float32
        assert np.all(np.isfinite(scaled))
        np.testing.assert_allclose(
            scaled, np.array([500.0, 1000.0, 0.0], dtype=np.float32)
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
