"""Unit tests for falcon.ms_io readers, MGF writer, and dispatch."""

import logging
import math
import shutil

import numpy as np
import pytest

from falcon import ms_io
from tests.helpers import (
    make_consensus_record,
    write_mgf,
    write_toy_mzml,
    write_toy_mzxml,
)


class TestDispatch:
    def test_rejects_missing_directory_and_unsupported_extensions(
        self, tmp_path
    ):
        with pytest.raises(FileNotFoundError, match="Non-existing"):
            list(ms_io.get_spectra(tmp_path / "missing.mgf"))
        with pytest.raises(IsADirectoryError, match="directory"):
            list(ms_io.get_spectra(tmp_path))

        bad_ext = tmp_path / "test.xyz"
        bad_ext.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown spectrum file type"):
            list(ms_io.get_spectra(bad_ext))
        with pytest.raises(ValueError, match="Unsupported peak file format"):
            ms_io.write_spectra(tmp_path / "out.mzml", [])

    def test_case_insensitive_extension_dispatch(self, tmp_path, data_dir):
        for src_name, dst_name in (
            ("sample.mzml", "run.MZML"),
            ("sample.mzxml", "run.MZXML"),
        ):
            dst = tmp_path / dst_name
            shutil.copyfile(data_dir / src_name, dst)
            spectra = list(ms_io.get_spectra(dst))
            assert len(spectra) == 1
            assert spectra[0].identifier == "run:scan:2"


class TestMgfReaderAndWriter:
    def test_parses_metadata_rt_range_negative_charge_and_skips_malformed(
        self, tmp_path
    ):
        mgf = write_mgf(
            tmp_path / "run.v1.mgf",
            [
                "BEGIN IONS\n"
                "TITLE=mzspec:PXD000001:run:scan:10\n"
                "PEPMASS=500.25 1000\n"
                "CHARGE=2+\n"
                "RTINSECONDS=100.0-110.0\n"
                "SCANS=10\n"
                "150.0 100.0\n250.0 200.0\n"
                "END IONS\n",
                "BEGIN IONS\n"
                "TITLE=plain title\n"
                "PEPMASS=600.5\n"
                "CHARGE=3-\n"
                "SCAN=20\n"
                "200.0 50.0\n"
                "END IONS\n",
                "BEGIN IONS\n"
                "FILENAME=override.mzML\n"
                "TITLE=override:cluster:5\n"
                "PEPMASS=400.0\n"
                "100.0 10.0\n"
                "END IONS\n",
                "BEGIN IONS\nTITLE=bad_no_pepmass\n100.0 10.0\nEND IONS\n",
                "BEGIN IONS\nPEPMASS=nan\nSCANS=6\n100.0 10.0\nEND IONS\n",
                "BEGIN IONS\nPEPMASS=700.0\n100.0 10.0\nEND IONS\n",
            ],
        )
        spectra = list(ms_io.read_mgf(mgf))
        assert [s.identifier for s in spectra] == [
            "mzspec:PXD000001:run:scan:10",
            "run.v1:scan:20",
            "override:cluster:5",
            "run.v1:index:5",
        ]
        assert spectra[0].precursor_charge == 2
        assert spectra[0].retention_time == pytest.approx(105.0)
        assert spectra[1].precursor_charge == -3
        assert math.isnan(spectra[1].retention_time)
        assert spectra[2].precursor_charge is None

    def test_mgf_writer_roundtrip_scaling_and_atomic_failure(self, tmp_path):
        reps = [
            make_consensus_record(
                precursor_mz=500.0,
                charge=2,
                mz=(100.0, 200.0),
                intensity=(1e20, 2e20),
                rt=60.0,
                cluster_id=0,
            ),
            make_consensus_record(
                precursor_mz=600.0,
                charge=-2,
                mz=(150.0, 250.0),
                intensity=(0.25, 0.5),
                rt=None,
                cluster_id=1,
            ),
            make_consensus_record(
                precursor_mz=700.0,
                charge=None,
                mz=(300.0,),
                intensity=(1.0,),
                rt=12.0,
                cluster_id=2,
            ),
        ]
        out = tmp_path / "reps.MGF"
        ms_io.write_spectra(out, reps)
        text = out.read_text(encoding="utf-8")
        assert "CHARGE=2+" in text
        assert "CHARGE=2-" in text

        read_back = list(ms_io.get_spectra(out))
        assert len(read_back) == 3
        np.testing.assert_allclose(read_back[0].intensity, [500.0, 1000.0])
        assert read_back[1].precursor_charge == -2
        assert math.isnan(read_back[1].retention_time)
        assert read_back[2].precursor_charge is None

        bad_path = tmp_path / "bad.mgf"
        bad_rep = make_consensus_record(
            mz=(100.0, np.nan), intensity=(1.0, 1.0)
        )
        with pytest.raises(ValueError, match="finite"):
            ms_io.write_spectra(bad_path, [bad_rep])
        assert not bad_path.exists()


class TestXmlReaders:
    def test_mzml_sample_negative_scan_possible_charge_and_missing_scan(
        self, tmp_path, data_dir, caplog
    ):
        sample = list(ms_io.read_mzml(data_dir / "sample.mzml"))
        assert len(sample) == 1
        assert sample[0].identifier == "sample:scan:2"
        assert sample[0].precursor_mz == pytest.approx(400.0, rel=1e-4)
        assert sample[0].precursor_charge == 2

        custom = write_toy_mzml(
            tmp_path / "custom.mzml",
            [
                {
                    "scan": 1,
                    "ms_level": 1,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "id_str": "no_scan_token",
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 3,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "possible_charges": [3, 4],
                    "negative_scan": True,
                    "omit_scan_list": True,
                    "mz": [100.0, 200.0],
                    "intensity": [10.0, 20.0],
                },
            ],
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(ms_io.read_mzml(custom))
        assert len(spectra) == 1
        assert spectra[0].identifier == "custom:scan:3"
        assert spectra[0].precursor_charge == -3
        assert math.isnan(spectra[0].retention_time)
        assert any("unrecognized id" in r.message for r in caplog.records)

    def test_mzxml_sample_negative_polarity_and_bad_xml_warning(
        self, tmp_path, data_dir, caplog
    ):
        sample = list(ms_io.read_mzxml(data_dir / "sample.mzxml"))
        assert len(sample) == 1
        assert sample[0].identifier == "sample:scan:2"
        assert sample[0].precursor_charge == 2

        custom = write_toy_mzxml(
            tmp_path / "neg.mzxml",
            [
                {
                    "scan": 1,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "polarity": "-",
                    "mz": [100.0, 200.0],
                    "intensity": [10.0, 20.0],
                },
                {
                    "scan": 2,
                    "ms_level": 2,
                    "precursor_mz": 600.0,
                    "mz": [150.0],
                    "intensity": [30.0],
                },
            ],
        )
        spectra = list(ms_io.read_mzxml(custom))
        assert [s.precursor_charge for s in spectra] == [-2, None]

        broken = tmp_path / "broken.mzml"
        broken.write_text("<mzML><broken", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert list(ms_io.read_mzml(broken)) == []
        assert any("Failed to read file" in r.message for r in caplog.records)
