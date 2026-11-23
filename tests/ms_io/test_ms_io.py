"""Unit tests for falcon.ms_io.ms_io format dispatch."""

import os
import shutil

import numpy as np
import pytest

from falcon.cluster.consensus import ConsensusTuple
from falcon.ms_io import ms_io

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TestGetSpectraDispatch:
    def test_nonexistent_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.mgf"
        with pytest.raises(ValueError, match="Non-existing peak file"):
            list(ms_io.get_spectra(str(missing)))

    def test_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Unknown spectrum file type"):
            list(ms_io.get_spectra(str(path)))

    @pytest.mark.parametrize("ext", [".mgf", ".MGF"])
    def test_dispatches_mgf_case_insensitively(self, tmp_path, ext):
        path = tmp_path / f"sample{ext}"
        path.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        spectra = list(ms_io.get_spectra(str(path)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:1"

    @pytest.mark.parametrize("ext", [".mzml", ".MZML"])
    def test_dispatches_mzml_case_insensitively(self, tmp_path, ext):
        dst = tmp_path / f"sample{ext}"
        shutil.copyfile(os.path.join(DATA_DIR, "sample.mzml"), dst)
        spectra = list(ms_io.get_spectra(str(dst)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"

    @pytest.mark.parametrize("ext", [".mzxml", ".MZXML"])
    def test_dispatches_mzxml_case_insensitively(self, tmp_path, ext):
        dst = tmp_path / f"sample{ext}"
        shutil.copyfile(os.path.join(DATA_DIR, "sample.mzxml"), dst)
        spectra = list(ms_io.get_spectra(str(dst)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"


class TestWriteSpectraDispatch:
    @pytest.mark.parametrize("ext", [".mzml", ".mzxml", ".txt"])
    def test_unsupported_format_raises(self, tmp_path, ext):
        with pytest.raises(ValueError, match="Unsupported peak file format"):
            ms_io.write_spectra(str(tmp_path / f"out{ext}"), [])

    @pytest.mark.parametrize("ext", [".mgf", ".MGF"])
    def test_writes_mgf_case_insensitively(self, tmp_path, ext):
        rep = ConsensusTuple(
            precursor_mz=np.float32(500.0),
            precursor_charge=np.int32(2),
            mz=np.array([150.0, 300.0], dtype=np.float32),
            intensity=np.array([0.5, 1.0], dtype=np.float32),
            retention_time=np.float32(42.0),
            cluster_id=np.int32(7),
            mz_split=np.int32(0),
        )
        out = tmp_path / f"reps{ext}"
        ms_io.write_spectra(str(out), [rep])
        read_back = list(ms_io.get_spectra(str(out)))
        assert len(read_back) == 1
        assert read_back[0].precursor_mz == pytest.approx(500.0)
        assert read_back[0].precursor_charge == 2
        assert read_back[0].retention_time == pytest.approx(42.0)
