"""Unit tests for falcon.ms_io format dispatch and lazy imports."""

import shutil

import pytest

from falcon import ms_io


class TestLazySubmoduleImports:
    @pytest.mark.parametrize("name", ["mgf_io", "mzml_io", "mzxml_io"])
    def test_lazy_getattr_loads_format_submodules(self, monkeypatch, name):
        monkeypatch.delattr(ms_io, name, raising=False)
        mod = getattr(ms_io, name)
        assert mod.__name__ == f"falcon.ms_io.{name}"
        assert callable(mod.get_spectra)

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
            _ = ms_io.unknown


class TestGetSpectraDispatch:
    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.mgf"
        with pytest.raises(FileNotFoundError, match="Non-existing peak file"):
            list(ms_io.get_spectra(missing))

    def test_directory_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Non-existing peak file"):
            list(ms_io.get_spectra(tmp_path))

    def test_unknown_extension_raises_value_error(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown spectrum file type"):
            list(ms_io.get_spectra(path))

    @pytest.mark.parametrize("ext", [".mgf", ".MGF"])
    def test_dispatches_mgf_case_insensitively(
        self, tmp_path, ext, mgf_block, mgf_writer
    ):
        path = mgf_writer(
            tmp_path / f"sample{ext}",
            [mgf_block([(100.0, 50.0), (200.0, 100.0)], 500.0, scan=1)],
        )
        spectra = list(ms_io.get_spectra(path))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:1"

    @pytest.mark.parametrize("ext", [".mzml", ".MZML"])
    def test_dispatches_mzml_case_insensitively(self, tmp_path, data_dir, ext):
        dst = tmp_path / f"sample{ext}"
        shutil.copyfile(data_dir / "sample.mzml", dst)
        spectra = list(ms_io.get_spectra(dst))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"

    @pytest.mark.parametrize("ext", [".mzxml", ".MZXML"])
    def test_dispatches_mzxml_case_insensitively(
        self, tmp_path, data_dir, ext
    ):
        dst = tmp_path / f"sample{ext}"
        shutil.copyfile(data_dir / "sample.mzxml", dst)
        spectra = list(ms_io.get_spectra(dst))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"


class TestWriteSpectraDispatch:
    @pytest.mark.parametrize("ext", [".mzml", ".mzxml", ".txt"])
    def test_unsupported_format_raises(self, tmp_path, ext):
        with pytest.raises(ValueError, match="Unsupported peak file format"):
            ms_io.write_spectra(tmp_path / f"out{ext}", [])

    @pytest.mark.parametrize("ext", [".mgf", ".MGF"])
    def test_writes_mgf_case_insensitively(
        self, tmp_path, ext, consensus_factory
    ):
        rep = consensus_factory(
            precursor_mz=500.0,
            charge=2,
            mz=(150.0, 300.0),
            intensity=(0.5, 1.0),
            rt=42.0,
            cluster_id=7,
        )
        out = tmp_path / f"reps{ext}"
        ms_io.write_spectra(out, [rep])
        read_back = list(ms_io.get_spectra(out))
        assert len(read_back) == 1
        assert read_back[0].precursor_mz == pytest.approx(500.0)
        assert read_back[0].precursor_charge == 2
        assert read_back[0].retention_time == pytest.approx(42.0)
