"""Unit tests for falcon.ms_io.mzxml_io."""

import logging
import math

import numpy as np
import pyteomics.mzxml
import pytest

from falcon.ms_io import mzxml_io


def _mzxml_dict(extra_precursor=None, rt=None, more_precursors=None):
    """Build a minimal Pyteomics-style mzXML spectrum dict."""
    precursors = [{"precursorMz": 400.0, **(extra_precursor or {})}]
    if more_precursors:
        precursors.extend(more_precursors)
    d = {
        "id": "7",
        "msLevel": 2,
        "m/z array": np.array([100.0, 200.0], dtype=np.float32),
        "intensity array": np.array([1.0, 2.0], dtype=np.float32),
        "precursorMz": precursors,
    }
    if rt is not None:
        d["retentionTime"] = rt
    return d


class TestMzxmlParseSpectrum:
    def test_missing_rt_and_charge(self):
        raw = _mzxml_dict()
        orig_id = raw["id"]
        spec = mzxml_io._parse_spectrum(raw, "run:scan:7")
        assert spec.identifier == "run:scan:7"
        assert raw["id"] == orig_id
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == pytest.approx(400.0)
        assert spec.precursor_charge is None

    def test_rt_and_charge_present(self):
        spec = mzxml_io._parse_spectrum(
            _mzxml_dict(extra_precursor={"precursorCharge": 2}, rt=6.0),
            "run:scan:7",
        )
        assert spec.retention_time == pytest.approx(6.0)
        assert spec.precursor_charge == 2

    def test_multiple_precursors_uses_first(self):
        spec = mzxml_io._parse_spectrum(
            _mzxml_dict(
                extra_precursor={"precursorCharge": 2},
                more_precursors=[{"precursorMz": 550.0, "precursorCharge": 3}],
            ),
            "run:scan:7",
        )
        assert spec.precursor_mz == pytest.approx(400.0)
        assert spec.precursor_charge == 2


class TestMzxmlGetSpectra:
    def test_sample_fixture_filters_ms1_and_extracts_metadata(self, data_dir):
        spectra = list(mzxml_io.get_spectra(data_dir / "sample.mzxml"))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "sample:scan:2"
        assert s.precursor_mz == pytest.approx(400.0)
        assert s.precursor_charge == 2
        # Pyteomics converts ISO duration PT6S to 0.1 minutes; the
        # sample.mzxml binary block encodes (m/z=160.0, int=10.0).
        assert s.retention_time == pytest.approx(0.1)
        np.testing.assert_allclose(s.mz, [160.0])
        np.testing.assert_allclose(s.intensity, [10.0])

    def test_get_spectra_does_not_mutate_spectrum_dict_id(self, monkeypatch):
        raw = _mzxml_dict(extra_precursor={"precursorCharge": 2})
        orig_id = raw["id"]

        class _FakeMzXML:
            def __init__(self, _source):
                pass

            def __enter__(self):
                return iter([raw])

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(pyteomics.mzxml, "MzXML", _FakeMzXML)
        spectra = list(mzxml_io.get_spectra("run.mzxml"))
        assert [s.identifier for s in spectra] == ["run:scan:7"]
        assert raw["id"] == orig_id

    def test_reads_from_open_file_object(self, data_dir):
        with (data_dir / "sample.mzxml").open("rb") as fh:
            spectra = list(mzxml_io.get_spectra(fh))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"

    def test_retains_ms3_and_skips_ms1(self, tmp_path, mzxml_writer):
        path = mzxml_writer(
            tmp_path / "levels.mzxml",
            [
                {
                    "scan": 1,
                    "ms_level": 1,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 3,
                    "ms_level": 3,
                    "precursor_mz": 450.0,
                    "charge": 2,
                    "rt": 60,
                    "mz": [150.0, 300.0],
                    "intensity": [20.0, 40.0],
                },
            ],
        )
        spectra = list(mzxml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["levels:scan:3"]
        assert spectra[0].retention_time == pytest.approx(1.0)

    def test_skips_ms2_scan_missing_precursor(self, tmp_path, mzxml_writer):
        path = mzxml_writer(
            tmp_path / "bad.mzxml",
            [
                # Missing precursorMz -> KeyError -> logged and skipped.
                {
                    "scan": 1,
                    "ms_level": 2,
                    "precursor_mz": None,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                # Valid scan -> retained.
                {
                    "scan": 3,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 3,
                    "mz": [120.0, 240.0],
                    "intensity": [15.0, 30.0],
                },
            ],
        )
        spectra = list(mzxml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["bad:scan:3"]

    def test_malformed_xml_logs_warning_instead_of_crashing(
        self, tmp_path, caplog
    ):
        corrupt = tmp_path / "corrupt.mzxml"
        corrupt.write_text("<mzXML><msRun><scan num='1'")
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(mzxml_io.get_spectra(corrupt))
        assert spectra == []
        assert any("Failed to read file" in r.message for r in caplog.records)

    def test_skips_empty_precursor_charge_type_error(
        self, tmp_path, mzxml_writer
    ):
        # Empty precursorCharge="" yields None in Pyteomics -> TypeError
        # on int(None); reader must skip it instead of aborting.
        path = mzxml_writer(
            tmp_path / "empty_charge.mzxml",
            [
                {
                    "scan": 1,
                    "ms_level": 2,
                    "precursor_mz": 400.0,
                    "charge": "",
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 5,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "mz": [120.0],
                    "intensity": [20.0],
                },
            ],
        )
        spectra = list(mzxml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["empty_charge:scan:5"]
