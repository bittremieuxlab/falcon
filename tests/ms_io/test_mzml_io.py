"""Unit tests for falcon.ms_io.mzml_io."""

import logging
import math

import numpy as np
import pyteomics.mzml
import pytest

from falcon.ms_io import mzml_io


def _mzml_dict(scan=None, ion_extra=None, extra_ions=None):
    """Build a minimal Pyteomics-style mzML spectrum dict."""
    ions = [{"selected ion m/z": 500.0, **(ion_extra or {})}]
    if extra_ions:
        ions.extend(extra_ions)
    return {
        "id": "controllerType=0 controllerNumber=1 scan=5",
        "ms level": 2,
        "m/z array": np.array([100.0, 200.0], dtype=np.float32),
        "intensity array": np.array([1.0, 2.0], dtype=np.float32),
        "scanList": {"scan": [scan if scan is not None else {}]},
        "precursorList": {
            "precursor": [{"selectedIonList": {"selectedIon": ions}}]
        },
    }


class TestMzmlParseSpectrum:
    def test_missing_rt_is_nan(self):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        orig_id = raw["id"]
        spec = mzml_io._parse_spectrum(raw, "run:scan:5")
        assert spec.identifier == "run:scan:5"
        assert raw["id"] == orig_id
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == pytest.approx(500.0)
        assert spec.precursor_charge == 2

    def test_rt_present(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(
                scan={"scan start time": 6.0}, ion_extra={"charge state": 2}
            ),
            "run:scan:5",
        )
        assert spec.retention_time == pytest.approx(6.0)

    def test_possible_charge_state_fallback(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"possible charge state": 3}),
            "run:scan:5",
        )
        assert spec.precursor_charge == 3

    def test_charge_state_takes_precedence_over_possible(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(
                ion_extra={"charge state": 2, "possible charge state": 3}
            ),
            "run:scan:5",
        )
        assert spec.precursor_charge == 2

    def test_missing_charge_is_none(self):
        spec = mzml_io._parse_spectrum(_mzml_dict(), "run:scan:5")
        assert spec.precursor_charge is None

    def test_multiple_selected_ions_uses_first(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(
                ion_extra={"charge state": 2},
                extra_ions=[{"selected ion m/z": 600.0, "charge state": 3}],
            ),
            "run:scan:5",
        )
        assert spec.precursor_mz == pytest.approx(500.0)
        assert spec.precursor_charge == 2


class TestMzmlGetSpectra:
    def test_sample_fixture_filters_ms1_and_extracts_metadata(self, data_dir):
        spectra = list(mzml_io.get_spectra(data_dir / "sample.mzml"))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "sample:scan:2"
        assert s.precursor_mz == pytest.approx(400.0)
        assert s.precursor_charge == 2
        assert s.retention_time == pytest.approx(6.0)
        np.testing.assert_allclose(s.mz, [100.0])
        np.testing.assert_allclose(s.intensity, [100.0])

    def test_get_spectra_does_not_mutate_spectrum_dict_id(self, monkeypatch):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        orig_id = raw["id"]

        class _FakeMzML:
            def __init__(self, _source):
                pass

            def __enter__(self):
                return iter([raw])

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(pyteomics.mzml, "MzML", _FakeMzML)
        spectra = list(mzml_io.get_spectra("run.mzml"))
        assert [s.identifier for s in spectra] == ["run:scan:5"]
        assert raw["id"] == orig_id

    def test_retains_ms3_and_skips_missing_ms_level(
        self, tmp_path, mzml_writer
    ):
        path = mzml_writer(
            tmp_path / "levels.mzml",
            [
                {
                    "scan": 1,
                    "ms_level": 1,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 2,
                    "ms_level": None,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 3,
                    "ms_level": 3,
                    "precursor_mz": 450.0,
                    "charge": 2,
                    "rt": 12.5,
                    "mz": [150.0, 300.0],
                    "intensity": [20.0, 40.0],
                },
            ],
        )
        spectra = list(mzml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["levels:scan:3"]
        assert spectra[0].precursor_mz == pytest.approx(450.0)
        assert spectra[0].retention_time == pytest.approx(12.5)

    def test_skips_unrecognized_id_and_missing_precursor(
        self, tmp_path, caplog, mzml_writer
    ):
        path = mzml_writer(
            tmp_path / "skip.mzml",
            [
                # No "scan=" in id -> warning and skipped.
                {
                    "id_str": "spectrum=1",
                    "ms_level": 2,
                    "precursor_mz": 400.0,
                    "charge": 2,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                # Missing precursorList -> KeyError -> skipped.
                {
                    "scan": 2,
                    "ms_level": 2,
                    "precursor_mz": None,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                # Valid MS2 spectrum -> retained.
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
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(mzml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["skip:scan:3"]
        assert any("no 'scan=' field" in r.message for r in caplog.records)

    def test_malformed_xml_logs_warning_instead_of_crashing(
        self, tmp_path, caplog
    ):
        corrupt = tmp_path / "corrupt.mzml"
        corrupt.write_text("<mzML><run><spectrumList><spectrum")
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(mzml_io.get_spectra(corrupt))
        assert spectra == []
        assert any("Failed to read file" in r.message for r in caplog.records)

    def test_reads_from_open_file_object(self, data_dir):
        with (data_dir / "sample.mzml").open("rb") as fh:
            spectra = list(mzml_io.get_spectra(fh))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"

    def test_skips_spectrum_with_empty_selected_ion_list(self, tmp_path):
        # Empty <selectedIonList count="0"/> triggers IndexError.
        bad_mzml = tmp_path / "empty_ion.mzml"
        bad_mzml.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
            ' <run id="run1">\n'
            '  <spectrumList count="1">\n'
            '   <spectrum index="0" id="scan=1" defaultArrayLength="0">\n'
            '    <cvParam cvRef="MS" accession="MS:1000511" '
            'name="ms level" value="2"/>\n'
            '    <scanList count="1"><scan/></scanList>\n'
            '    <precursorList count="1"><precursor>'
            '<selectedIonList count="0"/>'
            "</precursor></precursorList>\n"
            '    <binaryDataArrayList count="0"/>\n'
            "   </spectrum>\n"
            "  </spectrumList>\n"
            " </run>\n"
            "</mzML>\n",
            encoding="utf-8",
        )
        assert list(mzml_io.get_spectra(bad_mzml)) == []
