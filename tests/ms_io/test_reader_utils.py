"""Unit tests for falcon.ms_io.reader_utils."""

import io
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lxml.etree import LxmlError

from falcon.ms_io import reader_utils


class TestNormalizeSource:
    def test_pathlike_converted_to_string(self, tmp_path):
        path = tmp_path / "run.mgf"
        result = reader_utils.normalize_source(path)
        assert isinstance(result, str)
        assert result == str(path)

    def test_string_and_file_object_returned_unchanged(self):
        assert reader_utils.normalize_source("run.mgf") == "run.mgf"
        stream = io.StringIO("data")
        assert reader_utils.normalize_source(stream) is stream


class TestBaseFilename:
    def test_string_and_path_return_stem(self):
        assert (
            reader_utils.base_filename("/data/runs/experiment.mzML", None)
            == "experiment"
        )
        assert (
            reader_utils.base_filename(Path("/data/runs/sample.v1.mgf"), None)
            == "sample.v1"
        )

    def test_string_without_extension_and_empty_string(self):
        assert reader_utils.base_filename("raw_run", None) == "raw_run"
        assert reader_utils.base_filename("", None) == "unknown"

    def test_file_object_uses_source_name(self, tmp_path):
        path = tmp_path / "run_alpha.mgf"
        path.write_text("", encoding="utf-8")
        with path.open() as fh:
            assert reader_utils.base_filename(fh, None) == "run_alpha"

    def test_falls_back_to_reader_name_when_source_unnamed(self):
        buf = io.StringIO("")
        reader = SimpleNamespace(name="/dir/from_reader.mzxml")
        assert reader_utils.base_filename(buf, reader) == "from_reader"

        path_reader = SimpleNamespace(name=Path("/dir/from_path.mzml"))
        assert reader_utils.base_filename(buf, path_reader) == "from_path"

    def test_returns_unknown_when_no_name_available(self):
        buf = io.StringIO("")
        reader = SimpleNamespace()
        assert reader_utils.base_filename(buf, reader) == "unknown"


class TestBuildMsmsSpectrum:
    def test_builds_spectrum_with_integer_charge_and_rt(self):
        mz = np.array([100.0, 200.0], dtype=np.float32)
        intensity = np.array([50.0, 100.0], dtype=np.float32)
        spec = reader_utils.build_msms_spectrum(
            "run:scan:1", 500.25, 2, mz, intensity, 42.5
        )
        assert spec.identifier == "run:scan:1"
        assert spec.precursor_mz == pytest.approx(500.25)
        assert spec.precursor_charge == 2
        assert spec.retention_time == pytest.approx(42.5)
        np.testing.assert_allclose(spec.mz, mz)
        np.testing.assert_allclose(spec.intensity, intensity)

    def test_allows_none_charge_and_nan_rt(self):
        mz = np.array([150.0], dtype=np.float32)
        intensity = np.array([10.0], dtype=np.float32)
        spec = reader_utils.build_msms_spectrum(
            "run:scan:2", 400.0, None, mz, intensity, float("nan")
        )
        assert spec.precursor_charge is None
        assert math.isnan(spec.retention_time)


class TestLogSkippedSpectrum:
    def test_emits_debug_record_with_context(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="falcon"):
            reader_utils.log_skipped_spectrum(
                "run.mgf", "run:scan:9", ValueError("bad pepmass")
            )
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelno == logging.DEBUG
        assert "run:scan:9" in rec.message
        assert "run.mgf" in rec.message
        assert "bad pepmass" in rec.message


class TestIterXmlSpectra:
    def test_filters_ms1_skips_errors_and_preserves_input_dicts(
        self, tmp_path, caplog
    ):
        received_sources = []
        raw_dicts = [
            {"id": "scan=1", "ms level": 1},
            {"id": "scan=2"},  # Missing ms level -> skipped.
            {"id": "no_scan_field", "ms level": 2},
            {"id": "scan=4", "ms level": 2, "bad": True},
            {"id": "scan=5", "ms level": 2, "mz": 500.0},
            {"id": "scan=6", "ms level": 3, "mz": 600.0},
        ]
        snapshots = [dict(entry) for entry in raw_dicts]

        class _FakeReader:
            def __init__(self, xml_source):
                received_sources.append(xml_source)

            def __enter__(self):
                return iter(raw_dicts)

            def __exit__(self, *_args):
                return False

        def _extract_scan_nr(spec_dict):
            raw_id = spec_dict["id"]
            if "scan=" not in raw_id:
                return None
            return raw_id.split("scan=", maxsplit=1)[1]

        def _parse(spec_dict, identifier):
            if spec_dict.get("bad"):
                raise KeyError("precursorList")
            return reader_utils.build_msms_spectrum(
                identifier,
                spec_dict["mz"],
                2,
                np.array([100.0], dtype=np.float32),
                np.array([10.0], dtype=np.float32),
                1.0,
            )

        source_path = tmp_path / "run.mzml"
        test_logger = logging.getLogger("falcon.test_iter_xml")
        with caplog.at_level(logging.DEBUG, logger="falcon"):
            spectra = list(
                reader_utils.iter_xml_spectra(
                    source_path,
                    _FakeReader,
                    "ms level",
                    _extract_scan_nr,
                    _parse,
                    test_logger,
                )
            )

        assert received_sources == [str(source_path)]
        assert [s.identifier for s in spectra] == [
            "run:scan:5",
            "run:scan:6",
        ]
        assert raw_dicts == snapshots
        assert any(
            rec.levelno == logging.DEBUG and "scan=4" in rec.message
            for rec in caplog.records
        )

    def test_logs_warning_and_stops_on_lxml_error(self, caplog):
        class _BrokenReader:
            def __init__(self, _source):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                yield {"id": "scan=1", "ms level": 2}
                raise LxmlError("truncated XML")

        def _parse(_spec_dict, identifier):
            return reader_utils.build_msms_spectrum(
                identifier,
                400.0,
                2,
                np.array([100.0], dtype=np.float32),
                np.array([10.0], dtype=np.float32),
                1.0,
            )

        test_logger = logging.getLogger("falcon.test_iter_xml_err")
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(
                reader_utils.iter_xml_spectra(
                    "broken.mzml",
                    _BrokenReader,
                    "ms level",
                    lambda d: d["id"].split("=")[1],
                    _parse,
                    test_logger,
                )
            )

        assert [s.identifier for s in spectra] == ["broken:scan:1"]
        assert any(
            rec.levelno == logging.WARNING
            and "Failed to read file" in rec.message
            and "truncated XML" in rec.message
            for rec in caplog.records
        )
