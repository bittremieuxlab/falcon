"""Unit tests for falcon.ms_io.reader_utils."""

import io
import logging
from types import SimpleNamespace

from falcon.ms_io import reader_utils


class TestBaseFilename:
    def test_string_path_returns_stem(self):
        assert (
            reader_utils.base_filename("/data/runs/experiment.mzML", None)
            == "experiment"
        )

    def test_string_without_extension(self):
        assert reader_utils.base_filename("raw_run", None) == "raw_run"

    def test_file_object_uses_source_name(self, tmp_path):
        path = tmp_path / "run_alpha.mgf"
        path.write_text("")
        with path.open() as fh:
            assert reader_utils.base_filename(fh, None) == "run_alpha"

    def test_falls_back_to_reader_name_when_source_unnamed(self):
        buf = io.StringIO("")
        reader = SimpleNamespace(name="/dir/from_reader.mzxml")
        assert reader_utils.base_filename(buf, reader) == "from_reader"

    def test_returns_unknown_when_no_name_available(self):
        buf = io.StringIO("")
        reader = SimpleNamespace()
        assert reader_utils.base_filename(buf, reader) == "unknown"


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
