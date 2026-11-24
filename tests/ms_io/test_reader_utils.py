"""Unit tests for falcon.ms_io.reader_utils."""

import binascii
import io
import logging
import math
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import numpy as np
import pytest
from lxml import etree
from lxml.etree import LxmlError
from pyteomics.auxiliary import PyteomicsError
from pyteomics.auxiliary.structures import unitfloat, unitstr

from falcon.ms_io import reader_utils


class _BytesPath:
    """Minimal os.PathLike returning bytes."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __fspath__(self) -> bytes:
        return self._raw


class TestStdlibOnlyImport:
    def test_reader_utils_and_ms_io_defer_heavy_dependencies(self):
        script = (
            "import sys\n"
            "import falcon.ms_io\n"
            "import falcon.ms_io.reader_utils\n"
            "heavy = ('numpy', 'pyteomics', 'lxml', "
            "'spectrum_utils', 'psims')\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'Heavy modules loaded eagerly: {loaded}'\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


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


class TestParseOptionalRt:
    @pytest.mark.parametrize(
        "raw_rt", [None, "", "   ", unitstr("", "second"), (), []]
    )
    def test_absent_blank_or_empty_returns_nan(self, raw_rt):
        rt = reader_utils.parse_optional_rt(raw_rt)
        assert type(rt) is float
        assert math.isnan(rt)

    @pytest.mark.parametrize(
        ("raw_rt", "expected"),
        [
            (12.5, 12.5),
            ("42.25", 42.25),
            (0, 0.0),
            (unitfloat(6.0, "second"), 6.0),
            ((42.5,), 42.5),
            ([10.0], 10.0),
            ((100.0, 110.0), 105.0),
            ([10.0, 20.0], 15.0),
        ],
    )
    def test_parses_scalars_singletons_and_range_midpoints(
        self, raw_rt, expected
    ):
        rt = reader_utils.parse_optional_rt(raw_rt)
        assert type(rt) is float
        assert rt == pytest.approx(expected)

    @pytest.mark.parametrize(
        "bad_range",
        [
            (float("inf"), float("-inf")),
            (float("-inf"), 10.0),
            (10.0, float("inf")),
        ],
    )
    def test_infinite_range_endpoints_raise_value_error(self, bad_range):
        with pytest.raises(
            ValueError,
            match="Retention time range endpoints must be finite or NaN",
        ):
            reader_utils.parse_optional_rt(bad_range)

    def test_non_numeric_string_raises_value_error(self):
        with pytest.raises(ValueError, match="could not convert"):
            reader_utils.parse_optional_rt("not_a_number")


class TestParseOptionalCharge:
    @pytest.mark.parametrize(
        "raw_charge",
        [None, "", "   ", [], (), [None], [""], ["   "]],
    )
    @pytest.mark.parametrize("negative", [False, True])
    def test_absent_blank_or_empty_returns_none(self, raw_charge, negative):
        assert (
            reader_utils.parse_optional_charge(raw_charge, negative=negative)
            is None
        )

    @pytest.mark.parametrize(
        ("raw_charge", "negative", "expected"),
        [
            (2, False, 2),
            (-2, False, -2),
            ("3", False, 3),
            (0, False, 0),
            ([2, 3], False, 2),
            ((4, 5), False, 4),
            (2, True, -2),
            (-2, True, -2),
            (0, True, 0),
            ([3, 4], True, -3),
        ],
    )
    def test_parses_scalars_sequences_and_negative_mode(
        self, raw_charge, negative, expected
    ):
        assert (
            reader_utils.parse_optional_charge(raw_charge, negative=negative)
            == expected
        )

    @pytest.mark.parametrize("bad_charge", ["2-", "abc", ["2-"]])
    def test_invalid_charge_string_raises_value_error(self, bad_charge):
        with pytest.raises(ValueError, match="invalid literal"):
            reader_utils.parse_optional_charge(bad_charge)


class TestPathStem:
    def test_extracts_stem_from_pathlike_and_returns_none_otherwise(self):
        assert reader_utils.path_stem("/data/runs/exp.v1.mzML") == "exp.v1"
        assert reader_utils.path_stem(Path("/dir/sample.mgf")) == "sample"
        assert reader_utils.path_stem(b"/dir/bytes_run.mgf") == "bytes_run"
        assert (
            reader_utils.path_stem(_BytesPath(b"/dir/bpath.mzml")) == "bpath"
        )
        for empty_or_invalid in (
            "",
            "   ",
            "   .mgf",
            "/",
            ".",
            "..",
            b"/",
            b".",
            b"..",
            b"   ",
            None,
            3,
            object(),
        ):
            assert reader_utils.path_stem(empty_or_invalid) is None


class TestBaseFilename:
    def test_string_bytes_and_path_return_stem(self):
        assert (
            reader_utils.base_filename("/data/runs/experiment.mzML", None)
            == "experiment"
        )
        assert (
            reader_utils.base_filename(Path("/data/runs/sample.v1.mgf"), None)
            == "sample.v1"
        )
        assert (
            reader_utils.base_filename(b"/data/runs/bytes_run.mgf", None)
            == "bytes_run"
        )
        assert (
            reader_utils.base_filename(_BytesPath(b"/dir/bpath.mzml"), None)
            == "bpath"
        )

    @pytest.mark.parametrize(
        "empty_stem",
        ["", "   ", "   .mgf", "/", ".", "..", b"/", b".", b".."],
    )
    def test_empty_or_root_stems_return_unknown(self, empty_stem):
        assert reader_utils.base_filename(empty_stem, None) == "unknown"

    def test_string_without_extension(self):
        assert reader_utils.base_filename("raw_run", None) == "raw_run"

    def test_file_object_uses_source_name(self, tmp_path):
        path = tmp_path / "run_alpha.mgf"
        path.write_text("", encoding="utf-8")
        with path.open() as fh:
            assert reader_utils.base_filename(fh, None) == "run_alpha"

    def test_ignores_int_fd_name_and_accepts_bytes_name_on_file_objects(self):
        fd_stream = cast("Any", SimpleNamespace(name=3))
        assert reader_utils.base_filename(fd_stream, None) == "unknown"

        reader = SimpleNamespace(name="/dir/from_reader.mzml")
        assert reader_utils.base_filename(fd_stream, reader) == "from_reader"

        bytes_stream = cast(
            "Any", SimpleNamespace(name=b"/dir/from_bytes_fh.mgf")
        )
        assert (
            reader_utils.base_filename(bytes_stream, None) == "from_bytes_fh"
        )

        root_stream = cast("Any", SimpleNamespace(name="/"))
        assert reader_utils.base_filename(root_stream, reader) == "from_reader"

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
    def test_builds_spectrum_with_integer_charge_and_plain_floats(self):
        mz = np.array([100.0, 200.0], dtype=np.float32)
        intensity = np.array([50.0, 100.0], dtype=np.float32)
        spec = reader_utils.build_msms_spectrum(
            "run:scan:1",
            unitfloat(500.25, "m/z"),
            2,
            mz,
            intensity,
            unitfloat(42.5, "second"),
        )
        assert spec.identifier == "run:scan:1"
        assert type(spec.precursor_mz) is float
        assert spec.precursor_mz == pytest.approx(500.25)
        assert spec.precursor_charge == 2
        assert type(spec.retention_time) is float
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

    @pytest.mark.parametrize(
        ("bad_mz", "pattern"),
        [
            (float("nan"), "Precursor m/z must be finite"),
            (float("inf"), "Precursor m/z must be finite"),
            (float("-inf"), "Precursor m/z must be finite"),
            ("not_a_number", "could not convert"),
            (unitstr("bad", "m/z"), "could not convert"),
        ],
    )
    def test_rejects_non_finite_or_non_numeric_precursor_mz(
        self, bad_mz, pattern
    ):
        mz = np.array([100.0], dtype=np.float32)
        intensity = np.array([10.0], dtype=np.float32)
        with pytest.raises(ValueError, match=pattern):
            reader_utils.build_msms_spectrum(
                "run:scan:1", bad_mz, 2, mz, intensity, 1.0
            )

    @pytest.mark.parametrize(
        ("bad_rt", "pattern"),
        [
            ("not_a_number", "could not convert"),
            (unitstr("bad", "second"), "could not convert"),
            (float("inf"), "Retention time must be finite or NaN"),
            (float("-inf"), "Retention time must be finite or NaN"),
        ],
    )
    def test_rejects_non_numeric_or_infinite_retention_time(
        self, bad_rt, pattern
    ):
        mz = np.array([100.0], dtype=np.float32)
        intensity = np.array([10.0], dtype=np.float32)
        with pytest.raises(ValueError, match=pattern):
            reader_utils.build_msms_spectrum(
                "run:scan:1", 500.0, 2, mz, intensity, bad_rt
            )


class TestLogSkippedSpectrum:
    def test_emits_debug_record_with_default_and_custom_logger(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="falcon"):
            reader_utils.log_skipped_spectrum(
                "run.mgf", "run:scan:9", ValueError("bad pepmass")
            )
            custom_log = logging.getLogger("falcon.custom_skip")
            reader_utils.log_skipped_spectrum(
                "run.mzml",
                "run:scan:10",
                KeyError("precursorList"),
                log=custom_log,
            )
        assert len(caplog.records) == 2
        first, second = caplog.records
        assert first.levelno == logging.DEBUG
        assert first.name == "falcon.ms_io.reader_utils"
        assert "run:scan:9" in first.message
        assert "run.mgf" in first.message
        assert "bad pepmass" in first.message

        assert second.levelno == logging.DEBUG
        assert second.name == "falcon.custom_skip"
        assert "run:scan:10" in second.message


def _unreachable_parse(_spec_dict: dict[str, Any], _id: str) -> NoReturn:
    raise AssertionError("parse_spectrum should not be called")


class TestIterXmlSpectra:
    def test_filters_ms1_skips_errors_and_preserves_input_dicts(
        self, tmp_path, caplog, fake_reader_factory
    ):
        received_sources: list[Any] = []
        raw_dicts = [
            {"id": "scan=1", "ms level": 1},
            {"id": "scan=2"},  # Missing ms level -> skipped.
            {"id": "no_scan_field", "ms level": 2},
            {
                "id": "scan=4",
                "ms level": 2,
                "error": KeyError("precursorList"),
            },
            {"id": "scan=5", "ms level": 2, "error": OverflowError("too big")},
            {"id": "scan=6", "ms level": 2, "error": AttributeError("none")},
            {"id": "scan=7", "ms level": 2, "mz": 500.0},
            {"id": "scan=8", "ms level": 3, "mz": 600.0},
        ]
        snapshots = [dict(entry) for entry in raw_dicts]

        def _extract_scan_nr(spec_dict):
            raw_id = spec_dict["id"]
            if "scan=" not in raw_id:
                return None
            return raw_id.split("scan=", maxsplit=1)[1]

        def _parse(spec_dict, identifier):
            if "error" in spec_dict:
                raise spec_dict["error"]
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
                    fake_reader_factory(raw_dicts, sources=received_sources),
                    "ms level",
                    _extract_scan_nr,
                    _parse,
                    test_logger,
                )
            )

        assert received_sources == [str(source_path)]
        assert [s.identifier for s in spectra] == [
            "run:scan:7",
            "run:scan:8",
        ]
        assert raw_dicts == snapshots
        for skipped_id in ("scan=4", "scan=5", "scan=6"):
            assert any(
                rec.levelno == logging.DEBUG
                and rec.name == "falcon.test_iter_xml"
                and skipped_id in rec.message
                for rec in caplog.records
            )

    @pytest.mark.parametrize(
        "exc_to_raise",
        [
            LxmlError("truncated XML"),
            PyteomicsError("bad pyteomics format"),
            ValueError("buffer size must be a multiple of element size"),
            binascii.Error("corrupt base64"),
            zlib.error("corrupt zlib"),
        ],
    )
    def test_logs_warning_and_stops_on_whole_file_format_errors(
        self, caplog, exc_to_raise, fake_reader_factory
    ):
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
                    fake_reader_factory(
                        [{"id": "scan=1", "ms level": 2}],
                        error=exc_to_raise,
                    ),
                    "ms level",
                    lambda d: d["id"].split("=")[1],
                    _parse,
                    test_logger,
                )
            )

        assert [s.identifier for s in spectra] == ["broken:scan:1"]
        assert any(
            rec.levelno == logging.WARNING
            and rec.name == "falcon.test_iter_xml_err"
            and "Failed to read file" in rec.message
            for rec in caplog.records
        )

    def test_catches_xml_syntax_error_during_reader_construction(self, caplog):
        def _failing_opener(_source):
            etree.fromstring(b"")

        test_logger = logging.getLogger("falcon.test_iter_xml_ctor")
        with caplog.at_level(logging.WARNING, logger="falcon"):
            spectra = list(
                reader_utils.iter_xml_spectra(
                    "empty.mzml",
                    _failing_opener,
                    "ms level",
                    lambda d: d["id"],
                    _unreachable_parse,
                    test_logger,
                )
            )

        assert spectra == []
        assert any(
            rec.levelno == logging.WARNING
            and "Failed to read file" in rec.message
            for rec in caplog.records
        )

    def test_propagates_os_error_uncaught(self):
        def _missing_opener(_source):
            raise FileNotFoundError("no such file")

        test_logger = logging.getLogger("falcon.test_iter_xml_os")
        with pytest.raises(FileNotFoundError, match="no such file"):
            list(
                reader_utils.iter_xml_spectra(
                    "missing.mzml",
                    _missing_opener,
                    "ms level",
                    lambda d: d["id"],
                    _unreachable_parse,
                    test_logger,
                )
            )
