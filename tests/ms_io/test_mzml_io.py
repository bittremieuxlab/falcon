"""Unit tests for falcon.ms_io.mzml_io."""

import logging
import math

import numpy as np
import psims.controlled_vocabulary.controlled_vocabulary as psims_cv
import pyteomics.mzml
import pyteomics.xml
import pytest
from pyteomics.auxiliary import PyteomicsError
from pyteomics.auxiliary.structures import unitstr

from falcon.ms_io import mzml_io


def _mzml_dict(scan=None, ion_extra=None, extra_ions=None, scan_nr=5):
    """Build a minimal Pyteomics-style mzML spectrum dict."""
    ions = [{"selected ion m/z": 500.0, **(ion_extra or {})}]
    if extra_ions:
        ions.extend(extra_ions)
    return {
        "id": f"controllerType=0 controllerNumber=1 scan={scan_nr}",
        "ms level": 2,
        "m/z array": np.array([100.0, 200.0], dtype=np.float32),
        "intensity array": np.array([1.0, 2.0], dtype=np.float32),
        "scanList": {"scan": [scan if scan is not None else {}]},
        "precursorList": {
            "precursor": [{"selectedIonList": {"selectedIon": ions}}]
        },
    }


def _read_mzml_dicts(
    monkeypatch, fake_reader_factory, raw_dicts, source="run.mzml"
):
    """Run ``mzml_io.get_spectra`` against stubbed Pyteomics records."""
    monkeypatch.setattr(pyteomics.mzml, "MzML", fake_reader_factory(raw_dicts))
    return list(mzml_io.get_spectra(source))


class TestOfflinePsimsCv:
    def test_caches_vendored_cv_without_network(self, monkeypatch):
        init_kwargs = []
        orig_obo_cache = psims_cv.OBOCache

        class _TrackingOBOCache(orig_obo_cache):
            def __init__(self, *args, **kwargs):
                init_kwargs.append(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(psims_cv, "OBOCache", _TrackingOBOCache)
        mzml_io._load_offline_psims_cv.cache_clear()
        try:
            cv1 = mzml_io._load_offline_psims_cv()
            cv2 = mzml_io._load_offline_psims_cv()
            assert cv1 is not None
            assert cv1 is cv2
            assert init_kwargs == [{"enabled": False, "use_remote": False}]
        finally:
            mzml_io._load_offline_psims_cv.cache_clear()

    def test_fallback_returns_none_and_omits_cv_kwarg(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("obo load failed")

        received_kwargs = []

        def _fake_mzml(_source, **kwargs):
            received_kwargs.append(kwargs)
            return object()

        monkeypatch.setattr(psims_cv.OBOCache, "load", _boom)
        monkeypatch.setattr(pyteomics.mzml, "MzML", _fake_mzml)
        mzml_io._load_offline_psims_cv.cache_clear()
        try:
            assert mzml_io._load_offline_psims_cv() is None
            mzml_io._open_mzml("sample.mzml")
            assert received_kwargs == [
                {"use_index": False, "dtype": np.float32}
            ]
        finally:
            mzml_io._load_offline_psims_cv.cache_clear()

    def test_open_mzml_omits_cv_when_cvparamparser_missing(self, monkeypatch):
        received_kwargs = []

        def _fake_mzml(_source, **kwargs):
            received_kwargs.append(kwargs)
            return object()

        monkeypatch.delattr(pyteomics.xml, "CVParamParser", raising=False)
        monkeypatch.setattr(pyteomics.mzml, "MzML", _fake_mzml)
        mzml_io._open_mzml("sample.mzml")
        assert received_kwargs == [{"use_index": False, "dtype": np.float32}]


class TestExtractScanNr:
    @pytest.mark.parametrize(
        ("raw_id", "expected"),
        [
            ("controllerType=0 controllerNumber=1 scan=5", "5"),
            ("scan=42 frame=7", "42"),
            ("prescan=1 scan=42", "42"),
            ("prescan=1", None),
            ("scan=", None),
            ("scan=   frame=7", None),
            ("spectrum=1", None),
            (None, None),
            (123, None),
        ],
    )
    def test_extracts_scan_token_strictly(self, raw_id, expected):
        assert mzml_io._extract_scan_nr({"id": raw_id}) == expected

    def test_missing_id_key_returns_none(self):
        assert mzml_io._extract_scan_nr({}) is None


class TestMzmlParseSpectrum:
    def test_missing_rt_is_nan(self, monkeypatch, fake_reader_factory):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        orig_id = raw["id"]
        spectra = _read_mzml_dicts(monkeypatch, fake_reader_factory, [raw])
        assert len(spectra) == 1
        spec = spectra[0]
        assert spec.identifier == "run:scan:5"
        assert raw["id"] == orig_id
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == pytest.approx(500.0)
        assert spec.precursor_charge == 2

    @pytest.mark.parametrize("scan_list", [None, {}, {"scan": []}])
    def test_absent_or_empty_scan_list_defaults_rt_to_nan(
        self, monkeypatch, fake_reader_factory, scan_list
    ):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        if scan_list is None:
            del raw["scanList"]
        else:
            raw["scanList"] = scan_list
        spectra = _read_mzml_dicts(monkeypatch, fake_reader_factory, [raw])
        assert len(spectra) == 1
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)

    @pytest.mark.parametrize(
        "blank_rt", [None, "", "   ", unitstr("", "second")]
    )
    def test_empty_or_none_scan_start_time_defaults_to_nan(
        self, monkeypatch, fake_reader_factory, blank_rt
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [_mzml_dict(scan={"scan start time": blank_rt})],
        )
        assert len(spectra) == 1
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)

    def test_rt_present(self, monkeypatch, fake_reader_factory):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(
                    scan={"scan start time": 6.0},
                    ion_extra={"charge state": 2},
                )
            ],
        )
        assert len(spectra) == 1
        assert spectra[0].retention_time == pytest.approx(6.0)

    def test_possible_charge_state_fallback_scalar_and_sequences(
        self, monkeypatch, fake_reader_factory
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(ion_extra={"possible charge state": 3}, scan_nr=1),
                _mzml_dict(
                    ion_extra={"possible charge state": [2, 3]}, scan_nr=2
                ),
                _mzml_dict(ion_extra={"possible charge state": []}, scan_nr=3),
            ],
        )
        assert [s.precursor_charge for s in spectra] == [3, 2, None]

    def test_charge_state_sequence_and_empty_fallback(
        self, monkeypatch, fake_reader_factory
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(ion_extra={"charge state": (4, 5)}, scan_nr=1),
                _mzml_dict(
                    ion_extra={
                        "charge state": [],
                        "possible charge state": [3, 4],
                    },
                    scan_nr=2,
                ),
                _mzml_dict(ion_extra={"charge state": ()}, scan_nr=3),
            ],
        )
        assert [s.precursor_charge for s in spectra] == [4, 3, None]

    def test_charge_state_takes_precedence_over_possible(
        self, monkeypatch, fake_reader_factory
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(
                    ion_extra={"charge state": 2, "possible charge state": 3}
                )
            ],
        )
        assert [s.precursor_charge for s in spectra] == [2]

    def test_missing_charge_is_none(self, monkeypatch, fake_reader_factory):
        spectra = _read_mzml_dicts(
            monkeypatch, fake_reader_factory, [_mzml_dict()]
        )
        assert [s.precursor_charge for s in spectra] == [None]

    @pytest.mark.parametrize("blank_charge", [None, "", "   ", [""], [None]])
    def test_empty_charge_cv_param_returns_none_and_allows_fallback(
        self, monkeypatch, fake_reader_factory, blank_charge
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(
                    ion_extra={"charge state": blank_charge}, scan_nr=1
                ),
                _mzml_dict(
                    ion_extra={
                        "charge state": blank_charge,
                        "possible charge state": 3,
                    },
                    scan_nr=2,
                ),
            ],
        )
        assert [s.precursor_charge for s in spectra] == [None, 3]

    @pytest.mark.parametrize(
        ("raw_charge", "expected"),
        [(2, -2), (-2, -2), (None, None)],
    )
    def test_negative_scan_negates_charge_magnitude(
        self, monkeypatch, fake_reader_factory, raw_charge, expected
    ):
        ion_extra = (
            {"charge state": raw_charge} if raw_charge is not None else {}
        )
        raw = {**_mzml_dict(ion_extra=ion_extra), "negative scan": ""}
        spectra = _read_mzml_dicts(monkeypatch, fake_reader_factory, [raw])
        assert [s.precursor_charge for s in spectra] == [expected]

    def test_multiple_selected_ions_uses_first(
        self, monkeypatch, fake_reader_factory
    ):
        spectra = _read_mzml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzml_dict(
                    ion_extra={"charge state": 2},
                    extra_ions=[
                        {"selected ion m/z": 600.0, "charge state": 3}
                    ],
                )
            ],
        )
        assert len(spectra) == 1
        assert spectra[0].precursor_mz == pytest.approx(500.0)
        assert spectra[0].precursor_charge == 2

    def test_skips_non_numeric_rt_and_non_finite_precursor_mz(
        self, monkeypatch, caplog, fake_reader_factory
    ):
        bad_rt_dict = _mzml_dict(
            scan={"scan start time": unitstr("bad", "second")}, scan_nr=1
        )
        bad_mz_dict = _mzml_dict(scan_nr=2)
        bad_mz_dict["precursorList"]["precursor"][0]["selectedIonList"][
            "selectedIon"
        ][0]["selected ion m/z"] = float("nan")
        valid_dict = _mzml_dict(scan_nr=3)

        with caplog.at_level(logging.DEBUG, logger="falcon.ms_io.mzml_io"):
            spectra = _read_mzml_dicts(
                monkeypatch,
                fake_reader_factory,
                [bad_rt_dict, bad_mz_dict, valid_dict],
            )
        assert [s.identifier for s in spectra] == ["run:scan:3"]
        messages = [
            rec.message
            for rec in caplog.records
            if rec.levelno == logging.DEBUG
        ]
        assert any("could not convert" in m for m in messages)
        assert any("Precursor m/z must be finite" in m for m in messages)


class TestMzmlGetSpectra:
    def test_sample_fixture_filters_ms1_and_extracts_metadata(self, data_dir):
        spectra = list(mzml_io.get_spectra(data_dir / "sample.mzml"))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "sample:scan:2"
        assert s.precursor_mz == pytest.approx(400.0)
        assert s.precursor_charge == 2
        assert s.retention_time == pytest.approx(6.0)
        assert s.mz.dtype == np.float32
        assert s.intensity.dtype == np.float32
        np.testing.assert_allclose(s.mz, [100.0])
        np.testing.assert_allclose(s.intensity, [100.0])

    def test_get_spectra_does_not_mutate_spectrum_dict_id_and_passes_options(
        self, monkeypatch, fake_reader_factory
    ):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        orig_id = raw["id"]
        received_kwargs = []

        monkeypatch.setattr(
            pyteomics.mzml,
            "MzML",
            fake_reader_factory([raw], kwargs_log=received_kwargs),
        )
        spectra = list(mzml_io.get_spectra("run.mzml"))
        assert [s.identifier for s in spectra] == ["run:scan:5"]
        assert raw["id"] == orig_id
        assert len(received_kwargs) == 1
        assert received_kwargs[0]["use_index"] is False
        assert received_kwargs[0]["dtype"] is np.float32
        assert received_kwargs[0]["cv"] is mzml_io._load_offline_psims_cv()

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

    def test_skips_unrecognized_id_warns_once_per_file_and_skips_bad_entries(
        self, tmp_path, caplog, mzml_writer
    ):
        path = mzml_writer(
            tmp_path / "skip.mzml",
            [
                # No "scan=" in id -> first WARNING, next two DEBUG.
                {
                    "id_str": "spectrum=1",
                    "ms_level": 2,
                    "precursor_mz": 400.0,
                    "charge": 2,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "id_str": "prescan=2",
                    "ms_level": 2,
                    "precursor_mz": 410.0,
                    "charge": 2,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "id_str": "scan=",
                    "ms_level": 2,
                    "precursor_mz": 420.0,
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
                # Valid MS2 with extra trailing token -> retained.
                {
                    "id_str": "prescan=1 scan=3 frame=7",
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 3,
                    "mz": [120.0, 240.0],
                    "intensity": [15.0, 30.0],
                },
            ],
        )
        with caplog.at_level(logging.DEBUG, logger="falcon.ms_io.mzml_io"):
            spectra = list(mzml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["skip:scan:3"]
        missing_scan_recs = [
            r for r in caplog.records if "no 'scan=' field" in r.message
        ]
        assert len(missing_scan_recs) == 3
        assert missing_scan_recs[0].levelno == logging.WARNING
        assert missing_scan_recs[1].levelno == logging.DEBUG
        assert missing_scan_recs[2].levelno == logging.DEBUG

    def test_parses_multi_possible_charge_state_and_missing_scanlist_end2end(
        self, tmp_path, mzml_writer
    ):
        mzml = mzml_writer(
            tmp_path / "multi_charge.mzml",
            [
                {
                    "id_str": "scan=1",
                    "precursor_mz": 500.0,
                    "possible_charges": [2, 3],
                    "omit_scan_list": True,
                    "mz": [100.0],
                    "intensity": [100.0],
                }
            ],
        )
        spectra = list(mzml_io.get_spectra(mzml))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge == 2
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)

    def test_parses_negative_scan_and_empty_cv_params_end2end(
        self, tmp_path, mzml_writer
    ):
        mzml = mzml_writer(
            tmp_path / "neg_and_empty.mzml",
            [
                {
                    "id_str": "scan=1",
                    "negative_scan": True,
                    "rt": "",
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "mz": [100.0],
                    "intensity": [100.0],
                },
                {
                    "id_str": "scan=2",
                    "omit_scan_list": True,
                    "precursor_mz": 600.0,
                    "charge": "",
                    "mz": [100.0],
                    "intensity": [100.0],
                },
            ],
        )
        spectra = list(mzml_io.get_spectra(mzml))
        assert [s.identifier for s in spectra] == [
            "neg_and_empty:scan:1",
            "neg_and_empty:scan:2",
        ]
        assert spectra[0].precursor_charge == -2
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)
        assert spectra[1].precursor_charge is None
        assert spectra[1].retention_time is not None
        assert math.isnan(spectra[1].retention_time)

    def test_malformed_and_zero_byte_xml_log_warning_instead_of_crashing(
        self, tmp_path, caplog
    ):
        for name, content in [
            ("corrupt.mzml", "<mzML><run><spectrumList><spectrum"),
            ("empty.mzml", ""),
        ]:
            path = tmp_path / name
            path.write_text(content, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="falcon"):
                spectra = list(mzml_io.get_spectra(path))
            assert spectra == []
            assert any(
                "Failed to read file" in r.message for r in caplog.records
            )

    def test_corrupt_base64_and_zlib_log_warning_instead_of_crashing(
        self, tmp_path, caplog, mzml_writer
    ):
        cases = [
            ("bad_b64.mzml", {"raw_mz_b64": "!!!not-base64!!!"}),
            (
                "bad_zlib.mzml",
                {
                    "mz_extra_cv": (
                        '<cvParam cvRef="MS" accession="MS:1000574" '
                        'name="zlib compression"/>'
                    ),
                    "raw_mz_b64": "AAAAAAAAWUA=",
                },
            ),
            # Valid base64 of 3 bytes (not a multiple of 8-byte f64).
            ("truncated_binary.mzml", {"raw_mz_b64": "YWJj"}),
        ]
        for filename, extra in cases:
            path = mzml_writer(
                tmp_path / filename,
                [
                    {
                        "id_str": "scan=1",
                        "precursor_mz": 500.0,
                        "omit_scan_list": True,
                        "mz": [100.0],
                        "intensity": [100.0],
                        **extra,
                    }
                ],
            )
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="falcon"):
                spectra = list(mzml_io.get_spectra(path))
            assert spectra == []
            assert any(
                "Failed to read file" in r.message for r in caplog.records
            )

    def test_pyteomics_error_logs_warning_and_missing_file_raises_os_error(
        self, tmp_path, caplog, monkeypatch
    ):
        with pytest.raises(FileNotFoundError):
            list(mzml_io.get_spectra(tmp_path / "does_not_exist.mzml"))

        def _raise_pyteomics(_source, *_args, **_kwargs):
            raise PyteomicsError("invalid mzML header")

        monkeypatch.setattr(pyteomics.mzml, "MzML", _raise_pyteomics)
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert list(mzml_io.get_spectra("bad.mzml")) == []
        assert any("Failed to read file" in r.message for r in caplog.records)

    def test_reads_from_open_file_object(self, data_dir):
        with (data_dir / "sample.mzml").open("rb") as fh:
            spectra = list(mzml_io.get_spectra(fh))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sample:scan:2"

    def test_skips_spectrum_with_empty_selected_ion_list(
        self, tmp_path, mzml_writer
    ):
        # Empty <selectedIonList count="0"/> triggers IndexError/KeyError.
        bad_mzml = mzml_writer(
            tmp_path / "empty_ion.mzml",
            [{"id_str": "scan=1", "empty_selected_ions": True}],
        )
        assert list(mzml_io.get_spectra(bad_mzml)) == []
