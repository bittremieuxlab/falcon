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

    @pytest.mark.parametrize("scan_list", [None, {}, {"scan": []}])
    def test_absent_or_empty_scan_list_defaults_rt_to_nan(self, scan_list):
        raw = _mzml_dict(ion_extra={"charge state": 2})
        if scan_list is None:
            del raw["scanList"]
        else:
            raw["scanList"] = scan_list
        spec = mzml_io._parse_spectrum(raw, "run:scan:5")
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)

    @pytest.mark.parametrize(
        "blank_rt", [None, "", "   ", unitstr("", "second")]
    )
    def test_empty_or_none_scan_start_time_defaults_to_nan(self, blank_rt):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(scan={"scan start time": blank_rt}),
            "run:scan:5",
        )
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)

    def test_rt_present(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(
                scan={"scan start time": 6.0}, ion_extra={"charge state": 2}
            ),
            "run:scan:5",
        )
        assert spec.retention_time == pytest.approx(6.0)

    def test_possible_charge_state_fallback_scalar_and_sequences(self):
        spec_scalar = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"possible charge state": 3}),
            "run:scan:5",
        )
        assert spec_scalar.precursor_charge == 3

        spec_list = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"possible charge state": [2, 3]}),
            "run:scan:5",
        )
        assert spec_list.precursor_charge == 2

        spec_empty = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"possible charge state": []}),
            "run:scan:5",
        )
        assert spec_empty.precursor_charge is None

    def test_charge_state_sequence_and_empty_fallback(self):
        spec_seq = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"charge state": (4, 5)}),
            "run:scan:5",
        )
        assert spec_seq.precursor_charge == 4

        spec_empty_falls_back = mzml_io._parse_spectrum(
            _mzml_dict(
                ion_extra={"charge state": [], "possible charge state": [3, 4]}
            ),
            "run:scan:5",
        )
        assert spec_empty_falls_back.precursor_charge == 3

        spec_both_empty = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"charge state": ()}),
            "run:scan:5",
        )
        assert spec_both_empty.precursor_charge is None

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

    @pytest.mark.parametrize("blank_charge", [None, "", "   ", [""], [None]])
    def test_empty_charge_cv_param_returns_none_and_allows_fallback(
        self, blank_charge
    ):
        spec_none = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"charge state": blank_charge}),
            "run:scan:5",
        )
        assert spec_none.precursor_charge is None

        spec_fallback = mzml_io._parse_spectrum(
            _mzml_dict(
                ion_extra={
                    "charge state": blank_charge,
                    "possible charge state": 3,
                }
            ),
            "run:scan:5",
        )
        assert spec_fallback.precursor_charge == 3

    @pytest.mark.parametrize(
        ("raw_charge", "expected"),
        [(2, -2), (-2, -2), (None, None)],
    )
    def test_negative_scan_negates_charge_magnitude(
        self, raw_charge, expected
    ):
        ion_extra = (
            {"charge state": raw_charge} if raw_charge is not None else {}
        )
        raw = {**_mzml_dict(ion_extra=ion_extra), "negative scan": ""}
        spec = mzml_io._parse_spectrum(raw, "run:scan:5")
        assert spec.precursor_charge == expected

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

    def test_rejects_non_numeric_rt_and_non_finite_precursor_mz(self):
        with pytest.raises(ValueError, match="could not convert"):
            mzml_io._parse_spectrum(
                _mzml_dict(scan={"scan start time": unitstr("bad", "second")}),
                "run:scan:5",
            )
        bad_mz_dict = _mzml_dict()
        bad_mz_dict["precursorList"]["precursor"][0]["selectedIonList"][
            "selectedIon"
        ][0]["selected ion m/z"] = float("nan")
        with pytest.raises(ValueError, match="Precursor m/z must be finite"):
            mzml_io._parse_spectrum(bad_mz_dict, "run:scan:5")


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
        self, tmp_path
    ):
        mzml = tmp_path / "multi_charge.mzml"
        mzml.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
            ' <run id="run1">\n'
            '  <spectrumList count="1">\n'
            '   <spectrum index="0" id="scan=1" defaultArrayLength="1">\n'
            '    <cvParam cvRef="MS" accession="MS:1000511" '
            'name="ms level" value="2"/>\n'
            '    <precursorList count="1"><precursor>'
            '<selectedIonList count="1"><selectedIon>'
            '<cvParam cvRef="MS" accession="MS:1000744" '
            'name="selected ion m/z" value="500.0"/>'
            '<cvParam cvRef="MS" accession="MS:1000633" '
            'name="possible charge state" value="2"/>'
            '<cvParam cvRef="MS" accession="MS:1000633" '
            'name="possible charge state" value="3"/>'
            "</selectedIon></selectedIonList>"
            "</precursor></precursorList>\n"
            '    <binaryDataArrayList count="2">\n'
            '     <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            '     <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000515" '
            'name="intensity array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            "    </binaryDataArrayList>\n"
            "   </spectrum>\n"
            "  </spectrumList>\n"
            " </run>\n"
            "</mzML>\n",
            encoding="utf-8",
        )
        spectra = list(mzml_io.get_spectra(mzml))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge == 2
        assert spectra[0].retention_time is not None
        assert math.isnan(spectra[0].retention_time)

    def test_parses_negative_scan_and_empty_cv_params_end2end(self, tmp_path):
        mzml = tmp_path / "neg_and_empty.mzml"
        mzml.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
            ' <run id="run1"><spectrumList count="2">\n'
            '  <spectrum index="0" id="scan=1" defaultArrayLength="1">\n'
            '   <cvParam cvRef="MS" accession="MS:1000511" '
            'name="ms level" value="2"/>\n'
            '   <cvParam cvRef="MS" accession="MS:1000129" '
            'name="negative scan"/>\n'
            '   <scanList count="1"><scan>'
            '<cvParam cvRef="MS" accession="MS:1000016" '
            'name="scan start time" value=""/>'
            "</scan></scanList>\n"
            '   <precursorList count="1"><precursor>'
            '<selectedIonList count="1"><selectedIon>'
            '<cvParam cvRef="MS" accession="MS:1000744" '
            'name="selected ion m/z" value="500.0"/>'
            '<cvParam cvRef="MS" accession="MS:1000041" '
            'name="charge state" value="2"/>'
            "</selectedIon></selectedIonList></precursor></precursorList>\n"
            '   <binaryDataArrayList count="2">\n'
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000515" '
            'name="intensity array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            "   </binaryDataArrayList>\n"
            "  </spectrum>\n"
            '  <spectrum index="1" id="scan=2" defaultArrayLength="1">\n'
            '   <cvParam cvRef="MS" accession="MS:1000511" '
            'name="ms level" value="2"/>\n'
            '   <precursorList count="1"><precursor>'
            '<selectedIonList count="1"><selectedIon>'
            '<cvParam cvRef="MS" accession="MS:1000744" '
            'name="selected ion m/z" value="600.0"/>'
            '<cvParam cvRef="MS" accession="MS:1000041" '
            'name="charge state" value=""/>'
            "</selectedIon></selectedIonList></precursor></precursorList>\n"
            '   <binaryDataArrayList count="2">\n'
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000515" '
            'name="intensity array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            "   </binaryDataArrayList>\n"
            "  </spectrum>\n"
            " </spectrumList></run></mzML>\n",
            encoding="utf-8",
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
        self, tmp_path, caplog
    ):
        template = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n'
            ' <run id="run1"><spectrumList count="1">\n'
            '  <spectrum index="0" id="scan=1" defaultArrayLength="1">\n'
            '   <cvParam cvRef="MS" accession="MS:1000511" '
            'name="ms level" value="2"/>\n'
            '   <precursorList count="1"><precursor>'
            '<selectedIonList count="1"><selectedIon>'
            '<cvParam cvRef="MS" accession="MS:1000744" '
            'name="selected ion m/z" value="500.0"/>'
            "</selectedIon></selectedIonList></precursor></precursorList>\n"
            '   <binaryDataArrayList count="2">\n'
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            "{extra_cv}"
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
            "<binary>{payload}</binary></binaryDataArray>\n"
            '    <binaryDataArray encodedLength="12">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000515" '
            'name="intensity array"/>'
            "<binary>AAAAAAAAWUA=</binary></binaryDataArray>\n"
            "   </binaryDataArrayList>\n"
            "  </spectrum>\n"
            " </spectrumList></run></mzML>\n"
        )
        cases = [
            ("bad_b64.mzml", "", "!!!not-base64!!!"),
            (
                "bad_zlib.mzml",
                '<cvParam cvRef="MS" accession="MS:1000574" '
                'name="zlib compression"/>',
                "AAAAAAAAWUA=",
            ),
            # Valid base64 of 3 bytes (not a multiple of 8-byte f64).
            ("truncated_binary.mzml", "", "YWJj"),
        ]
        for filename, extra_cv, payload in cases:
            path = tmp_path / filename
            path.write_text(
                template.format(extra_cv=extra_cv, payload=payload),
                encoding="utf-8",
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
