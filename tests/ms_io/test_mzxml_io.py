"""Unit tests for falcon.ms_io.mzxml_io."""

import base64
import logging
import math
import struct

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


def _read_mzxml_dicts(
    monkeypatch, fake_reader_factory, raw_dicts, source="run.mzxml"
):
    """Run ``mzxml_io.get_spectra`` against stubbed Pyteomics records."""
    monkeypatch.setattr(
        pyteomics.mzxml, "MzXML", fake_reader_factory(raw_dicts)
    )
    return list(mzxml_io.get_spectra(source))


class TestMzxmlParseSpectrum:
    def test_missing_rt_and_charge(self, monkeypatch, fake_reader_factory):
        raw = _mzxml_dict()
        orig_id = raw["id"]
        spectra = _read_mzxml_dicts(monkeypatch, fake_reader_factory, [raw])
        assert len(spectra) == 1
        spec = spectra[0]
        assert spec.identifier == "run:scan:7"
        assert raw["id"] == orig_id
        assert spec.retention_time is not None
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == pytest.approx(400.0)
        assert spec.precursor_charge is None

    def test_rt_and_charge_present(self, monkeypatch, fake_reader_factory):
        spectra = _read_mzxml_dicts(
            monkeypatch,
            fake_reader_factory,
            [_mzxml_dict(extra_precursor={"precursorCharge": 2}, rt=6.0)],
        )
        assert len(spectra) == 1
        assert spectra[0].retention_time == pytest.approx(6.0)
        assert spectra[0].precursor_charge == 2

    def test_multiple_precursors_uses_first(
        self, monkeypatch, fake_reader_factory
    ):
        spectra = _read_mzxml_dicts(
            monkeypatch,
            fake_reader_factory,
            [
                _mzxml_dict(
                    extra_precursor={"precursorCharge": 2},
                    more_precursors=[
                        {"precursorMz": 550.0, "precursorCharge": 3}
                    ],
                )
            ],
        )
        assert len(spectra) == 1
        assert spectra[0].precursor_mz == pytest.approx(400.0)
        assert spectra[0].precursor_charge == 2

    @pytest.mark.parametrize(
        ("polarity", "raw_charge", "expected"),
        [
            ("-", 2, -2),
            ("-", -2, -2),
            ("-", None, None),
            ("+", 2, 2),
        ],
    )
    def test_negative_polarity_negates_charge_magnitude(
        self, monkeypatch, fake_reader_factory, polarity, raw_charge, expected
    ):
        extra = (
            {"precursorCharge": raw_charge} if raw_charge is not None else None
        )
        raw = {**_mzxml_dict(extra_precursor=extra), "polarity": polarity}
        spectra = _read_mzxml_dicts(monkeypatch, fake_reader_factory, [raw])
        assert [s.precursor_charge for s in spectra] == [expected]


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
        assert s.mz.dtype == np.float32
        assert s.intensity.dtype == np.float32
        np.testing.assert_allclose(s.mz, [160.0])
        np.testing.assert_allclose(s.intensity, [10.0])

    def test_get_spectra_does_not_mutate_spectrum_dict_id_and_passes_options(
        self, monkeypatch, fake_reader_factory
    ):
        raw = _mzxml_dict(extra_precursor={"precursorCharge": 2})
        orig_id = raw["id"]
        received_kwargs = []

        monkeypatch.setattr(
            pyteomics.mzxml,
            "MzXML",
            fake_reader_factory([raw], kwargs_log=received_kwargs),
        )
        spectra = list(mzxml_io.get_spectra("run.mzxml"))
        assert [s.identifier for s in spectra] == ["run:scan:7"]
        assert raw["id"] == orig_id
        assert received_kwargs == [{"use_index": False, "dtype": np.float32}]

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

    def test_malformed_and_zero_byte_xml_log_warning_instead_of_crashing(
        self, tmp_path, caplog
    ):
        for name, content in [
            ("corrupt.mzxml", "<mzXML><msRun><scan num='1'"),
            ("empty.mzxml", ""),
        ]:
            path = tmp_path / name
            path.write_text(content, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="falcon"):
                spectra = list(mzxml_io.get_spectra(path))
            assert spectra == []
            assert any(
                "Failed to read file" in r.message for r in caplog.records
            )

    def test_corrupt_base64_and_zlib_log_warning_instead_of_crashing(
        self, tmp_path, caplog, mzxml_writer
    ):
        odd_peaks_b64 = base64.b64encode(
            struct.pack(">3f", 100.0, 50.0, 200.0)
        ).decode("ascii")
        for filename, comp, payload in [
            ("bad_b64.mzxml", "none", "!!!not-base64!!!"),
            ("bad_zlib.mzxml", "zlib", "AAAAAAAAWUA="),
            ("odd_peaks.mzxml", "none", odd_peaks_b64),
        ]:
            path = mzxml_writer(
                tmp_path / filename,
                [
                    {
                        "scan": 1,
                        "precursor_mz": 400.0,
                        "compression": comp,
                        "raw_peaks_b64": payload,
                    }
                ],
            )
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="falcon"):
                spectra = list(mzxml_io.get_spectra(path))
            assert spectra == []
            assert any(
                "Failed to read file" in r.message for r in caplog.records
            )

    def test_skips_non_numeric_rt_and_non_finite_precursor_mz(
        self, tmp_path, monkeypatch, fake_reader_factory
    ):
        raws = [
            _mzxml_dict(rt="not_a_number"),
            {
                **_mzxml_dict(),
                "id": "8",
                "precursorMz": [{"precursorMz": float("inf")}],
            },
            {**_mzxml_dict(rt=1.5), "id": "9"},
        ]

        spectra = _read_mzxml_dicts(
            monkeypatch,
            fake_reader_factory,
            raws,
            source=tmp_path / "mixed.mzxml",
        )
        assert [s.identifier for s in spectra] == ["mixed:scan:9"]

    def test_parses_negative_polarity_end2end(self, tmp_path, mzxml_writer):
        path = mzxml_writer(
            tmp_path / "neg.mzxml",
            [
                {
                    "scan": 3,
                    "polarity": "-",
                    "precursor_mz": 420.0,
                    "charge": 2,
                    "mz": [150.0],
                    "intensity": [25.0],
                }
            ],
        )
        spectra = list(mzxml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == ["neg:scan:3"]
        assert spectra[0].precursor_mz == pytest.approx(420.0)
        assert spectra[0].precursor_charge == -2

    @pytest.mark.parametrize(
        ("filename", "spec_overrides"),
        [
            ("bad_num.mzxml", {"raw_attrs": 'num="abc" msLevel="2"'}),
            ("missing_mslevel.mzxml", {"scan": 2, "ms_level": None}),
        ],
    )
    def test_non_integer_num_and_missing_mslevel_warn_without_crashing(
        self, tmp_path, caplog, mzxml_writer, filename, spec_overrides
    ):
        path = mzxml_writer(
            tmp_path / filename,
            [
                {
                    "precursor_mz": 400.0,
                    "charge": 2,
                    "mz": [150.0],
                    "intensity": [25.0],
                    **spec_overrides,
                }
            ],
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert list(mzxml_io.get_spectra(path)) == []
        assert any("Failed to read file" in r.message for r in caplog.records)

    def test_skips_non_integer_id_and_missing_mslevel_per_spectrum(
        self, monkeypatch, fake_reader_factory
    ):
        no_level = _mzxml_dict()
        del no_level["msLevel"]
        raws = [
            {**_mzxml_dict(), "id": "abc"},
            no_level,
            {**_mzxml_dict(), "id": "4"},
        ]
        spectra = _read_mzxml_dicts(
            monkeypatch, fake_reader_factory, raws, source="mixed_ids.mzxml"
        )
        assert [s.identifier for s in spectra] == ["mixed_ids:scan:4"]

    def test_missing_file_propagates_os_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(mzxml_io.get_spectra(tmp_path / "missing.mzxml"))

    def test_empty_precursor_charge_and_attributeless_precursor_retained(
        self, tmp_path, mzxml_writer
    ):
        # Empty precursorCharge="" yields None in Pyteomics; retain with
        # precursor_charge=None (matching mzml_io).
        path = mzxml_writer(
            tmp_path / "empty_charge.mzxml",
            [
                {
                    "scan": 1,
                    "precursor_mz": 400.0,
                    "empty_charge": True,
                    "mz": [100.0],
                    "intensity": [10.0],
                },
                {
                    "scan": 5,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "mz": [120.0],
                    "intensity": [20.0],
                },
            ],
        )
        spectra = list(mzxml_io.get_spectra(path))
        assert [s.identifier for s in spectra] == [
            "empty_charge:scan:1",
            "empty_charge:scan:5",
        ]
        assert spectra[0].precursor_charge is None
        assert spectra[1].precursor_charge == 2

        # Minimal <precursorMz> without XML attributes collapses to a
        # bare string in Pyteomics; retain with precursor_charge=None.
        bare_path = mzxml_writer(
            tmp_path / "bare_prec.mzxml",
            [
                {
                    "scan": 7,
                    "precursor_mz": 450.25,
                    "mz": [150.0],
                    "intensity": [25.0],
                }
            ],
        )
        bare_spectra = list(mzxml_io.get_spectra(bare_path))
        assert [s.identifier for s in bare_spectra] == ["bare_prec:scan:7"]
        assert bare_spectra[0].precursor_mz == pytest.approx(450.25)
        assert bare_spectra[0].precursor_charge is None

    def test_skips_none_or_bare_string_precursor_mz(
        self, monkeypatch, fake_reader_factory
    ):
        raws = [
            {
                **_mzxml_dict(),
                "id": "1",
                "precursorMz": [{"precursorMz": None}],
            },
            {
                **_mzxml_dict(),
                "id": "2",
                "precursorMz": "450.25",
            },
            {**_mzxml_dict(), "id": "3"},
        ]
        spectra = _read_mzxml_dicts(
            monkeypatch, fake_reader_factory, raws, source="bad_prec.mzxml"
        )
        assert [s.identifier for s in spectra] == ["bad_prec:scan:3"]

    def test_blank_retention_time_yields_nan_rt(
        self, monkeypatch, fake_reader_factory
    ):
        raws = [
            {**_mzxml_dict(rt=""), "id": "10"},
            {**_mzxml_dict(rt="   "), "id": "11"},
        ]
        spectra = _read_mzxml_dicts(
            monkeypatch, fake_reader_factory, raws, source="blank_rt.mzxml"
        )
        assert [s.identifier for s in spectra] == [
            "blank_rt:scan:10",
            "blank_rt:scan:11",
        ]
        assert all(
            s.retention_time is not None and math.isnan(s.retention_time)
            for s in spectra
        )
