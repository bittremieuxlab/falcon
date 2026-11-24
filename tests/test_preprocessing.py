"""Unit tests for falcon.preprocessing."""

import math
import warnings
from typing import Any, cast

import numpy as np
import pytest
import spectrum_utils.spectrum as sus

from falcon import preprocessing

_DEFAULT_MZ = (100.0, 200.0, 300.0, 400.0, 500.0)
_DEFAULT_INTENSITY = (10.0, 20.0, 30.0, 40.0, 50.0)


def _make_msms(
    mz=_DEFAULT_MZ,
    intensity=_DEFAULT_INTENSITY,
    precursor_mz=500.0,
    precursor_charge=2,
    identifier="test:scan:1",
    retention_time=120.0,
    annotation=None,
):
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        annotation=annotation,
        retention_time=retention_time,
    )


class TestProcessSpectrum:
    def test_valid_spectrum_preserves_metadata_and_normalizes(self):
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 400.0, 450.0, 480.0, 490.0],
            intensity=[100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 10.0],
        )
        raw_intensities = spec.intensity.copy()
        expected = raw_intensities / np.linalg.norm(raw_intensities)

        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=250.0,
            min_mz=100.0,
            max_mz=1500.0,
        )
        assert result is not None
        assert result["identifier"] == "test:scan:1"
        assert result["precursor_mz"] == pytest.approx(500.0)
        assert result["precursor_charge"] == 2
        assert result["retention_time"] == pytest.approx(120.0)
        assert result["intensity"].dtype == np.float32
        np.testing.assert_allclose(result["intensity"], expected, rtol=1e-5)
        assert np.linalg.norm(result["intensity"]) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_exact_peak_count_span_and_single_peak_boundaries(self):
        mz = [100.0, 150.0, 200.0, 250.0, 350.0]
        intensities = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert (
            preprocessing.process_spectrum(
                _make_msms(mz, intensities), min_peaks=5, min_mz_range=250.0
            )
            is not None
        )
        assert (
            preprocessing.process_spectrum(
                _make_msms(mz, intensities), min_peaks=6, min_mz_range=250.0
            )
            is None
        )
        assert (
            preprocessing.process_spectrum(
                _make_msms(mz, intensities), min_peaks=5, min_mz_range=250.1
            )
            is None
        )

        single = preprocessing.process_spectrum(
            _make_msms([300.0], [42.0]), min_peaks=1, min_mz_range=0.0
        )
        assert single is not None
        np.testing.assert_allclose(single["intensity"], [1.0], rtol=1e-6)
        assert (
            preprocessing.process_spectrum(
                _make_msms([300.0], [42.0]), min_peaks=1, min_mz_range=0.1
            )
            is None
        )

        # Wide float32 m/z span computes difference in float64 without
        # overflow warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            wide = preprocessing.process_spectrum(
                _make_msms([-3e38, 3e38], [3.0, 4.0]),
                min_peaks=2,
                min_mz_range=100.0,
            )
        assert wide is not None
        np.testing.assert_allclose(wide["intensity"], [0.6, 0.8], rtol=1e-6)

    @pytest.mark.parametrize("scale", [1e20, 1e-22])
    def test_extreme_intensity_magnitudes_avoid_float32_overflow_underflow(
        self, scale
    ):
        spec = _make_msms(
            mz=[100.0, 400.0],
            intensity=[3.0 * scale, 4.0 * scale],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = preprocessing.process_spectrum(
                spec, min_peaks=2, min_mz_range=100.0
            )
        assert result is not None
        normed = result["intensity"]
        assert normed.dtype == np.float32
        assert np.isfinite(normed).all()
        np.testing.assert_allclose(normed, [0.6, 0.8], rtol=1e-5)
        assert np.linalg.norm(normed) == pytest.approx(1.0, abs=1e-6)

    def test_rejects_after_mz_range_gate(self):
        few = _make_msms(mz=[200.0, 400.0], intensity=[100.0, 200.0])
        narrow = _make_msms(
            mz=[200.0, 200.5, 201.0, 201.5, 202.0],
            intensity=[100.0, 200.0, 50.0, 300.0, 150.0],
        )
        trimmed_few = _make_msms(
            mz=[50.0, 60.0, 200.0, 400.0, 1600.0],
            intensity=[100.0, 100.0, 100.0, 200.0, 100.0],
        )
        assert (
            preprocessing.process_spectrum(
                few,
                min_peaks=5,
                min_mz_range=10.0,
                min_mz=100.0,
                max_mz=1500.0,
            )
            is None
        )
        assert (
            preprocessing.process_spectrum(
                narrow,
                min_peaks=3,
                min_mz_range=250.0,
                min_mz=100.0,
                max_mz=1500.0,
            )
            is None
        )
        assert (
            preprocessing.process_spectrum(
                trimmed_few,
                min_peaks=4,
                min_mz_range=250.0,
                min_mz=100.0,
                max_mz=1500.0,
            )
            is None
        )

    def test_mz_bounds_filter_peaks_inclusively_and_one_sided(self):
        spec = _make_msms(
            mz=[50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            intensity=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            annotation=["a", "b", "c", "d", "e", "f", "g"],
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=300.0,
            min_mz=100.0,
            max_mz=500.0,
        )
        assert result is not None
        np.testing.assert_allclose(
            result["mz"], [100.0, 200.0, 300.0, 400.0, 500.0]
        )
        assert list(spec.annotation) == ["b", "c", "d", "e", "f"]

        res_min_only = preprocessing.process_spectrum(
            _make_msms(
                mz=[50.0, 100.0, 300.0, 600.0],
                intensity=[10.0, 20.0, 30.0, 40.0],
            ),
            min_peaks=3,
            min_mz_range=200.0,
            min_mz=100.0,
            max_mz=None,
        )
        assert res_min_only is not None
        np.testing.assert_allclose(res_min_only["mz"], [100.0, 300.0, 600.0])

        res_max_only = preprocessing.process_spectrum(
            _make_msms(
                mz=[50.0, 100.0, 300.0, 600.0],
                intensity=[10.0, 20.0, 30.0, 40.0],
            ),
            min_peaks=3,
            min_mz_range=200.0,
            min_mz=None,
            max_mz=400.0,
        )
        assert res_max_only is not None
        np.testing.assert_allclose(res_max_only["mz"], [50.0, 100.0, 300.0])

    def test_omitted_mz_bounds_and_python_float_scalars(self, monkeypatch):
        spec = _make_msms(
            precursor_mz=np.float32(500.25),
            retention_time=np.float32(12.5),
        )

        def _forbid_set_mz_range(*_args, **_kwargs):
            raise AssertionError("set_mz_range must not be called")

        monkeypatch.setattr(spec, "set_mz_range", _forbid_set_mz_range)
        res = preprocessing.process_spectrum(
            spec, min_peaks=5, min_mz_range=250.0, min_mz=None, max_mz=None
        )
        assert res is not None
        assert type(res["precursor_mz"]) is float
        assert res["precursor_mz"] == pytest.approx(500.25)
        assert type(res["retention_time"]) is float
        assert res["retention_time"] == pytest.approx(12.5)

    def test_low_mz_positive_charge_removes_charge_reduced_precursor(self):
        # precursor_mz=0.5 < _PROTON_MASS with charge=2 gives charge-1
        # target 2*(0.5 - H+) + H+ = 1.0 - H+ = -0.0072766 Da, so peak
        # at 0.0 Da is within 0.1 Da and must be removed.
        spec = _make_msms(
            mz=[0.0, 0.5, 100.0, 200.0, 300.0, 400.0],
            intensity=[99.0, 99.0, 10.0, 20.0, 30.0, 40.0],
            precursor_mz=0.5,
            precursor_charge=2,
        )
        res = preprocessing.process_spectrum(
            spec,
            min_peaks=4,
            min_mz_range=200.0,
            remove_precursor_tol=0.1,
        )
        assert res is not None
        np.testing.assert_allclose(res["mz"], [100.0, 200.0, 300.0, 400.0])

    def test_precursor_removal_strips_only_peaks_within_tolerance(self):
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 498.0, 499.0, 500.5, 600.0],
            intensity=[100.0, 200.0, 300.0, 150.0, 400.0, 500.0, 250.0],
            precursor_mz=500.0,
            precursor_charge=2,
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=4,
            min_mz_range=250.0,
            min_mz=100.0,
            max_mz=1500.0,
            remove_precursor_tol=1.5,
        )
        assert result is not None
        np.testing.assert_allclose(
            result["mz"], [110.0, 200.0, 300.0, 498.0, 600.0]
        )

    def test_precursor_removal_can_trigger_quality_gate_rejection(self):
        spec = _make_msms(
            mz=[110.0, 250.0, 499.5, 500.5, 600.0],
            intensity=[100.0, 200.0, 300.0, 300.0, 250.0],
            precursor_mz=500.0,
            precursor_charge=1,
        )
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=4,
                min_mz_range=250.0,
                remove_precursor_tol=1.5,
            )
            is None
        )

    @pytest.mark.parametrize("charge", [None, 0])
    def test_missing_or_zero_charge_removes_precursor_and_preserves_charge(
        self, charge
    ):
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            intensity=[100.0, 200.0, 50.0, 300.0, 999.0, 150.0],
            precursor_mz=500.0,
            precursor_charge=charge,
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=250.0,
            remove_precursor_tol=1.5,
        )
        assert result is not None
        assert result["precursor_charge"] == charge
        assert 500.0 not in result["mz"]

        spec_reject = _make_msms(
            mz=[110.0, 499.8, 500.2, 600.0],
            intensity=[100.0, 200.0, 200.0, 150.0],
            precursor_mz=500.0,
            precursor_charge=charge,
        )
        assert (
            preprocessing.process_spectrum(
                spec_reject,
                min_peaks=4,
                min_mz_range=250.0,
                remove_precursor_tol=1.5,
            )
            is None
        )
        assert spec_reject.precursor_charge == charge

    @pytest.mark.parametrize("charge", [2, -2, 0, None])
    def test_float32_precursor_mz_removed_without_scalar_overflow(
        self, charge
    ):
        raw_pmz = np.float32(500.25)
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 400.0, 500.25, 600.0],
            intensity=[100.0, 120.0, 140.0, 160.0, 500.0, 180.0],
            precursor_mz=raw_pmz,
            precursor_charge=charge,
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=250.0,
            remove_precursor_tol=1.5,
        )
        assert result is not None
        assert 500.25 not in result["mz"]
        assert type(spec.precursor_mz) is type(raw_pmz)
        assert spec.precursor_mz == raw_pmz

    def test_intensity_filter_and_max_peaks_used(self):
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 400.0, 450.0, 480.0],
            intensity=[1000.0, 5.0, 800.0, 900.0, 700.0, 600.0],
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=3,
            min_mz_range=50.0,
            min_intensity=0.01,
            max_peaks_used=4,
        )
        assert result is not None
        np.testing.assert_allclose(result["mz"], [110.0, 300.0, 400.0, 450.0])

    def test_intensity_filter_rejection(self):
        spec = _make_msms(
            intensity=[1000.0, 1.0, 1.0, 1.0, 900.0],
        )
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=4,
                min_mz_range=100.0,
                min_intensity=0.1,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("bad_intensities", "min_intensity", "scaling"),
        [
            ([0.0, 0.0, 0.0, 0.0, 0.0], None, "off"),
            ([0.0, 0.0, 0.0, 0.0, 0.0], 0.01, "rank"),
            ([-1.0, -2.0, -3.0, -4.0, -5.0], None, "off"),
            ([-1.0, 10.0, 20.0, 30.0, 40.0], 0.01, "root"),
            ([10.0, np.nan, 20.0, 30.0, 40.0], 0.01, "log"),
            ([10.0, np.inf, 20.0, 30.0, 40.0], None, "off"),
            ([10.0, -np.inf, 20.0, 30.0, 40.0], 0.01, "off"),
        ],
    )
    def test_invalid_or_non_positive_intensities_return_none(
        self, bad_intensities, min_intensity, scaling
    ):
        spec = _make_msms(intensity=bad_intensities)
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=5,
                min_mz_range=100.0,
                min_intensity=min_intensity,
                max_peaks_used=5,
                scaling=scaling,
            )
            is None
        )

    def test_max_peaks_used_without_min_intensity(self):
        spec = _make_msms(intensity=[10.0, 50.0, 20.0, 40.0, 30.0])
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=3,
            min_mz_range=100.0,
            min_intensity=None,
            max_peaks_used=3,
        )
        assert result is not None
        np.testing.assert_allclose(result["mz"], [200.0, 400.0, 500.0])

    @pytest.mark.parametrize(
        ("scaling", "max_peaks_used", "raw_expected"),
        [
            (
                "off",
                None,
                np.array([1.0, 4.0, 9.0, 16.0, 25.0], np.float32),
            ),
            (
                None,
                None,
                np.array([1.0, 4.0, 9.0, 16.0, 25.0], np.float32),
            ),
            (
                "root",
                None,
                np.sqrt(np.array([1.0, 4.0, 9.0, 16.0, 25.0], np.float32)),
            ),
            (
                "log",
                None,
                np.log2(
                    np.array([1.0, 4.0, 9.0, 16.0, 25.0], np.float32) + 1.0
                ),
            ),
            (
                "rank",
                5,
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
            ),
            (
                "rank",
                10,
                np.array([6.0, 7.0, 8.0, 9.0, 10.0], np.float32),
            ),
            (
                "rank",
                None,
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
            ),
        ],
    )
    def test_intensity_scaling_matches_reference(
        self, scaling, max_peaks_used, raw_expected
    ):
        spec = _make_msms(intensity=[1.0, 4.0, 9.0, 16.0, 25.0])
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=250.0,
            max_peaks_used=max_peaks_used,
            scaling=scaling,
        )
        assert result is not None
        expected = raw_expected / np.linalg.norm(raw_expected)
        np.testing.assert_allclose(result["intensity"], expected, rtol=1e-5)

    def test_unknown_scaling_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported intensity scaling"):
            preprocessing.process_spectrum(
                _make_msms(),
                min_peaks=1,
                min_mz_range=0.0,
                scaling=cast("Any", "invalid"),
            )

    @pytest.mark.parametrize(
        "bad_precursor_mz",
        [0.0, -500.0, np.nan, np.inf, -np.inf, 1e-46, 1e39],
    )
    def test_rejects_invalid_or_out_of_float32_precursor_mz(
        self, bad_precursor_mz
    ):
        spec = _make_msms(precursor_mz=bad_precursor_mz)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert (
                preprocessing.process_spectrum(
                    spec, min_peaks=1, min_mz_range=0.0
                )
                is None
            )

    @pytest.mark.parametrize(
        "boundary_pmz",
        [
            float(np.nextafter(np.float32(0.0), np.float32(1.0))),
            float(np.finfo(np.float32).max),
        ],
    )
    def test_accepts_float32_boundary_precursor_mz(self, boundary_pmz):
        spec = _make_msms(precursor_mz=boundary_pmz)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = preprocessing.process_spectrum(
                spec, min_peaks=5, min_mz_range=250.0
            )
        assert res is not None
        assert res["precursor_mz"] == boundary_pmz

    @pytest.mark.parametrize(
        "bad_rt", [float("inf"), float("-inf"), 1e40, -1e40]
    )
    def test_rejects_infinite_or_float32_overflow_retention_time(self, bad_rt):
        spec = _make_msms(retention_time=bad_rt)
        assert (
            preprocessing.process_spectrum(spec, min_peaks=1, min_mz_range=0.0)
            is None
        )

    @pytest.mark.parametrize(
        "valid_rt",
        [None, float("nan"), 0.0, -5.0, float(np.finfo(np.float32).max)],
    )
    def test_accepts_none_nan_and_float32_retention_time(self, valid_rt):
        spec = _make_msms(retention_time=valid_rt)
        res = preprocessing.process_spectrum(
            spec, min_peaks=5, min_mz_range=250.0
        )
        assert res is not None
        if valid_rt is None:
            assert res["retention_time"] is None
        elif math.isnan(valid_rt):
            assert res["retention_time"] is not None
            assert math.isnan(res["retention_time"])
        else:
            assert res["retention_time"] == pytest.approx(valid_rt)

    @pytest.mark.parametrize(
        "bad_charge",
        [
            -129,
            128,
            1000,
            True,
            False,
            2.0,
            2.5,
            np.bool_(True),
            np.float32(2),
        ],
    )
    def test_rejects_invalid_or_out_of_int8_precursor_charge(self, bad_charge):
        spec = _make_msms(precursor_charge=bad_charge)
        assert (
            preprocessing.process_spectrum(spec, min_peaks=1, min_mz_range=0.0)
            is None
        )

    @pytest.mark.parametrize(
        "boundary_charge", [-128, 127, np.int8(-128), np.int32(2)]
    )
    def test_accepts_int8_and_numpy_integer_precursor_charge(
        self, boundary_charge
    ):
        spec = _make_msms(precursor_charge=boundary_charge)
        remove_tol = 1.5 if int(boundary_charge) == -128 else None
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = preprocessing.process_spectrum(
                spec,
                min_peaks=4,
                min_mz_range=250.0,
                remove_precursor_tol=remove_tol,
            )
        assert res is not None
        assert type(res["precursor_charge"]) is int
        assert res["precursor_charge"] == int(boundary_charge)
        if remove_tol is not None:
            assert 500.0 not in res["mz"]

    @pytest.mark.parametrize(
        ("min_peaks", "min_mz_range"),
        [(1, 0.0), (0, 0.0), (0, -1e38)],
    )
    def test_rejects_zero_peak_spectrum_for_any_thresholds(
        self, min_peaks, min_mz_range
    ):
        empty = _make_msms(mz=[], intensity=[])
        assert (
            preprocessing.process_spectrum(
                empty,
                min_peaks=min_peaks,
                min_mz_range=min_mz_range,
                min_mz=None,
                max_mz=1500.0,
            )
            is None
        )

    def test_rejects_non_finite_mz_before_mz_range_filter(self):
        for bad_mz in (np.nan, np.inf, -np.inf):
            spec = _make_msms(
                mz=[100.0, 200.0, 300.0, 400.0, bad_mz],
                intensity=[10.0, 20.0, 30.0, 40.0, 50.0],
            )
            assert (
                preprocessing.process_spectrum(
                    spec,
                    min_peaks=1,
                    min_mz_range=0.0,
                    min_mz=100.0,
                    max_mz=1500.0,
                )
                is None
            )

        assert (
            preprocessing.process_spectrum(
                _make_msms(),
                min_peaks=1,
                min_mz_range=0.0,
                min_mz=600.0,
                max_mz=200.0,
            )
            is None
        )

    @pytest.mark.parametrize("min_mz", [None, 100.0])
    def test_all_peaks_above_max_mz_returns_none(self, min_mz):
        spec = _make_msms(
            mz=[2000.0, 3000.0],
            intensity=[100.0, 200.0],
        )
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=1,
                min_mz_range=0.0,
                min_mz=min_mz,
                max_mz=1500.0,
            )
            is None
        )

    def test_negative_charge_removes_precursor_and_charge_reduced_peaks(self):
        # For charge -2 at m/z 500.0:
        # c=2 target = 500.0; c=1 target = 2*(500 + H+) - H+ = 1001.01.
        spec = _make_msms(
            mz=[110.0, 250.0, 498.0, 500.5, 700.0, 1001.2, 1004.0],
            intensity=[100.0, 200.0, 150.0, 500.0, 300.0, 400.0, 250.0],
            precursor_mz=500.0,
            precursor_charge=-2,
            annotation=["a", "b", "c", "p2", "d", "p1", "e"],
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=4,
            min_mz_range=250.0,
            remove_precursor_tol=1.5,
        )
        assert result is not None
        assert result["precursor_charge"] == -2
        np.testing.assert_allclose(
            result["mz"], [110.0, 250.0, 498.0, 700.0, 1004.0]
        )
        assert list(spec.annotation) == ["a", "b", "c", "d", "e"]

        spec_unannotated = _make_msms(
            mz=[110.0, 250.0, 499.5, 500.5, 600.0],
            intensity=[100.0, 200.0, 300.0, 300.0, 250.0],
            precursor_mz=500.0,
            precursor_charge=-1,
        )
        assert (
            preprocessing.process_spectrum(
                spec_unannotated,
                min_peaks=4,
                min_mz_range=250.0,
                remove_precursor_tol=1.5,
            )
            is None
        )

    @pytest.mark.parametrize(
        "scaled_bad",
        [
            np.zeros(5, dtype=np.float32),
            np.array([1.0, np.nan, 2.0, 3.0, 4.0], dtype=np.float32),
        ],
    )
    def test_post_scaling_guard_rejects_invalid_scaled_intensities(
        self, monkeypatch, scaled_bad
    ):
        spec = _make_msms()

        def _bad_scale(*_args, **_kwargs):
            spec.intensity = scaled_bad
            return spec

        monkeypatch.setattr(spec, "scale_intensity", _bad_scale)
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=5,
                min_mz_range=100.0,
                scaling="log",
            )
            is None
        )


class TestBuildProcessSpectrum:
    def test_aligns_mz_bounds_outward_to_fragment_tol_grid(
        self, config_factory
    ):
        # min_mz=101.03, max_mz=1500.0, fragment_tol=0.05 -> aligned to
        # [101.0, 1500.05].
        cfg = config_factory(
            preprocessing={
                "min_mz": 101.03,
                "max_mz": 1500.0,
                "min_peaks": 3,
                "min_mz_range": 100.0,
                "min_intensity": 0.0,
                "scaling": "off",
            },
            clustering={"fragment_tol": 0.05},
        )
        proc = preprocessing.build_process_spectrum(cfg)

        spec = _make_msms(
            mz=[100.99, 101.01, 300.0, 700.0, 1500.04, 1500.10],
            intensity=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            precursor_mz=600.0,
        )
        result = proc(spec)
        assert result is not None
        np.testing.assert_allclose(
            result["mz"], [101.01, 300.0, 700.0, 1500.04]
        )

    def test_applies_configured_scaling_and_filters(self, config_factory):
        raw_mz = [150.0, 250.0, 350.0, 450.0, 550.0]
        raw_int = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)

        cfg_off = config_factory(
            preprocessing={
                "scaling": "off",
                "min_intensity": 0.0,
                "remove_precursor_tol": None,
            }
        )
        cfg_root = config_factory(
            preprocessing={
                "scaling": "root",
                "min_intensity": 0.0,
                "remove_precursor_tol": None,
            }
        )

        res_off = preprocessing.build_process_spectrum(cfg_off)(
            _make_msms(mz=raw_mz, intensity=raw_int)
        )
        res_root = preprocessing.build_process_spectrum(cfg_root)(
            _make_msms(mz=raw_mz, intensity=raw_int)
        )
        assert res_off is not None
        assert res_root is not None
        np.testing.assert_allclose(
            res_off["intensity"],
            raw_int / np.linalg.norm(raw_int),
            rtol=1e-5,
        )
        sqrt_int = np.sqrt(raw_int)
        np.testing.assert_allclose(
            res_root["intensity"],
            sqrt_int / np.linalg.norm(sqrt_int),
            rtol=1e-5,
        )
