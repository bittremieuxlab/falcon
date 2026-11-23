"""Unit tests for falcon.preprocessing."""

import numpy as np
import pytest
import spectrum_utils.spectrum as sus

from falcon import preprocessing


def _make_msms(
    mz,
    intensity,
    precursor_mz=500.0,
    precursor_charge=2,
    identifier="test:scan:1",
    retention_time=120.0,
):
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        mz=np.asarray(mz, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32),
        retention_time=retention_time,
    )


class TestCheckSpectrumValid:
    def test_valid_and_exact_boundaries(self):
        mz = np.array([100.0, 150.0, 200.0, 250.0, 350.0], dtype=np.float32)
        assert preprocessing._check_spectrum_valid(
            mz, min_peaks=5, min_mz_range=250.0
        )
        assert not preprocessing._check_spectrum_valid(
            mz, min_peaks=6, min_mz_range=250.0
        )
        assert not preprocessing._check_spectrum_valid(
            mz, min_peaks=5, min_mz_range=250.1
        )

    def test_single_peak_with_zero_range_allowed(self):
        mz = np.array([300.0], dtype=np.float32)
        assert preprocessing._check_spectrum_valid(
            mz, min_peaks=1, min_mz_range=0.0
        )
        assert not preprocessing._check_spectrum_valid(
            mz, min_peaks=1, min_mz_range=0.1
        )

    @pytest.mark.parametrize(
        ("min_peaks", "min_mz_range"),
        [(1, 0.0), (0, 0.0), (0, -1e38)],
    )
    def test_empty_spectrum_always_invalid(self, min_peaks, min_mz_range):
        empty = np.array([], dtype=np.float32)
        assert not preprocessing._check_spectrum_valid(
            empty, min_peaks=min_peaks, min_mz_range=min_mz_range
        )


class TestNormIntensity:
    def test_exact_unit_norm_and_dtype(self):
        intensity = np.array([3.0, 4.0], dtype=np.float32)
        normed = preprocessing._norm_intensity(intensity)
        assert normed.dtype == np.float32
        np.testing.assert_allclose(normed, [0.6, 0.8], rtol=1e-6)
        assert np.linalg.norm(normed) == pytest.approx(1.0, abs=1e-6)

    def test_single_peak_normalizes_to_one(self):
        normed = preprocessing._norm_intensity(
            np.array([42.0], dtype=np.float32)
        )
        np.testing.assert_allclose(normed, [1.0], rtol=1e-6)


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
        np.testing.assert_allclose(result["intensity"], expected, rtol=1e-5)
        assert np.linalg.norm(result["intensity"]) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_rejects_after_mz_range_gate(self):
        few = _make_msms(mz=[200.0, 400.0], intensity=[100.0, 200.0])
        narrow = _make_msms(
            mz=[200.0, 200.5, 201.0, 201.5, 202.0],
            intensity=[100.0, 200.0, 50.0, 300.0, 150.0],
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

    def test_mz_bounds_filter_peaks_inclusively(self):
        spec = _make_msms(
            mz=[50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            intensity=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
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

    def test_missing_charge_removes_precursor_and_restores_none(self):
        spec = _make_msms(
            mz=[110.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            intensity=[100.0, 200.0, 50.0, 300.0, 999.0, 150.0],
            precursor_mz=500.0,
            precursor_charge=None,
        )
        result = preprocessing.process_spectrum(
            spec,
            min_peaks=5,
            min_mz_range=250.0,
            remove_precursor_tol=1.5,
        )
        assert result is not None
        assert result["precursor_charge"] is None
        assert 500.0 not in result["mz"]

        spec_reject = _make_msms(
            mz=[110.0, 499.8, 500.2, 600.0],
            intensity=[100.0, 200.0, 200.0, 150.0],
            precursor_mz=500.0,
            precursor_charge=None,
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
        assert spec_reject.precursor_charge is None

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

    def test_intensity_filter_rejection_and_all_zero_with_min_intensity(self):
        spec = _make_msms(
            mz=[100.0, 200.0, 300.0, 400.0, 500.0],
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

        all_zero = _make_msms(
            mz=[100.0, 200.0, 300.0, 400.0, 500.0],
            intensity=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert (
            preprocessing.process_spectrum(
                all_zero,
                min_peaks=5,
                min_mz_range=100.0,
                min_intensity=0.01,
            )
            is None
        )

    @pytest.mark.parametrize(
        "bad_intensities",
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [10.0, np.nan, 20.0, 30.0, 40.0],
            [10.0, np.inf, 20.0, 30.0, 40.0],
            [10.0, -np.inf, 20.0, 30.0, 40.0],
        ],
    )
    def test_non_finite_and_non_positive_intensities_return_none(
        self, bad_intensities
    ):
        spec = _make_msms(
            mz=[100.0, 200.0, 300.0, 400.0, 500.0],
            intensity=bad_intensities,
        )
        assert (
            preprocessing.process_spectrum(
                spec,
                min_peaks=5,
                min_mz_range=100.0,
                scaling="off",
            )
            is None
        )

    def test_max_peaks_used_without_min_intensity(self):
        spec = _make_msms(
            mz=[100.0, 200.0, 300.0, 400.0, 500.0],
            intensity=[10.0, 50.0, 20.0, 40.0, 30.0],
        )
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
                None,
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
            ),
        ],
    )
    def test_intensity_scaling_matches_reference(
        self, scaling, max_peaks_used, raw_expected
    ):
        spec = _make_msms(
            mz=[100.0, 200.0, 300.0, 400.0, 500.0],
            intensity=[1.0, 4.0, 9.0, 16.0, 25.0],
        )
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
