"""Unit tests for falcon.preprocessing."""

import math

import numpy as np
import pytest
import spectrum_utils.spectrum as sus

from falcon.preprocessing import build_process_spectrum, process_spectrum


def _make_spectrum(
    mz,
    intensity,
    *,
    identifier="test:scan:1",
    precursor_mz=500.0,
    precursor_charge=2,
    retention_time=10.0,
):
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        np.asarray(mz, dtype=np.float32),
        np.asarray(intensity, dtype=np.float32),
        None,
        retention_time,
    )


class TestProcessSpectrum:
    def test_valid_spectrum_normalizes_and_preserves_metadata(self):
        spec = _make_spectrum(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [1e20, 2e20, 3e20, 4e20, 5e20],
            retention_time=42.5,
        )
        out = process_spectrum(spec, min_peaks=5, min_mz_range=250.0)
        assert out is not None
        assert out["identifier"] == "test:scan:1"
        assert out["precursor_charge"] == 2
        assert out["retention_time"] == pytest.approx(42.5)
        assert out["intensity"].dtype == np.float32
        assert np.linalg.norm(out["intensity"]) == pytest.approx(1.0, abs=1e-5)

    def test_quality_gates_and_inclusive_mz_bounds(self):
        # Exact boundary passes.
        spec = _make_spectrum([100.0, 200.0, 350.0], [10.0, 20.0, 30.0])
        assert (
            process_spectrum(
                spec,
                min_peaks=3,
                min_mz_range=250.0,
                min_mz=100.0,
                max_mz=350.0,
            )
            is not None
        )
        # Fewer peaks or narrower m/z span fails.
        too_narrow = _make_spectrum([100.0, 200.0, 349.0], [10.0, 20.0, 30.0])
        assert (
            process_spectrum(too_narrow, min_peaks=3, min_mz_range=250.0)
            is None
        )
        # m/z range restriction can trigger rejection.
        wide = _make_spectrum([50.0, 200.0, 300.0, 400.0, 1600.0], [10.0] * 5)
        assert (
            process_spectrum(
                wide,
                min_peaks=3,
                min_mz_range=250.0,
                min_mz=100.0,
                max_mz=1500.0,
            )
            is None
        )

    @pytest.mark.parametrize("charge", [2, -2, 0, None])
    def test_precursor_and_charge_reduced_removal(self, charge):
        # Precursor at 500 Da; for |z|=2, charge-reduced m/z is
        # 1000 - s*1.0072766 (998.99 for +, 1001.01 for -).
        reduced = 998.9927 if charge == 2 else 1001.0073
        mzs = [150.0, 300.0, 500.0, 700.0, reduced, 1100.0]
        spec = _make_spectrum(
            mzs,
            [10.0] * len(mzs),
            precursor_mz=500.0,
            precursor_charge=charge,
        )
        out = process_spectrum(
            spec, min_peaks=3, min_mz_range=200.0, remove_precursor_tol=1.5
        )
        assert out is not None
        assert all(abs(mz - 500.0) > 1.5 for mz in out["mz"])
        if charge in (2, -2):
            assert all(abs(mz - reduced) > 1.5 for mz in out["mz"])
        else:
            assert any(abs(mz - reduced) < 0.01 for mz in out["mz"])

    def test_intensity_filter_and_max_peaks_used(self):
        spec = _make_spectrum(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [100.0, 0.5, 50.0, 80.0, 60.0],
        )
        out = process_spectrum(
            spec,
            min_peaks=3,
            min_mz_range=200.0,
            min_intensity=0.01,
            max_peaks_used=3,
        )
        assert out is not None
        np.testing.assert_allclose(out["mz"], [100.0, 400.0, 500.0])

    @pytest.mark.parametrize("scaling", ["off", "root", "log", "rank"])
    def test_intensity_scaling_matches_reference(self, scaling):
        raw = np.array([10.0, 40.0, 90.0, 160.0, 250.0], dtype=np.float32)
        spec = _make_spectrum([100.0, 200.0, 300.0, 400.0, 500.0], raw)
        out = process_spectrum(
            spec,
            min_peaks=3,
            min_mz_range=200.0,
            max_peaks_used=5,
            scaling=scaling,
        )
        assert out is not None
        if scaling == "off":
            expected = raw
        elif scaling == "root":
            expected = np.sqrt(raw)
        elif scaling == "log":
            expected = np.log2(raw + 1.0)
        else:
            expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(out["intensity"], expected, rtol=1e-5)

    def test_rejects_invalid_metadata_and_non_finite_peaks(self):
        good_mz = [100.0, 200.0, 300.0, 400.0, 500.0]
        good_int = [10.0] * 5
        bad_cases = [
            _make_spectrum(good_mz, good_int, precursor_mz=0.0),
            _make_spectrum(good_mz, good_int, precursor_mz=float("nan")),
            _make_spectrum(good_mz, good_int, retention_time=float("inf")),
            _make_spectrum(good_mz, good_int, precursor_charge=200),
            _make_spectrum(good_mz, [0.0] * 5),
            _make_spectrum(good_mz, [-1.0, 10.0, 10.0, 10.0, 10.0]),
            _make_spectrum(good_mz, [10.0, np.nan, 10.0, 10.0, 10.0]),
            _make_spectrum([100.0, np.nan, 300.0, 400.0, 500.0], good_int),
            _make_spectrum([], []),
        ]
        for spec in bad_cases:
            assert (
                process_spectrum(spec, min_peaks=3, min_mz_range=100.0) is None
            )

        # None / NaN retention time is accepted.
        nan_rt = _make_spectrum(good_mz, good_int, retention_time=float("nan"))
        out = process_spectrum(nan_rt, min_peaks=3, min_mz_range=100.0)
        assert out is not None
        assert math.isnan(out["retention_time"])


class TestBuildProcessSpectrum:
    def test_aligns_mz_bounds_outward_to_fragment_tol_grid(
        self, config_factory
    ):
        cfg = config_factory(
            preprocessing={
                "min_peaks": 3,
                "min_mz_range": 100.0,
                "min_mz": 101.03,
                "max_mz": 500.02,
                "remove_precursor_tol": "none",
                "min_intensity": 0.0,
            },
            clustering={"fragment_tol": 0.05},
        )
        proc = build_process_spectrum(cfg)
        # 101.00 and 500.04 lie outside [101.03, 500.02] but inside aligned
        # outward bounds [101.00, 500.05].
        spec = _make_spectrum(
            [100.9, 101.00, 300.0, 500.04, 500.10], [10.0] * 5
        )
        out = proc(spec)
        assert out is not None
        np.testing.assert_allclose(out["mz"], [101.00, 300.0, 500.04])
