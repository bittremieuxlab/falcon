"""Unit tests for falcon.cluster.consensus."""

import math

import numpy as np
import pytest

from falcon.cluster import consensus
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.types import mz_bin_bounds


class TestIterClusterGroups:
    def test_noise_singletons_and_contiguous_clusters(self):
        clusters = np.array([-1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
        assert list(consensus.iter_cluster_groups(clusters)) == [
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 7),
        ]

    def test_non_contiguous_labels_and_all_noise(self):
        clusters = np.array([0, 0, 5, 10, 10], dtype=np.int32)
        assert list(consensus.iter_cluster_groups(clusters)) == [
            (0, 2),
            (2, 3),
            (3, 5),
        ]

        all_noise = np.array([-1, -1, -1], dtype=np.int32)
        assert list(consensus.iter_cluster_groups(all_noise)) == [
            (0, 1),
            (1, 2),
            (2, 3),
        ]

    def test_empty_and_single_element(self):
        assert (
            list(consensus.iter_cluster_groups(np.array([], np.int32))) == []
        )
        assert list(
            consensus.iter_cluster_groups(np.array([0], np.int32))
        ) == [(0, 1)]


class TestClusterMedoids:
    def test_three_member_cluster_selects_minimum_distance_sum_with_order_map(
        self, spec_factory
    ):
        specs = [
            spec_factory([100.0], [1.0], precursor_mz=500.0, charge=2),
            spec_factory([200.0], [1.0], precursor_mz=501.0, charge=2),
            spec_factory([300.0], [1.0], precursor_mz=502.0, charge=3),
        ]
        pdist = np.array([0.9, 0.1, 0.1], dtype=np.float32)
        labels = np.array([0, 0, 0], dtype=np.int32)
        rts = np.array([30.0, 10.0, 20.0], dtype=np.float32)
        order_map = np.array([2, 0, 1], dtype=np.int64)

        reps = consensus.get_representative_spectra(
            specs, labels, rts, order_map, MedoidConsensusConfig(), pdist=pdist
        )
        assert len(reps) == 1
        assert reps[0].precursor_mz == pytest.approx(502.0)
        assert reps[0].precursor_charge == 3
        np.testing.assert_allclose(reps[0].mz, [300.0])
        assert reps[0].retention_time == pytest.approx(30.0)
        assert reps[0].cluster_id == 0

    def test_singleton_and_pair_select_first_member(self, spec_factory):
        specs = [
            spec_factory([100.0], [1.0], precursor_mz=400.0),
            spec_factory([200.0], [1.0], precursor_mz=500.0),
            spec_factory([300.0], [1.0], precursor_mz=600.0),
        ]
        pdist = np.array([0.5, 0.5, 0.2], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int32)
        rts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        order_map = np.array([0, 1, 2], dtype=np.int64)

        reps = consensus.get_representative_spectra(
            specs, labels, rts, order_map, MedoidConsensusConfig(), pdist=pdist
        )
        assert [r.cluster_id for r in reps] == [0, 1]
        assert [float(r.precursor_mz) for r in reps] == [400.0, 500.0]
        assert [float(r.retention_time) for r in reps] == [1.0, 2.0]


class TestBinClusterPeaks:
    def test_occupancy_threshold_exact_and_multiple_peaks_per_spectrum(
        self, spec_factory
    ):
        spectra = []
        for i in range(10):
            mzs, ints = [], []
            if i < 7:
                mzs.append(100.05)
                ints.append(0.5)
                mzs.extend([300.01, 300.02])
                ints.extend([0.3, 0.4])
            if i < 6:
                mzs.append(200.05)
                ints.append(0.8)
            spectra.append(spec_factory(mzs, ints, normalize=False))

        n_bins, start_dim, _ = mz_bin_bounds(50.0, 400.0, 0.1)
        sorted_mz, sorted_int, starts, stops = consensus.bin_cluster_peaks(
            spectra, start_dim, n_bins, 0.1
        )
        assert len(starts) == 2
        assert stops[0] - starts[0] == 7
        assert stops[1] - starts[1] == 14
        np.testing.assert_allclose(sorted_mz[starts[0] : stops[0]], 100.05)
        np.testing.assert_allclose(sorted_int[starts[0] : stops[0]], 0.5)

    def test_out_of_range_and_empty_inputs_ignored(self, spec_factory):
        n_bins, start_dim, _ = mz_bin_bounds(100.0, 200.0, 1.0)
        s1 = spec_factory(
            [50.0, 150.0, 250.0], [1.0, 0.5, 1.0], normalize=False
        )
        s2 = spec_factory(
            [50.0, 150.2, 250.0], [1.0, 0.7, 1.0], normalize=False
        )

        _, _, starts, stops = consensus.bin_cluster_peaks(
            [s1, s2], start_dim, n_bins, 1.0
        )
        assert len(starts) == 1
        assert stops[0] - starts[0] == 2

        assert (
            len(consensus.bin_cluster_peaks([], start_dim, n_bins, 1.0)[2])
            == 0
        )
        empty_spec = spec_factory([], [], normalize=False)
        assert (
            len(
                consensus.bin_cluster_peaks(
                    [empty_spec, empty_spec], start_dim, n_bins, 1.0
                )[2]
            )
            == 0
        )
        oob = spec_factory([50.0, 250.0], [1.0, 1.0], normalize=False)
        assert (
            len(
                consensus.bin_cluster_peaks(
                    [oob, oob], start_dim, n_bins, 1.0
                )[2]
            )
            == 0
        )


class TestSigmaClipping:
    def test_rejects_high_outlier_and_stops_at_zero_std_or_size_two(self):
        intensities = np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32)
        mzs = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=np.float32)
        kept_i, kept_m = consensus.sigma_clip(intensities, mzs, 1.5, 1.5)
        np.testing.assert_allclose(kept_i, [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(kept_m, [100.0, 100.1, 100.2, 100.3])

        const_i = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        const_m = np.array([100.0, 100.1, 100.2], dtype=np.float32)
        out_i, out_m = consensus.sigma_clip(const_i, const_m, 0.5, 0.5)
        np.testing.assert_array_equal(out_i, const_i)
        np.testing.assert_array_equal(out_m, const_m)

        pair_i = np.array([1.0, 100.0], dtype=np.float32)
        pair_m = np.array([100.0, 100.1], dtype=np.float32)
        out_pi, out_pm = consensus.sigma_clip(pair_i, pair_m, 0.1, 0.1)
        np.testing.assert_array_equal(out_pi, pair_i)
        np.testing.assert_array_equal(out_pm, pair_m)

    def test_extreme_cutoff_drops_all_peaks_in_bin(self, spec_factory):
        s1 = spec_factory([100.0, 200.0], [1.0, 5.0], normalize=False)
        s2 = spec_factory([100.0, 200.0], [2.0, 5.0], normalize=False)
        s3 = spec_factory([100.0, 200.0], [10.0, 5.0], normalize=False)
        s4 = spec_factory([100.0, 200.0], [11.0, 5.0], normalize=False)
        reps = consensus.get_representative_spectra(
            [s1, s2, s3, s4],
            np.array([0, 0, 0, 0], dtype=np.int32),
            np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
            np.array([0, 1, 2, 3], dtype=np.int64),
            AverageConsensusConfig(
                outlier_cutoff_lower=0.01, outlier_cutoff_upper=0.01
            ),
            min_mz=50.0,
            max_mz=300.0,
            fragment_tol=0.5,
        )
        np.testing.assert_allclose(reps[0].mz, [200.0])
        np.testing.assert_allclose(reps[0].intensity, [5.0])


class TestRepresentativeDispatch:
    def test_average_multi_cluster_and_singleton_with_nan_rt_and_none_charge(
        self, spec_factory
    ):
        s0 = spec_factory(
            [100.0, 200.0], [0.4, 0.6], precursor_mz=500.0, normalize=False
        )
        s1 = spec_factory(
            [100.2, 200.2], [0.6, 0.8], precursor_mz=500.4, normalize=False
        )
        s2 = spec_factory(
            [150.0, 250.0],
            [0.5, 0.5],
            precursor_mz=600.0,
            charge=None,
            normalize=False,
        )
        reps = consensus.get_representative_spectra(
            [s0, s1, s2],
            np.array([0, 0, 1], dtype=np.int32),
            np.array([10.0, np.nan, 42.0], dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int64),
            AverageConsensusConfig(),
            min_mz=50.0,
            max_mz=300.0,
            fragment_tol=0.5,
        )
        assert len(reps) == 2
        assert reps[0].cluster_id == 0
        assert reps[0].precursor_mz == pytest.approx(500.2, abs=1e-4)
        assert reps[0].precursor_charge == 2
        assert math.isnan(reps[0].retention_time)
        np.testing.assert_allclose(reps[0].mz, [100.1, 200.1], atol=1e-4)
        np.testing.assert_allclose(reps[0].intensity, [0.5, 0.7], atol=1e-5)

        assert reps[1].cluster_id == 1
        assert math.isnan(reps[1].precursor_charge)
        assert reps[1].retention_time == pytest.approx(42.0)
        np.testing.assert_allclose(reps[1].mz, [150.0, 250.0])

    def test_average_bins_relative_to_start_dim_retains_boundary_peaks(
        self, spec_factory
    ):
        s1 = spec_factory([101.01, 200.0], [0.5, 0.8], normalize=False)
        s2 = spec_factory([101.02, 200.0], [0.5, 0.8], normalize=False)
        reps = consensus.get_representative_spectra(
            [s1, s2],
            np.array([0, 0], dtype=np.int32),
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            AverageConsensusConfig(),
            min_mz=101.03,
            max_mz=300.0,
            fragment_tol=0.05,
        )
        assert len(reps) == 1
        np.testing.assert_allclose(reps[0].mz, [101.015, 200.0], atol=1e-4)

    def test_average_with_no_surviving_bins_and_nan_charge(self, spec_factory):
        s1 = spec_factory([100.0], [1.0], charge=np.nan)
        s2 = spec_factory([200.0], [1.0], charge=np.nan)
        reps = consensus.get_representative_spectra(
            [s1, s2],
            np.array([0, 0], dtype=np.int32),
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            AverageConsensusConfig(),
            min_mz=50.0,
            max_mz=300.0,
            fragment_tol=0.5,
        )
        assert len(reps) == 1
        assert math.isnan(reps[0].precursor_charge)
        assert len(reps[0].mz) == 0
        assert len(reps[0].intensity) == 0

    def test_missing_required_kwargs_or_invalid_config_raises(
        self, mock_spectra
    ):
        specs = mock_spectra(2)
        labels = np.array([0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0], dtype=np.float32)
        order = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError, match="pdist is required"):
            consensus.get_representative_spectra(
                specs, labels, rts, order, MedoidConsensusConfig()
            )
        with pytest.raises(ValueError, match="min_mz"):
            consensus.get_representative_spectra(
                specs, labels, rts, order, AverageConsensusConfig()
            )
        with pytest.raises(ValueError, match="Unsupported consensus"):
            consensus.get_representative_spectra(
                specs,
                labels,
                rts,
                order,
                "invalid",  # type: ignore[arg-type]
            )
