"""Unit tests for falcon.cluster.consensus (medoid and average)."""

import math

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from falcon.cluster import consensus
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.types import mz_bin_bounds


class TestClusterMedoids:
    def test_randomized_medoids_match_dense_squareform_reference(
        self, spec_factory
    ):
        rng = np.random.default_rng(77)
        for _ in range(10):
            n = int(rng.integers(5, 25))
            specs = [
                spec_factory(
                    [100.0 + i], [1.0], precursor_mz=300.0 + i, charge=2
                )
                for i in range(n)
            ]
            pdist = rng.uniform(0.0, 1.0, size=n * (n - 1) // 2).astype(
                np.float32
            )
            dense = squareform(pdist.astype(np.float64))
            order_map = rng.permutation(n).astype(np.int64)
            rts = rng.uniform(1.0, 500.0, size=n).astype(np.float32)
            raw_labels = np.sort(
                rng.integers(0, max(2, n // 3), size=n, dtype=np.int32)
            )

            reps = consensus.get_cluster_medoids(
                specs, raw_labels, rts, order_map, pdist
            )
            unique_labels = np.unique(raw_labels)
            assert [int(r.cluster_id) for r in reps] == unique_labels.tolist()

            for rep, lbl in zip(reps, unique_labels, strict=True):
                positions = np.flatnonzero(raw_labels == lbl)
                members = order_map[positions]
                if len(members) < 3:
                    expected_pos = int(positions[0])
                else:
                    sub = dense[np.ix_(members, members)]
                    expected_pos = int(positions[int(np.argmin(sub.sum(1)))])
                expected_spec = specs[int(order_map[expected_pos])]
                assert float(rep.precursor_mz) == pytest.approx(
                    float(expected_spec.precursor_mz)
                )
                assert float(rep.retention_time) == pytest.approx(
                    float(rts[expected_pos])
                )

    def test_empty_singletons_and_noise_medoids(self, spec_factory):
        specs = [
            spec_factory([100.0], [1.0], precursor_mz=500.0),
            spec_factory([200.0], [1.0], precursor_mz=501.0),
            spec_factory([300.0], [1.0], precursor_mz=502.0),
        ]
        reps = consensus.get_cluster_medoids(
            specs,
            np.array([-1, -1, 0], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0, 1, 2], dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
        assert [int(r.cluster_id) for r in reps] == [-1, -1, 0]
        assert (
            consensus.get_cluster_medoids(
                [],
                np.empty(0, np.int32),
                np.empty(0, np.float32),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
            )
            == []
        )


class TestBinningSigmaClipAndAverages:
    def test_bin_cluster_peaks_70_percent_occupancy(self, spec_factory):
        # 10 spectra: bin @ 100 in 7 (kept), bin @ 200 in 6 (dropped).
        specs = [
            spec_factory(
                [100.0, 200.0] if i < 6 else ([100.0] if i == 6 else [300.0]),
                [1.0, 1.0] if i < 6 else [1.0],
                normalize=False,
            )
            for i in range(10)
        ]
        n_bins, start_dim, _ = mz_bin_bounds(50.0, 400.0, 1.0)
        sorted_mz, _, starts, stops = consensus.bin_cluster_peaks(
            specs, start_dim, n_bins, 1.0
        )
        assert len(starts) == 1
        np.testing.assert_allclose(
            sorted_mz[starts[0] : stops[0]], [100.0] * 7
        )

    def test_sigma_clip_outliers_asymmetric_and_stop_conditions(self):
        mzs = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=np.float32)
        kept_i, _ = consensus.sigma_clip(
            np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32),
            mzs,
            1.5,
            1.5,
        )
        np.testing.assert_allclose(kept_i, [1.0, 1.0, 1.0, 1.0])

        # Asymmetric bounds clip only the tighter side.
        sym = np.array([1.0, 10.0, 10.0, 10.0, 19.0], dtype=np.float32)
        low_i, _ = consensus.sigma_clip(sym, mzs, 1.0, 3.0)
        high_i, _ = consensus.sigma_clip(sym, mzs, 3.0, 1.0)
        np.testing.assert_allclose(low_i, [10.0, 10.0, 10.0, 19.0])
        np.testing.assert_allclose(high_i, [1.0, 10.0, 10.0, 10.0])

        # Pair and zero-std arrays stop immediately.
        pair = np.array([1.0, 100.0], dtype=np.float32)
        out_pair, _ = consensus.sigma_clip(pair, mzs[:2], 0.1, 0.1)
        np.testing.assert_array_equal(out_pair, pair)

    def test_large_cluster_avoids_float32_summation_drift(self, spec_factory):
        n = 2000
        peak_mz = np.float32(1000.01)
        peak_int = np.float32(0.75)
        spectra = [
            spec_factory(
                [float(peak_mz)],
                [float(peak_int)],
                precursor_mz=float(peak_mz),
                normalize=False,
            )
            for _ in range(n)
        ]
        reps = consensus.get_cluster_averages(
            spectra,
            np.zeros(n, dtype=np.int32),
            np.full(n, 500.01, dtype=np.float32),
            np.arange(n, dtype=np.int64),
            min_mz=100.0,
            max_mz=1500.0,
            fragment_tol=0.05,
        )
        assert len(reps) == 1
        assert abs(float(reps[0].mz[0]) - float(peak_mz)) < 1e-4
        assert abs(float(reps[0].intensity[0]) - float(peak_int)) < 1e-5
        assert abs(float(reps[0].retention_time) - 500.01) < 1e-4

    def test_average_multi_cluster_nan_rt_none_charge_and_dispatch(
        self, spec_factory
    ):
        specs = [
            spec_factory(
                [100.0, 200.0], [0.4, 0.6], precursor_mz=500.0, normalize=False
            ),
            spec_factory(
                [100.2, 200.2], [0.6, 0.8], precursor_mz=500.4, normalize=False
            ),
            spec_factory(
                [300.0],
                [1.0],
                precursor_mz=600.0,
                charge=None,
                normalize=False,
            ),
        ]
        labels = np.array([0, 0, 1], dtype=np.int32)
        rts = np.array([10.0, np.nan, 30.0], dtype=np.float32)
        order = np.array([0, 1, 2], dtype=np.int64)

        avg_reps = consensus.get_representative_spectra(
            specs,
            labels,
            rts,
            order,
            AverageConsensusConfig(),
            min_mz=50.0,
            max_mz=400.0,
            fragment_tol=0.5,
        )
        assert [int(r.cluster_id) for r in avg_reps] == [0, 1]
        assert avg_reps[0].precursor_mz == pytest.approx(500.2, abs=1e-4)
        assert math.isnan(float(avg_reps[0].retention_time))
        np.testing.assert_allclose(avg_reps[0].mz, [100.1, 200.1], atol=1e-4)
        np.testing.assert_allclose(
            avg_reps[0].intensity, [0.5, 0.7], atol=1e-4
        )
        assert math.isnan(float(avg_reps[1].precursor_charge))

        med_reps = consensus.get_representative_spectra(
            specs,
            labels,
            rts,
            order,
            MedoidConsensusConfig(),
            pdist=np.array([0.1, 0.9, 0.9], dtype=np.float32),
        )
        assert [int(r.cluster_id) for r in med_reps] == [0, 1]
