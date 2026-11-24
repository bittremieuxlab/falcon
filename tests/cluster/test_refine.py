"""Unit tests for falcon.cluster.refine (1D linkage and postprocessing)."""

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist as scipy_pdist

from falcon.cluster import refine


class TestCutLinkage:
    def test_returns_zero_based_int32_labels_across_thresholds(self):
        values = np.array([100.0, 101.0, 105.0, 107.0, 110.0])
        z = refine.linkage_1d(values, "Da")

        all_singletons = refine.cut_linkage(z, 0.5)
        assert all_singletons.dtype == np.int32
        assert len(np.unique(all_singletons)) == 5
        assert set(all_singletons.tolist()) == {0, 1, 2, 3, 4}

        mid = refine.cut_linkage(z, 2.0)
        assert mid.dtype == np.int32
        assert mid[0] == mid[1]
        assert mid[2] == mid[3]
        assert len({mid[0], mid[2], mid[4]}) == 3
        assert mid.min() == 0

        merged = refine.cut_linkage(z, 10.0)
        assert merged.dtype == np.int32
        np.testing.assert_array_equal(merged, np.zeros(5, dtype=np.int32))

    def test_matches_scipy_fcluster_zero_based_boundary(self):
        values = np.array([10.0, 11.0, 20.0, 22.0])
        z = refine.linkage_1d(values, "Da")
        for threshold in (0.99, 1.0, 1.99, 2.0, 12.0):
            expected = (sch.fcluster(z, threshold, "distance") - 1).astype(
                np.int32
            )
            actual = refine.cut_linkage(z, threshold)
            assert actual.dtype == np.int32
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "bad_threshold", [-0.1, float("nan"), float("inf"), float("-inf")]
    )
    def test_rejects_negative_or_non_finite_threshold(self, bad_threshold):
        z = refine.linkage_1d(np.array([10.0, 11.0, 20.0]), "Da")
        with pytest.raises(ValueError, match="finite and non-negative"):
            refine.cut_linkage(z, bad_threshold)


class TestLinkage1D:
    def test_da_exact_and_matches_scipy_complete_linkage(self):
        values = np.array([100.0, 101.0, 105.0, 107.0, 110.0])
        z_falcon = refine.linkage_1d(values, "Da")
        z_scipy = sch.linkage(scipy_pdist(values[:, None]), method="complete")
        np.testing.assert_allclose(z_falcon[:, 2], z_scipy[:, 2])
        np.testing.assert_array_equal(z_falcon[:, 3], z_scipy[:, 3])
        np.testing.assert_array_equal(
            z_falcon,
            np.array(
                [[0, 1, 1, 2], [2, 3, 2, 2], [6, 4, 5, 3], [5, 7, 10, 5]]
            ),
        )

    def test_unsorted_input_preserves_original_indices(self):
        values = np.array([110.0, 100.0, 105.0, 101.0])
        z = refine.linkage_1d(values, "Da")
        assert sorted(z[0, :2]) == [1.0, 3.0]
        assert z[0, 2] == pytest.approx(1.0)
        assert z[0, 3] == 2

    def test_ppm_scales_by_lower_cluster_min(self):
        values = np.array([100.0, 100.001, 200.0])
        z_ppm = refine.linkage_1d(values, "ppm")
        assert z_ppm[0, 2] == pytest.approx(10.0, rel=1e-4)
        assert z_ppm[1, 2] == pytest.approx(1_000_000.0, rel=1e-4)

    def test_rt_none_mode_identical_and_negative_values(self):
        z_dup = refine.linkage_1d(np.array([10.0, 10.0, 10.0]), None)
        np.testing.assert_allclose(z_dup[:, 2], [0.0, 0.0])

        z_neg = refine.linkage_1d(np.array([-5.0, -4.5, 10.0]), None)
        assert z_neg[0, 2] == pytest.approx(0.5)
        assert z_neg[1, 2] == pytest.approx(15.0)

    def test_randomized_da_and_ppm_match_scipy_complete_linkage(self):
        rng = np.random.default_rng(7)
        # Unsorted, negative, and duplicate Da arrays match SciPy exactly.
        for _ in range(15):
            n = int(rng.integers(2, 25))
            vals = rng.uniform(-50.0, 500.0, size=n)
            if n >= 4:
                vals[1] = vals[0]
            z_falcon = refine.linkage_1d(vals, "Da")
            z_scipy = sch.linkage(
                scipy_pdist(vals[:, None]), method="complete"
            )
            np.testing.assert_allclose(
                z_falcon[:, 2], z_scipy[:, 2], rtol=1e-12, atol=1e-12
            )

        # Positive unsorted ppm arrays match independent O(n^2) ppm
        # complete linkage: |x_i - x_j| / min(x_i, x_j) * 1e6.
        for _ in range(15):
            n = int(rng.integers(2, 20))
            vals = rng.uniform(100.0, 1500.0, size=n)
            condensed_ppm = np.array(
                [
                    abs(vals[i] - vals[j]) / min(vals[i], vals[j]) * 1e6
                    for i in range(n)
                    for j in range(i + 1, n)
                ],
                dtype=np.float64,
            )
            z_falcon = refine.linkage_1d(vals, "ppm")
            z_scipy = sch.linkage(condensed_ppm, method="complete")
            np.testing.assert_allclose(
                z_falcon[:, 2], z_scipy[:, 2], rtol=1e-10, atol=1e-10
            )

    def test_validates_inputs_and_handles_non_finite_without_corruption(self):
        with pytest.raises(ValueError, match="1D"):
            refine.linkage_1d(np.zeros((2, 2)), "Da")
        for short in (np.array([]), np.array([100.0])):
            with pytest.raises(ValueError, match="at least 2"):
                refine.linkage_1d(short, "Da")
        with pytest.raises(ValueError, match="tol_mode"):
            refine.linkage_1d(np.array([100.0, 101.0]), "PPM")
        with pytest.raises(ValueError, match="strictly positive"):
            refine.linkage_1d(np.array([0.0, 100.0]), "ppm")
        for bad_ppm in (
            np.array([100.0, np.nan]),
            np.array([100.0, np.inf]),
            np.array([-np.inf, 100.0]),
        ):
            with pytest.raises(ValueError, match="finite strictly positive"):
                refine.linkage_1d(bad_ppm, "ppm")

        # NaN/inf inputs must never trigger negative-index wrap-around.
        z_nan = refine.linkage_1d(np.array([100.0, np.nan, 200.0]), "Da")
        assert z_nan.shape == (2, 4)
        assert z_nan[0, 2] == pytest.approx(100.0)
        assert set(z_nan[0, :2].tolist()) == {0.0, 2.0}

        z_inf = refine.linkage_1d(np.array([100.0, np.inf]), "Da")
        assert z_inf.shape == (1, 4)
        assert np.isinf(z_inf[0, 2])


class TestIterClusterGroups:
    def test_noise_singletons_and_contiguous_clusters(self):
        clusters = np.array([-1, -1, 0, 0, -1, 1, 1, 1], dtype=np.int32)
        assert list(refine.iter_cluster_groups(clusters)) == [
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 5),
            (5, 8),
        ]

    def test_non_contiguous_labels_empty_and_single_element(self):
        clusters = np.array([0, 0, 5, 10, 10], dtype=np.int32)
        assert list(refine.iter_cluster_groups(clusters)) == [
            (0, 2),
            (2, 3),
            (3, 5),
        ]
        assert list(refine.iter_cluster_groups(np.array([], np.int32))) == []
        assert list(refine.iter_cluster_groups(np.array([0], np.int32))) == [
            (0, 1)
        ]


class TestPostprocessCluster:
    def test_homogeneous_cluster_keeps_single_label(self):
        labels = np.array([99, 99, 99, 99], dtype=np.int32)
        mzs = np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        n = refine.postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", None, min_samples=2, start_label=7
        )
        assert n == 1
        assert np.all(labels == 7)

    def test_splits_distant_mz_da_and_ppm_and_respects_min_samples(self):
        labels = np.zeros(4, dtype=np.int32)
        mzs = np.array([100.0, 100.1, 200.0, 200.1], dtype=np.float64)
        rts = np.zeros(4, dtype=np.float64)
        n = refine.postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", None, min_samples=2, start_label=0
        )
        assert n == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

        labels_ms3 = np.zeros(4, dtype=np.int32)
        n_ms3 = refine.postprocess_cluster(
            labels_ms3,
            mzs,
            rts,
            0.5,
            "Da",
            None,
            min_samples=3,
            start_label=10,
        )
        assert n_ms3 == 4
        assert sorted(labels_ms3.tolist()) == [10, 11, 12, 13]

        labels_ppm = np.zeros(4, dtype=np.int32)
        mzs_ppm = np.array([100.0, 100.001, 100.01, 100.011], dtype=np.float64)
        n_ppm = refine.postprocess_cluster(
            labels_ppm, mzs_ppm, rts, 20.0, "ppm", None, 2, 0
        )
        assert n_ppm == 2
        assert labels_ppm[0] == labels_ppm[1]
        assert labels_ppm[2] == labels_ppm[3]
        assert labels_ppm[0] != labels_ppm[2]

    def test_all_singletons_and_too_few_items(self):
        labels = np.zeros(3, dtype=np.int32)
        n = refine.postprocess_cluster(
            labels,
            np.array([100.0, 200.0, 300.0]),
            np.array([10.0, 20.0, 30.0]),
            0.5,
            "Da",
            None,
            min_samples=2,
            start_label=3,
        )
        assert n == 3
        assert sorted(labels.tolist()) == [3, 4, 5]

        one = np.array([0], dtype=np.int32)
        assert (
            refine.postprocess_cluster(
                one,
                np.array([100.0]),
                np.array([10.0]),
                0.5,
                "Da",
                None,
                min_samples=2,
                start_label=5,
            )
            == 1
        )
        assert one[0] == 5

    def test_rt_grid_per_mz_group_and_interleaved_rt_pairs(self):
        # Interleaved RTs across distinct precursor m/z groups must not
        # shatter valid (m/z, RT) pairs into singletons.
        inter_mzs = np.array([500.0, 500.0, 501.0, 501.0], dtype=np.float64)
        inter_rts = np.array([10.0, 12.0, 11.0, 13.0], dtype=np.float64)
        inter_labels = np.zeros(4, dtype=np.int32)
        n_inter = refine.postprocess_cluster(
            inter_labels, inter_mzs, inter_rts, 0.5, "Da", 2.0, 2, 0
        )
        assert n_inter == 2
        assert inter_labels[0] == inter_labels[1]
        assert inter_labels[2] == inter_labels[3]
        assert inter_labels[0] != inter_labels[2]

        # Grid across multiple m/z and RT bands produces exact clusters.
        mzs, rts = [], []
        for base_mz in (100.0, 200.0, 300.0, 400.0):
            for base_rt in (10.0, 50.0, 90.0):
                mzs.extend([base_mz, base_mz + 0.05])
                rts.extend([base_rt, base_rt + 0.1])
        mzs_arr = np.array(mzs, dtype=np.float64)
        rts_arr = np.array(rts, dtype=np.float64)
        labels = np.zeros(len(mzs_arr), dtype=np.int32)

        n = refine.postprocess_cluster(
            labels, mzs_arr, rts_arr, 0.5, "Da", 1.0, 2, 0
        )
        assert n == 12
        assert len(np.unique(labels)) == 12
        for label in np.unique(labels):
            mask = labels == label
            assert mzs_arr[mask].max() - mzs_arr[mask].min() <= 0.5
            assert rts_arr[mask].max() - rts_arr[mask].min() <= 1.0

    def test_rt_skips_undersized_mz_groups_and_handles_mixed_group_sizes(self):
        # All m/z groups below min_samples skip RT linkage cleanly.
        labels_all_small = np.zeros(4, dtype=np.int32)
        n_small = refine.postprocess_cluster(
            labels_all_small,
            np.array([100.0, 100.1, 200.0, 200.1]),
            np.array([10.0, 50.0, 10.0, 50.0]),
            0.5,
            "Da",
            1.0,
            min_samples=3,
            start_label=2,
        )
        assert n_small == 4
        assert sorted(labels_all_small.tolist()) == [2, 3, 4, 5]

        # Mixed m/z groups: one singleton m/z (skips RT linkage) plus
        # one 4-spectrum m/z group splitting into two RT pairs.
        labels_mixed = np.zeros(5, dtype=np.int32)
        n_mixed = refine.postprocess_cluster(
            labels_mixed,
            np.array([100.0, 200.0, 200.05, 200.1, 200.15]),
            np.array([10.0, 10.0, 10.5, 50.0, 50.5]),
            0.5,
            "Da",
            2.0,
            min_samples=2,
            start_label=0,
        )
        assert n_mixed == 3
        assert labels_mixed[1] == labels_mixed[2]
        assert labels_mixed[3] == labels_mixed[4]
        assert len({labels_mixed[0], labels_mixed[1], labels_mixed[3]}) == 3

        # min_samples=1 still splits multi-spectrum m/z groups by RT.
        labels_ms1 = np.zeros(3, dtype=np.int32)
        n_ms1 = refine.postprocess_cluster(
            labels_ms1,
            np.array([100.0, 100.0, 200.0]),
            np.array([10.0, 50.0, 10.0]),
            0.5,
            "Da",
            1.0,
            min_samples=1,
            start_label=0,
        )
        assert n_ms1 == 3
        assert set(labels_ms1.tolist()) == {0, 1, 2}

    def test_rt_single_valid_with_nan_stays_separate_without_rt_linkage(self):
        labels = np.zeros(3, dtype=np.int32)
        mzs = np.array([100.0, 100.05, 100.1], dtype=np.float64)
        rts = np.array([10.0, np.nan, np.nan], dtype=np.float64)
        n = refine.postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", 1.0, min_samples=2, start_label=4
        )
        assert n == 2
        assert labels[1] == labels[2]
        assert labels[0] != labels[1]
        assert set(labels.tolist()) == {4, 5}

    def test_rt_with_mixed_nan_and_all_nan(self):
        mzs = np.array(
            [100.0, 100.05, 100.1, 100.15, 200.0, 200.05, 200.1, 200.15]
        )
        rts = np.array(
            [10.0, 10.1, np.nan, np.nan, 10.0, 10.1, np.nan, np.nan]
        )
        labels = np.zeros(len(mzs), dtype=np.int32)
        n = refine.postprocess_cluster(labels, mzs, rts, 0.5, "Da", 1.0, 2, 0)
        assert n == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[6] == labels[7]
        assert len({labels[0], labels[2], labels[4], labels[6]}) == 4

        labels_all_nan = np.zeros(4, dtype=np.int32)
        n_all_nan = refine.postprocess_cluster(
            labels_all_nan,
            np.array([100.0, 100.1, 200.0, 200.1]),
            np.full(4, np.nan),
            0.5,
            "Da",
            1.0,
            2,
            0,
        )
        assert n_all_nan == 2
        assert labels_all_nan[0] == labels_all_nan[1]
        assert labels_all_nan[2] == labels_all_nan[3]
        assert labels_all_nan[0] != labels_all_nan[2]

    def test_min_samples_one_single_point_and_zero_tolerances(self):
        one = np.array([0], dtype=np.int32)
        assert (
            refine.postprocess_cluster(
                one,
                np.array([100.0]),
                np.array([10.0]),
                0.0,
                "Da",
                0.0,
                min_samples=1,
                start_label=9,
            )
            == 1
        )
        assert one[0] == 9

        labels = np.zeros(3, dtype=np.int32)
        n = refine.postprocess_cluster(
            labels,
            np.array([100.0, 100.0, 100.001]),
            np.array([5.0, 5.0, 5.0]),
            0.0,
            "Da",
            0.0,
            min_samples=2,
            start_label=0,
        )
        assert n == 2
        assert labels[0] == labels[1]
        assert labels[0] != labels[2]

    def test_rt_with_infinite_values_separates_from_finite_without_polluting(
        self,
    ):
        labels = np.zeros(5, dtype=np.int32)
        mzs = np.array([100.0, 100.02, 100.04, 100.06, 100.08])
        rts = np.array([10.0, 10.1, np.inf, -np.inf, np.nan])
        n = refine.postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", 1.0, min_samples=2, start_label=0
        )
        assert n == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3] == labels[4]
        assert labels[0] != labels[2]

        labels_all_inf = np.zeros(4, dtype=np.int32)
        n_inf = refine.postprocess_cluster(
            labels_all_inf,
            np.array([100.0, 100.1, 200.0, 200.1]),
            np.array([np.inf, -np.inf, np.inf, np.nan]),
            0.5,
            "Da",
            1.0,
            min_samples=2,
            start_label=0,
        )
        assert n_inf == 2
        assert labels_all_inf[0] == labels_all_inf[1]
        assert labels_all_inf[2] == labels_all_inf[3]
        assert labels_all_inf[0] != labels_all_inf[2]

    @pytest.mark.parametrize("mode", ["Da", "ppm"])
    @pytest.mark.parametrize("rt_tol", [None, 2.0])
    def test_randomized_span_rt_and_label_contiguity_invariants(
        self, mode, rt_tol
    ):
        rng = np.random.default_rng(19)
        tol = 0.5 if mode == "Da" else 40.0
        for trial in range(12):
            n = int(rng.integers(2, 25))
            mzs = rng.uniform(499.5, 501.0, size=n)
            rts = rng.uniform(0.0, 10.0, size=n)
            if trial % 3 == 0:
                rts[rng.choice(n, size=max(1, n // 4), replace=False)] = np.nan
            labels = np.zeros(n, dtype=np.int32)
            start_label = trial * 10
            n_out = refine.postprocess_cluster(
                labels,
                mzs,
                rts,
                tol,
                mode,
                rt_tol,
                min_samples=2,
                start_label=start_label,
            )
            assert set(labels.tolist()) == set(
                range(start_label, start_label + n_out)
            )
            for lbl in np.unique(labels):
                mask = labels == lbl
                if int(np.count_nonzero(mask)) < 2:
                    continue
                sub_mzs = mzs[mask]
                mz_span = (
                    sub_mzs.max() - sub_mzs.min()
                    if mode == "Da"
                    else (sub_mzs.max() - sub_mzs.min()) / sub_mzs.min() * 1e6
                )
                assert mz_span <= tol + 1e-9
                if rt_tol is not None:
                    sub_rts = rts[mask]
                    finite = np.isfinite(sub_rts)
                    assert bool(finite.all()) or bool((~finite).all())
                    if finite.all():
                        assert sub_rts.max() - sub_rts.min() <= rt_tol + 1e-9

            # Verify per-m/z-group independence: restricting to any one
            # m/z sub-cluster reproduces the exact same partition.
            if rt_tol is not None:
                mz_groups = refine.cut_linkage(
                    refine.linkage_1d(mzs, mode), tol
                )
                for g in np.unique(mz_groups):
                    g_mask = mz_groups == g
                    g_labels = np.zeros(
                        int(np.count_nonzero(g_mask)), dtype=np.int32
                    )
                    refine.postprocess_cluster(
                        g_labels,
                        mzs[g_mask],
                        rts[g_mask],
                        tol,
                        mode,
                        rt_tol,
                        min_samples=2,
                        start_label=0,
                    )
                    # Same equivalence relation (pair co-membership).
                    full_sub = labels[g_mask]
                    np.testing.assert_array_equal(
                        full_sub[:, None] == full_sub[None, :],
                        g_labels[:, None] == g_labels[None, :],
                    )


class TestRefineClusters:
    def test_refines_multiple_unsorted_groups_preserving_canonical_order(
        self, da_clustering_factory
    ):
        initial = np.array([0, 1, 0, 1, 0, 0], dtype=np.int32)
        mzs = np.array([100.0, 150.0, 100.1, 150.1, 200.0, 200.1])
        rts = np.array([10.0, 10.0, 10.2, 50.0, 10.0, 10.2])

        refined_labels, label_order, n_clusters = refine.refine_clusters(
            initial,
            mzs,
            rts,
            da_clustering_factory(rt_tol=5.0),
            min_samples=2,
        )
        assert n_clusters == 4
        np.testing.assert_array_equal(label_order, [0, 2, 4, 5, 1, 3])
        restored = np.empty_like(initial)
        restored[label_order] = refined_labels
        assert restored[0] == restored[2]
        assert restored[4] == restored[5]
        assert len({restored[0], restored[4], restored[1], restored[3]}) == 4
        assert set(restored.tolist()) == {0, 1, 2, 3}

    def test_validates_1d_and_matching_lengths(self, da_clustering_factory):
        clust = da_clustering_factory()
        labels = np.array([0, 0], dtype=np.int32)
        mzs = np.array([500.0, 500.1])
        rts = np.array([10.0, 11.0])

        with pytest.raises(ValueError, match="1D arrays"):
            refine.refine_clusters(labels.reshape(1, 2), mzs, rts, clust, 2)
        with pytest.raises(ValueError, match="matching lengths"):
            refine.refine_clusters(labels, mzs[:-1], rts, clust, 2)
        with pytest.raises(ValueError, match="matching lengths"):
            refine.refine_clusters(labels, mzs, rts[:-1], clust, 2)
