"""Unit tests for falcon.cluster.refine (1D linkage and postprocessing)."""

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist as scipy_pdist

from falcon.cluster import refine


class TestLinkage1DAndCut:
    def test_cut_linkage_matches_scipy_fcluster(self):
        z = refine.linkage_1d(np.array([10.0, 11.0, 20.0, 22.0]), "Da")
        for threshold in (0.5, 1.0, 2.0, 12.0):
            expected = (sch.fcluster(z, threshold, "distance") - 1).astype(
                np.int32
            )
            actual = refine.cut_linkage(z, threshold)
            assert actual.dtype == np.int32
            np.testing.assert_array_equal(actual, expected)

    def test_randomized_da_and_ppm_match_scipy_complete_linkage(self):
        rng = np.random.default_rng(7)
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

        # PPM complete linkage: |x_i - x_j| / min(x_i, x_j) * 1e6.
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

    def test_iter_cluster_groups_handles_noise_and_runs(self):
        clusters = np.array([-1, -1, 0, 0, -1, 5, 5, 5], dtype=np.int32)
        assert list(refine.iter_cluster_groups(clusters)) == [
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 5),
            (5, 8),
        ]
        assert list(refine.iter_cluster_groups(np.array([], np.int32))) == []


class TestPostprocessAndRefineClusters:
    def test_splits_mz_respects_min_samples_and_isolates_nan_rt(self):
        # Distant m/z splits into two clusters or singletons below min_samples.
        mzs = np.array([100.0, 100.1, 200.0, 200.1])
        rts = np.zeros(4)
        labels = np.zeros(4, dtype=np.int32)
        n = refine.postprocess_cluster(labels, mzs, rts, 0.5, "Da", None, 2, 0)
        assert n == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

        labels_ms3 = np.zeros(4, dtype=np.int32)
        assert (
            refine.postprocess_cluster(
                labels_ms3, mzs, rts, 0.5, "Da", None, 3, 10
            )
            == 4
        )
        assert sorted(labels_ms3.tolist()) == [10, 11, 12, 13]

        # Interleaved RTs across distinct m/z groups refine independently, and
        # NaN RTs form their own group separate from finite RTs.
        labels_rt = np.zeros(6, dtype=np.int32)
        n_rt = refine.postprocess_cluster(
            labels_rt,
            np.array([100.0, 100.1, 100.05, 100.08, 200.0, 200.1]),
            np.array([10.0, 10.5, np.nan, np.nan, 10.0, 50.0]),
            0.5,
            "Da",
            2.0,
            min_samples=2,
            start_label=0,
        )
        assert n_rt == 4
        assert labels_rt[0] == labels_rt[1]
        assert labels_rt[2] == labels_rt[3]
        assert labels_rt[0] != labels_rt[2]
        assert labels_rt[4] != labels_rt[5]

    @pytest.mark.parametrize("mode", ["Da", "ppm"])
    @pytest.mark.parametrize("rt_tol", [None, 2.0])
    def test_randomized_span_rt_and_contiguity_invariants(self, mode, rt_tol):
        rng = np.random.default_rng(19)
        tol = 0.5 if mode == "Da" else 40.0
        for trial in range(12):
            n = int(rng.integers(2, 25))
            mzs = rng.uniform(499.5, 501.0, size=n)
            rts = rng.uniform(0.0, 10.0, size=n)
            if trial % 3 == 0:
                rts[rng.choice(n, size=max(1, n // 4), replace=False)] = np.nan
            labels = np.zeros(n, dtype=np.int32)
            start = trial * 10
            n_out = refine.postprocess_cluster(
                labels,
                mzs,
                rts,
                tol,
                mode,
                rt_tol,
                min_samples=2,
                start_label=start,
            )
            assert set(labels.tolist()) == set(range(start, start + n_out))
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

    def test_refine_clusters_preserves_canonical_order(
        self, da_clustering_factory
    ):
        initial = np.array([0, 1, 0, 1, 0, 0], dtype=np.int32)
        mzs = np.array([100.0, 150.0, 100.1, 150.1, 200.0, 200.1])
        rts = np.array([10.0, 10.0, 10.2, 50.0, 10.0, 10.2])

        refined, order, n_clusters = refine.refine_clusters(
            initial, mzs, rts, da_clustering_factory(rt_tol=5.0), min_samples=2
        )
        assert n_clusters == 4
        np.testing.assert_array_equal(order, [0, 2, 4, 5, 1, 3])
        restored = np.empty_like(initial)
        restored[order] = refined
        assert restored[0] == restored[2]
        assert restored[4] == restored[5]
        assert len({restored[0], restored[4], restored[1], restored[3]}) == 4
