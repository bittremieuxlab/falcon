"""Unit tests for falcon.cluster.refine (1D linkage and postprocessing)."""

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist as scipy_pdist

from falcon.cluster import refine


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

    def test_rt_grid_no_encoding_collision(self):
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

    def test_rt_with_mixed_nan_single_valid_and_all_nan(self):
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

        labels_1v = np.zeros(3, dtype=np.int32)
        n_1v = refine.postprocess_cluster(
            labels_1v,
            np.array([100.0, 100.05, 100.1]),
            np.array([10.0, np.nan, np.nan]),
            0.5,
            "Da",
            1.0,
            2,
            0,
        )
        assert n_1v == 2
        assert labels_1v[1] == labels_1v[2]
        assert labels_1v[0] != labels_1v[1]

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
