"""Unit tests for falcon.cluster.cluster."""

import itertools
import math
import os
from types import SimpleNamespace

import numba as nb
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist as scipy_pdist

from falcon.cluster import cluster, similarity
from falcon.cluster.consensus import ConsensusTuple


def _make_rep(mz_split, cluster_id):
    return ConsensusTuple(
        precursor_mz=np.float32(100.0),
        precursor_charge=np.int32(2),
        mz=np.array([100.0], dtype=np.float32),
        intensity=np.array([1.0], dtype=np.float32),
        retention_time=np.float32(0.0),
        cluster_id=np.int32(cluster_id),
        mz_split=np.int32(mz_split),
    )


def _interval_df(rows):
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Small pure helpers
# ----------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize(
        ("batch_size", "expected"),
        [(1, 2), (2, 2), (3, 2), (4, 2), (10, 5), (2**15, 2**14)],
    )
    def test_tile_threshold(self, batch_size, expected):
        assert cluster._tile_threshold(batch_size) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00:00"),
            (59.9, "0:00:59"),
            (60, "0:01:00"),
            (3661, "1:01:01"),
        ],
    )
    def test_format_hms(self, seconds, expected):
        assert cluster._format_hms(seconds) == expected

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("/data/spectra/spectra_charge_2.lance", "2"),
            ("/data/spectra/spectra_charge_1_2.lance", "1, 2"),
            ("/data/spectra/spectra_charge_-2.lance", "-2"),
            ("/data/spectra/spectra_charge_unknown.lance", "unknown"),
            ("/data/spectra/spectra_charge_other.lance", "other"),
        ],
    )
    def test_charge_label(self, uri, expected):
        assert cluster._charge_label(SimpleNamespace(uri=uri)) == expected

    @pytest.mark.parametrize("charge", [2, np.nan, None])
    def test_singleton_consensus_tuple(self, charge):
        spec = similarity.SpectrumTuple(
            precursor_mz=500.5,
            precursor_charge=charge,
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensity=np.array([0.6, 0.8], dtype=np.float64),
        )
        rep = cluster._singleton_consensus_tuple(spec, np.nan, 4)
        assert rep.precursor_mz == np.float32(500.5)
        assert rep.mz.dtype == np.float32
        assert rep.intensity.dtype == np.float32
        assert math.isnan(rep.retention_time)
        assert rep.cluster_id == np.int32(4)
        assert rep.mz_split is None
        if charge == 2:
            assert rep.precursor_charge == np.int32(2)
        else:
            assert math.isnan(rep.precursor_charge)


# ----------------------------------------------------------------------
# Precursor m/z splitting and scheduling
# ----------------------------------------------------------------------


class TestGetPrecursorMzSplits:
    def test_da_single_and_multiple_splits_with_exact_boundary(self):
        # Adjacent diff == 0.5 stays together (strict >).
        mzs = np.array([100.0, 100.5, 101.0, 101.5])
        assert cluster._get_precursor_mz_splits(
            mzs, 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 4])

        # Adjacent diff 1.0 > 0.5 creates a split at index 2.
        mzs_split = np.array([100.0, 100.5, 101.5, 102.0])
        assert cluster._get_precursor_mz_splits(
            mzs_split, 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 2, 4])

    def test_ppm_single_and_multiple_splits(self):
        # Sorted m/z: adjacent diffs are 20, 10, 20 ppm (all <= 50).
        mzs_one = np.array([100.0, 100.002, 100.003, 100.005])
        assert cluster._get_precursor_mz_splits(
            mzs_one, 50, "ppm", batch_size=100
        ) == nb.typed.List([0, 4])

        # Sorted m/z: gap from 100.004 to 101.0 is ~9960 ppm (> 50),
        # while 101.0 to 101.004 is ~39.6 ppm (<= 50).
        mzs_two = np.array([100.0, 100.004, 101.0, 101.004])
        assert cluster._get_precursor_mz_splits(
            mzs_two, 50, "ppm", batch_size=100
        ) == nb.typed.List([0, 2, 4])

    def test_batch_size_cuts_and_gap_at_batch_boundary(self):
        # Continuous block cut by batch_size=2.
        mzs = np.array([100.0, 100.2, 100.4, 100.5])
        assert cluster._get_precursor_mz_splits(
            mzs, 0.5, "Da", batch_size=2
        ) == nb.typed.List([0, 2, 4])

        # Block reaches batch_size=3 at index 3, then gap at index 4.
        mzs2 = np.array([100.0, 101.0, 102.0, 103.0, 106.0, 107.0])
        assert cluster._get_precursor_mz_splits(
            mzs2, 1.0, "Da", batch_size=3
        ) == nb.typed.List([0, 3, 4, 6])

        # Gap coincides exactly with block_size == batch_size (3).
        mzs3 = np.array([100.0, 100.1, 100.2, 200.0, 200.1])
        assert cluster._get_precursor_mz_splits(
            mzs3, 0.5, "Da", batch_size=3
        ) == nb.typed.List([0, 3, 5])

        # batch_size=1 splits every element into a singleton interval.
        assert cluster._get_precursor_mz_splits(
            np.array([100.0, 100.0, 100.0]), 0.5, "Da", batch_size=1
        ) == nb.typed.List([0, 1, 2, 3])

    def test_empty_and_single_value(self):
        assert cluster._get_precursor_mz_splits(
            np.array([]), 0.5, "Da", batch_size=100
        ) == nb.typed.List([0])
        assert cluster._get_precursor_mz_splits(
            np.array([100.0]), 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 1])


class TestIntervalRowBounds:
    @pytest.mark.parametrize("n", [2, 3, 10, 100])
    @pytest.mark.parametrize("n_tiles", [1, 2, 4, 20])
    def test_exact_partition_of_rows(self, n, n_tiles):
        bounds = cluster._interval_row_bounds(n, n_tiles)
        assert bounds
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        for r0, r1 in bounds:
            assert r0 < r1
        for prev, curr in itertools.pairwise(bounds):
            assert prev[1] == curr[0]

    def test_balances_pair_work_across_tiles(self):
        n = 400
        bounds = cluster._interval_row_bounds(n, 4)
        assert len(bounds) == 4
        pair_counts = [
            (2 * n - 1 - r0 - r1) * (r1 - r0) // 2 for r0, r1 in bounds
        ]
        assert sum(pair_counts) == n * (n - 1) // 2
        # Balanced tiles should stay within 15% of the ideal quarter.
        ideal = n * (n - 1) // 8
        assert all(abs(p - ideal) / ideal < 0.15 for p in pair_counts)


class TestCostBasedChunking:
    def test_lpt_balancing_and_task_preservation(self):
        splits = nb.typed.List([0, 100, 300, 500, 600])
        chunks = cluster.cost_based_chunking(splits, 2)
        assert len(chunks) == 2
        assigned = sorted(t for chunk in chunks for t in chunk)
        assert assigned == [
            (0, (0, 100)),
            (1, (100, 300)),
            (2, (300, 500)),
            (3, (500, 600)),
        ]
        # Two 200-wide and two 100-wide tasks split evenly: 50000 each.
        costs = [
            sum((stop - start) ** 2 for _, (start, stop) in c) for c in chunks
        ]
        assert costs == [50_000, 50_000]

    def test_dominant_task_isolated(self):
        splits = nb.typed.List([0, 100, 101, 102, 103])
        chunks = cluster.cost_based_chunking(splits, 2)
        assert sorted(len(c) for c in chunks) == [1, 3]

    def test_edge_cases_empty_single_worker_and_excess_workers(self):
        assert cluster.cost_based_chunking(nb.typed.List([0]), 4) == []
        assert (
            len(cluster.cost_based_chunking(nb.typed.List([0, 10, 20]), 1))
            == 1
        )
        assert (
            len(cluster.cost_based_chunking(nb.typed.List([0, 10, 20]), 5))
            == 2
        )


# ----------------------------------------------------------------------
# 1D complete linkage and cluster postprocessing
# ----------------------------------------------------------------------


class TestLinkage:
    def test_da_exact_and_matches_scipy_complete_linkage(self):
        values = np.array([100.0, 101.0, 105.0, 107.0, 110.0])
        z_falcon = cluster._linkage(values, "Da")
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
        z = cluster._linkage(values, "Da")
        # Closest pair is values[1]=100 and values[3]=101 (dist 1.0).
        assert sorted(z[0, :2]) == [1.0, 3.0]
        assert z[0, 2] == pytest.approx(1.0)
        assert z[0, 3] == 2

    def test_ppm_scales_by_lower_cluster_min(self):
        values = np.array([100.0, 100.001, 200.0])
        z_ppm = cluster._linkage(values, "ppm")
        assert z_ppm[0, 2] == pytest.approx(10.0, rel=1e-4)
        # Final merge span (200 - 100) / 100 * 1e6 = 1_000_000 ppm.
        assert z_ppm[1, 2] == pytest.approx(1_000_000.0, rel=1e-4)

    def test_rt_none_mode_identical_and_negative_values(self):
        z_dup = cluster._linkage(np.array([10.0, 10.0, 10.0]), None)
        np.testing.assert_allclose(z_dup[:, 2], [0.0, 0.0])

        z_neg = cluster._linkage(np.array([-5.0, -4.5, 10.0]), None)
        assert z_neg[0, 2] == pytest.approx(0.5)
        assert z_neg[1, 2] == pytest.approx(15.0)


class TestPostprocessCluster:
    def test_homogeneous_cluster_keeps_single_label(self):
        labels = np.array([99, 99, 99, 99], dtype=np.int32)
        mzs = np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", None, min_samples=2, start_label=7
        )
        assert n == 1
        assert np.all(labels == 7)

    def test_splits_distant_mz_da_and_ppm_and_respects_min_samples(self):
        # Full-span complete linkage splits chain [100.0, 100.4, 100.8]
        # when precursor_tol_mass=0.5 (span 0.8 > 0.5), even though
        # adjacent gaps are 0.4 <= 0.5.
        labels = np.zeros(4, dtype=np.int32)
        mzs = np.array([100.0, 100.1, 200.0, 200.1], dtype=np.float64)
        rts = np.zeros(4, dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", None, min_samples=2, start_label=0
        )
        assert n == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

        # With min_samples=3, both size-2 groups become 4 singletons.
        labels_ms3 = np.zeros(4, dtype=np.int32)
        n_ms3 = cluster._postprocess_cluster(
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

        # ppm: 100.0 vs 100.001 is 10 ppm (<= 20); 100.01 is 100 ppm.
        labels_ppm = np.zeros(4, dtype=np.int32)
        mzs_ppm = np.array([100.0, 100.001, 100.01, 100.011], dtype=np.float64)
        n_ppm = cluster._postprocess_cluster(
            labels_ppm, mzs_ppm, rts, 20.0, "ppm", None, 2, 0
        )
        assert n_ppm == 2
        assert labels_ppm[0] == labels_ppm[1]
        assert labels_ppm[2] == labels_ppm[3]
        assert labels_ppm[0] != labels_ppm[2]

    def test_all_singletons_and_too_few_items(self):
        labels = np.zeros(3, dtype=np.int32)
        n = cluster._postprocess_cluster(
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
            cluster._postprocess_cluster(
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
        """4 m/z x 3 RT islands must produce 12 distinct clusters."""
        mzs, rts = [], []
        for base_mz in (100.0, 200.0, 300.0, 400.0):
            for base_rt in (10.0, 50.0, 90.0):
                mzs.extend([base_mz, base_mz + 0.05])
                rts.extend([base_rt, base_rt + 0.1])
        mzs_arr = np.array(mzs, dtype=np.float64)
        rts_arr = np.array(rts, dtype=np.float64)
        labels = np.zeros(len(mzs_arr), dtype=np.int32)

        n = cluster._postprocess_cluster(
            labels, mzs_arr, rts_arr, 0.5, "Da", 1.0, 2, 0
        )
        assert n == 12
        assert len(np.unique(labels)) == 12
        for label in np.unique(labels):
            mask = labels == label
            assert mzs_arr[mask].max() - mzs_arr[mask].min() <= 0.5
            assert rts_arr[mask].max() - rts_arr[mask].min() <= 1.0

    def test_rt_with_mixed_nan_single_valid_and_all_nan(self):
        # Mixed real and NaN RTs across two m/z groups -> 4 clusters.
        mzs = np.array(
            [100.0, 100.05, 100.1, 100.15, 200.0, 200.05, 200.1, 200.15]
        )
        rts = np.array(
            [10.0, 10.1, np.nan, np.nan, 10.0, 10.1, np.nan, np.nan]
        )
        labels = np.zeros(len(mzs), dtype=np.int32)
        n = cluster._postprocess_cluster(
            labels, mzs, rts, 0.5, "Da", 1.0, 2, 0
        )
        assert n == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[6] == labels[7]
        assert len({labels[0], labels[2], labels[4], labels[6]}) == 4

        # One valid RT + two NaN RTs: valid RT stays separate from NaNs.
        labels_1v = np.zeros(3, dtype=np.int32)
        n_1v = cluster._postprocess_cluster(
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

        # All-NaN RTs: skip RT splitting and preserve m/z clusters.
        labels_all_nan = np.zeros(4, dtype=np.int32)
        n_all_nan = cluster._postprocess_cluster(
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


# ----------------------------------------------------------------------
# Label offsetting and representative relabeling
# ----------------------------------------------------------------------


class TestGlobalClusterLabels:
    def test_offset_cluster_labels_multiple_single_and_empty_splits(self):
        labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        offsets = cluster._offset_cluster_labels(
            labels, nb.typed.List([0, 3, 6])
        )
        assert list(offsets) == [0, 2]
        assert list(labels) == [0, 0, 1, 2, 3, 3]

        single = np.array([0, 1, 2], dtype=np.int32)
        assert list(
            cluster._offset_cluster_labels(single, nb.typed.List([0, 3]))
        ) == [0]
        assert list(single) == [0, 1, 2]

        empty = np.array([], dtype=np.int32)
        assert (
            list(cluster._offset_cluster_labels(empty, nb.typed.List([0])))
            == []
        )

    def test_assign_global_cluster_labels_handles_unordered_reps(self):
        labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        splits = nb.typed.List([0, 3, 6])
        # Deliberately pass split 1 reps before split 0 reps.
        reps = [
            _make_rep(mz_split=1, cluster_id=0),
            _make_rep(mz_split=1, cluster_id=1),
            _make_rep(mz_split=0, cluster_id=0),
            _make_rep(mz_split=0, cluster_id=1),
        ]
        out = cluster._assign_global_cluster_labels(labels, reps, splits)
        assert [int(s.cluster_id) for s in out] == [0, 1, 2, 3]


# ----------------------------------------------------------------------
# In-process interval clustering, chunking, tiling, and finalization
# ----------------------------------------------------------------------


class TestIntervalAndChunkClustering:
    def test_interval_spectrum_tuples_sorts_canonically_with_nan_rt_ties(
        self, spectrum_row
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 1.0, 1.0]
        df = _interval_df(
            [
                spectrum_row(
                    "s:b", 500.0, peaks, inten, retention_time=np.nan
                ),
                spectrum_row(
                    "s:a", 500.0, peaks, inten, retention_time=np.nan
                ),
                spectrum_row("s:c", 500.0, peaks, inten, retention_time=5.0),
                spectrum_row("s:d", 400.0, peaks, inten, retention_time=99.0),
            ]
        )
        tuples, rts = cluster._interval_spectrum_tuples(df)
        assert [float(t.precursor_mz) for t in tuples] == [
            400.0,
            500.0,
            500.0,
            500.0,
        ]
        assert rts[0] == pytest.approx(99.0)
        assert rts[1] == pytest.approx(5.0)
        assert math.isnan(rts[2])
        assert math.isnan(rts[3])

    def test_cluster_mz_interval_single_spectrum_and_all_singletons(
        self, spectrum_row
    ):
        df1 = _interval_df(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)]
        )
        reps1, labels1 = cluster._cluster_mz_interval(
            df1,
            np.array([0]),
            np.array([500.0]),
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
        )
        assert list(labels1) == [0]
        assert len(reps1) == 1
        assert reps1[0].cluster_id == 0

        # Orthogonal spectra -> every spectrum becomes a singleton.
        df_orth = _interval_df(
            [
                spectrum_row(
                    "s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
                spectrum_row(
                    "s:2", 500.1, [400.0, 500.0], [1.0, 1.0], 2, 20.0
                ),
            ]
        )
        reps_o, labels_o = cluster._cluster_mz_interval(
            df_orth,
            np.array([0, 1]),
            np.array([500.0, 500.1]),
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
        )
        assert len(np.unique(labels_o)) == 2
        assert len(reps_o) == 2

    def test_cluster_mz_interval_does_not_mutate_consensus_params(
        self, spectrum_row
    ):
        """Medoid consensus must not mutate caller's params dict."""
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        df = _interval_df(
            [
                spectrum_row("s:1", 500.0, peaks, inten, 2, 10.0),
                spectrum_row("s:2", 500.05, peaks, inten, 2, 11.0),
                spectrum_row("s:3", 500.1, peaks, inten, 2, 12.0),
            ]
        )
        params = {}
        reps, labels = cluster._cluster_mz_interval(
            df,
            np.array([0, 1, 2]),
            np.array([500.0, 500.05, 500.1]),
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            params,
        )
        assert params == {}
        assert len(reps) == 1
        assert len(np.unique(labels)) == 1

    def test_linkage_single_vs_complete_on_chain_and_min_matches(
        self, spectrum_row
    ):
        # Construct 3 unit-norm spectra on shared peaks where:
        #   d(0, 1) = 0.08, d(1, 2) = 0.08, d(0, 2) > 0.25.
        # At threshold 0.10: single merges all 3; complete splits them.
        # Basis peaks [100, 200, 300]:
        # v0 = [c, s, 0], v1 = [1, 0, 0], v2 = [c, 0, s]
        # cos(v0, v1) = cos(v1, v2) = 0.92 -> dist = 0.08
        # cos(v0, v2) = 0.92^2 = 0.8464 -> dist = 0.1536 > 0.10
        c = 0.92
        s = math.sqrt(1.0 - c * c)
        mzs = [100.0, 200.0, 300.0]
        df = _interval_df(
            [
                spectrum_row("s:0", 500.0, mzs, [c, s, 0.0], 2, 10.0),
                spectrum_row("s:1", 500.01, mzs, [1.0, 0.0, 0.0], 2, 11.0),
                spectrum_row("s:2", 500.02, mzs, [c, 0.0, s], 2, 12.0),
            ]
        )
        _, labels_single = cluster._cluster_mz_interval(
            df,
            np.array([0, 1, 2]),
            np.array([500.0, 500.01, 500.02]),
            "single",
            0.10,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
        )
        _, labels_complete = cluster._cluster_mz_interval(
            df,
            np.array([0, 1, 2]),
            np.array([500.0, 500.01, 500.02]),
            "complete",
            0.10,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
        )
        assert len(np.unique(labels_single)) == 1
        assert len(np.unique(labels_complete)) == 2

        # Requiring min_matches=4 (> 3 peaks) prevents any merge.
        _, labels_gated = cluster._cluster_mz_interval(
            df,
            np.array([0, 1, 2]),
            np.array([500.0, 500.01, 500.02]),
            "single",
            0.10,
            4,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
        )
        assert len(np.unique(labels_gated)) == 3

    def test_cluster_chunk_empty_batched_reads_and_tagged_wrapper(
        self, lance_dataset, spectrum_row
    ):
        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.0, peaks_a, inten, 2, 10.0),
                spectrum_row("s:1", 500.05, peaks_a, inten, 2, 11.0),
                spectrum_row("s:2", 700.0, peaks_b, inten, 2, 12.0),
                spectrum_row("s:3", 700.05, peaks_b, inten, 2, 13.0),
            ]
        )
        assert (
            cluster.cluster_chunk(
                [], ds, "complete", 0.1, 0, 0.5, "Da", None, 0.05, "medoid", {}
            )
            == []
        )
        chunk = [
            (0, [0, 1], np.array([0, 1]), np.array([500.0, 500.05])),
            (1, [2, 3], np.array([2, 3]), np.array([700.0, 700.05])),
        ]
        res_all = cluster.cluster_chunk(
            chunk,
            ds,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
            batch_size=None,
        )
        # batch_size=2 forces two separate dataset.take calls.
        res_batched = cluster.cluster_chunk(
            chunk,
            ds,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            "medoid",
            {},
            batch_size=2,
        )
        assert [tid for tid, _ in res_all] == [0, 1]
        for (_, (_, l1)), (_, (_, l2)) in zip(
            res_all, res_batched, strict=True
        ):
            np.testing.assert_array_equal(l1, l2)

        tag, bid, cost, n_chunk, tagged_res = cluster._cluster_chunk_tagged(
            chunk,
            ds,
            bucket_id=3,
            cost=8,
            n_chunk=4,
            linkage="complete",
            distance_threshold=0.1,
            min_matches=0,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=None,
            fragment_mz_tol=0.05,
            consensus_method="medoid",
            consensus_params={},
            batch_size=2,
        )
        assert (tag, bid, cost, n_chunk) == ("chunk", 3, 8, 4)
        assert len(tagged_res) == 2

    def test_in_process_giant_tiling_and_finalize_bucket_reordering(
        self, lance_dataset, spectrum_row
    ):
        """Exercise giant tiling and bucket finalization in-process."""
        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.0, peaks_a, inten, 2, 10.0),
                spectrum_row("s:1", 500.01, peaks_a, inten, 2, 11.0),
                spectrum_row("s:2", 500.02, peaks_b, inten, 2, 12.0),
                spectrum_row("s:3", 500.03, peaks_b, inten, 2, 13.0),
                spectrum_row("s:4", 800.0, peaks_a, inten, 2, 14.0),
            ]
        )
        n, splits, data_chunks, giant_intervals = (
            cluster._prepare_bucket_chunks(
                ds,
                precursor_tol_mass=0.5,
                precursor_tol_mode="Da",
                batch_size=6,
                num_chunks=4,
                tile_threshold=3,
            )
        )
        assert n == 5
        assert len(giant_intervals) == 1
        assert len(data_chunks) == 1

        gi_task_id, gi_rows, gi_idx, gi_mz, gi_n = giant_intervals[0]
        giants = [
            {
                "bucket_id": 0,
                "dataset": ds,
                "task_id": gi_task_id,
                "row_ids": gi_rows,
                "idx": gi_idx,
                "mz": gi_mz,
                "n": gi_n,
                "cost": gi_n * gi_n,
            }
        ]
        memmap_paths = []
        try:
            tile_tasks = cluster._build_giant_tile_tasks(
                giants, n_workers=4, memmap_paths=memmap_paths
            )
            assert len(memmap_paths) == 1
            assert sum(t[5] for t in tile_tasks) == pytest.approx(
                giants[0]["cost"]
            )
            for gi, row_ids, r0, r1, path, tile_cost in tile_tasks:
                tag = cluster._tile_distance(
                    ds, row_ids, r0, r1, path, 0.05, 0, gi, tile_cost
                )
                assert tag == ("tile", gi, tile_cost)

            giant_tag = cluster._finalize_giant(
                ds,
                gi_rows,
                gi_idx,
                gi_mz,
                giants[0]["path"],
                bucket_id=0,
                task_id=gi_task_id,
                n_chunk=gi_n,
                linkage="complete",
                distance_threshold=0.1,
                min_matches=0,
                precursor_tol_mass=0.5,
                precursor_tol_mode="Da",
                rt_tol=None,
                fragment_mz_tol=0.05,
                consensus_method="medoid",
                consensus_params={},
            )
            assert giant_tag[0] == "giant"
            normal_res = cluster.cluster_chunk(
                data_chunks[0],
                ds,
                "complete",
                0.1,
                0,
                0.5,
                "Da",
                None,
                0.05,
                "medoid",
                {},
            )
            # Supply split results out of task_id order.
            unordered = [*normal_res, (gi_task_id, giant_tag[4])]
            labels, reps = cluster._finalize_bucket(n, splits, unordered, "2")
            assert len(labels) == 5
            assert len(reps) == 3
            assert labels[0] == labels[1]
            assert labels[2] == labels[3]
            assert len({labels[0], labels[2], labels[4]}) == 3
        finally:
            for path in memmap_paths:
                if os.path.exists(path):
                    os.unlink(path)
