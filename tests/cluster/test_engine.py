"""Unit tests for falcon.cluster.engine (splitting, tiling, invariance)."""

import itertools

import joblib
import numpy as np
import pyarrow as pa
import pytest

from falcon.cluster import engine, refine
from falcon.config import ConsensusConfig
from falcon.staging import SPECTRUM_SCHEMA
from tests.helpers import make_da_clustering, make_spectrum_row


class TestPrecursorSplitsAndTiling:
    def test_da_ppm_even_chunking_and_float64_boundary(self):
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(
                np.array([100.0, 100.5, 101.5, 102.0]), 0.5, "Da", 100
            ),
            [0, 2, 4],
        )
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(
                np.array([100.0, 100.004, 101.0, 101.004]), 50, "ppm", 100
            ),
            [0, 2, 4],
        )
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(
                np.linspace(100.0, 100.6, 7), 0.5, "Da", batch_size=5
            ),
            [0, 4, 7],
        )
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(np.array([]), 0.5, "Da", 100), [0]
        )

        # Float32 precursor m/z uses float64 ppm math matching linkage_1d.
        mzs_f32 = np.array([300.0, 300.00592], dtype=np.float32)
        ppm_f32 = float(
            np.float32((mzs_f32[1] - mzs_f32[0]) / mzs_f32[0])
            * np.float32(1e6)
        )
        ppm_f64 = float(refine.linkage_1d(mzs_f32, "ppm")[0, 2])
        tol = 0.5 * (ppm_f32 + ppm_f64)
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(mzs_f32, tol, "ppm", batch_size=10),
            [0, 1, 2],
        )

    @pytest.mark.parametrize("mode", ["Da", "ppm"])
    @pytest.mark.parametrize("batch_size", [1, 3, 7, 25])
    def test_randomized_split_balance_and_gap_soundness(
        self, mode, batch_size
    ):
        rng = np.random.default_rng(42)
        tol = 0.5 if mode == "Da" else 50.0
        for _ in range(10):
            centers = rng.uniform(150.0, 1200.0, size=6)
            jitter = 0.25 if mode == "Da" else 0.015
            vals = np.concatenate(
                [
                    c + rng.uniform(0.0, jitter, size=rng.integers(1, 9))
                    for c in centers
                ]
            )
            vals = np.sort(np.concatenate([vals, vals[:3]]))
            n = len(vals)

            splits = engine.precursor_mz_splits(vals, tol, mode, batch_size)
            assert splits[0] == 0
            assert splits[-1] == n
            lengths = np.diff(splits)
            assert np.all((lengths >= 1) & (lengths <= batch_size))

            diffs = (
                np.diff(vals)
                if mode == "Da"
                else np.diff(vals) / vals[:-1] * 1e6
            )
            block_bounds = [0, *(np.flatnonzero(diffs > tol) + 1).tolist(), n]
            assert set(block_bounds).issubset(set(splits.tolist()))
            for b_start, b_end in itertools.pairwise(block_bounds):
                sub = splits[(splits >= b_start) & (splits <= b_end)]
                sub_lens = np.diff(sub)
                expected = ((b_end - b_start) + batch_size - 1) // batch_size
                assert len(sub_lens) == expected
                assert int(sub_lens.max() - sub_lens.min()) <= 1

    def test_interval_row_bounds_balances_upper_triangle_work(self):
        n = 400
        bounds = engine.interval_row_bounds(n, 4)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        pair_counts = [
            (2 * n - 1 - r0 - r1) * (r1 - r0) // 2 for r0, r1 in bounds
        ]
        assert sum(pair_counts) == n * (n - 1) // 2
        ideal = n * (n - 1) // 8
        assert all(abs(p - ideal) / ideal < 0.15 for p in pair_counts)


class TestIntervalAndPermutationInvariance:
    def test_cluster_arrow_interval_empty_singleton_and_no_reps(self):
        opts = engine._ClusterOptions(
            clustering=make_da_clustering(),
            consensus=ConsensusConfig(),
            compute_representatives=True,
        )
        empty = pa.Table.from_pylist([], SPECTRUM_SCHEMA)
        res_empty = engine.cluster_arrow_interval(empty, opts)
        assert res_empty.labels.size == 0
        assert res_empty.representatives == []

        one = pa.Table.from_pylist(
            [make_spectrum_row("s:0", 500.0)], SPECTRUM_SCHEMA
        )
        res_one = engine.cluster_arrow_interval(one, opts)
        np.testing.assert_array_equal(res_one.labels, [0])
        assert len(res_one.representatives) == 1

        rows = [
            make_spectrum_row("s:0", 500.0, [100.0, 200.0], [1.0, 1.0]),
            make_spectrum_row("s:1", 500.1, [400.0, 500.0], [1.0, 1.0]),
            make_spectrum_row("s:2", 500.2, [700.0, 800.0], [1.0, 1.0]),
        ]
        table = pa.Table.from_pylist(rows, SPECTRUM_SCHEMA)
        res_all = engine.cluster_arrow_interval(table, opts)
        assert [int(r.cluster_id) for r in res_all.representatives] == [
            0,
            1,
            2,
        ]

        opts_off = engine._ClusterOptions(
            clustering=make_da_clustering(), compute_representatives=False
        )
        res_off = engine.cluster_arrow_interval(table, opts_off)
        np.testing.assert_array_equal(res_off.labels, res_all.labels)
        assert res_off.representatives == []

    def test_randomized_row_shuffle_permutation_invariance(
        self, lance_dataset, monkeypatch
    ):
        rng = np.random.default_rng(123)
        templates = [
            ([100.0, 200.0, 300.0, 400.0], [1.0, 0.8, 0.6, 0.4]),
            ([150.0, 250.0, 350.0, 450.0], [0.5, 1.0, 0.7, 0.3]),
            ([180.0, 280.0, 380.0, 480.0], [0.9, 0.4, 0.8, 0.5]),
        ]
        base_rows = [
            make_spectrum_row(
                f"spec:{idx:02d}",
                [400.0, 400.0, 700.0][idx // 6]
                + (0.0 if idx % 3 == 0 else 0.02 * (idx % 3)),
                mz=templates[idx % 3][0],
                intensity=templates[idx % 3][1],
                precursor_charge=2,
                retention_time=np.nan if idx % 4 == 0 else float(10 + idx % 5),
            )
            for idx in range(18)
        ]
        monkeypatch.setattr(joblib, "effective_n_jobs", lambda _n=-1: 1)
        clust = make_da_clustering(rt_tol=3.0, batch_size=20)
        res_orig = engine.generate_clusters_multi(
            [lance_dataset(base_rows, name="orig.lance")], clust
        )[0]
        perm = rng.permutation(len(base_rows))
        res_shuf = engine.generate_clusters_multi(
            [
                lance_dataset(
                    [base_rows[int(i)] for i in perm], name="shuf.lance"
                )
            ],
            clust,
        )[0]
        np.testing.assert_array_equal(res_shuf.labels, res_orig.labels)
        assert [int(r.cluster_id) for r in res_shuf.representatives] == [
            int(r.cluster_id) for r in res_orig.representatives
        ]
