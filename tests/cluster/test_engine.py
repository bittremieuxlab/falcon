"""Unit tests for falcon.cluster.engine (splitting, tiling, clustering)."""

import itertools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from falcon.cluster import distance, engine, refine
from falcon.config import MedoidConsensusConfig
from falcon.staging import SPECTRUM_SCHEMA
from falcon.types import ClusterResult


def _make_spec(task_id: int, n: int) -> engine.IntervalSpec:
    return engine.IntervalSpec(
        task_id=task_id, row_ids=np.arange(n, dtype=np.int64)
    )


class TestPrecursorSplitsAndScheduling:
    def test_da_ppm_and_even_chunking(self):
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
        # Oversized blocks split evenly (size difference <= 1).
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(
                np.linspace(100.0, 100.6, 7), 0.5, "Da", batch_size=5
            ),
            [0, 4, 7],
        )
        np.testing.assert_array_equal(
            engine.precursor_mz_splits(np.array([]), 0.5, "Da", 100), [0]
        )

    def test_ppm_float32_matches_float64_linkage_1d_boundary(self):
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

    def test_lpt_chunking_and_tile_bounds(self, monkeypatch):
        specs = [
            _make_spec(0, 100),
            _make_spec(1, 200),
            _make_spec(2, 200),
            _make_spec(3, 100),
        ]
        chunks = engine.cost_based_chunking(specs, 2)
        assert [sum(s.cost for s in c) for c in chunks] == [50_000, 50_000]
        assert engine.cost_based_chunking([], 4) == []

        # Row bounds partition [0, n) evenly by upper-triangle pair work.
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

        # Effective threshold disables tiling for 1 worker and honors budget.
        assert engine.effective_tile_threshold(8, n_workers=1) == 9
        monkeypatch.setattr(engine, "MIN_PAIRS_PER_TILE", 3)
        planned = engine.plan_tile_bounds(6, n_workers=4)
        assert len(planned) >= 2
        assert sum(cost for _, _, cost in planned) == pytest.approx(36.0)


class TestIntervalTilingAndCoordinator:
    def test_cluster_arrow_interval_empty_singleton_and_no_reps(
        self, spectrum_row, da_clustering_factory
    ):
        opts = engine.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
            compute_representatives=True,
        )
        empty = pa.Table.from_pylist([], SPECTRUM_SCHEMA)
        res_empty = engine.cluster_arrow_interval(empty, opts)
        assert res_empty.labels.size == 0
        assert res_empty.representatives == []

        one = pa.Table.from_pylist(
            [spectrum_row("s:0", 500.0)], SPECTRUM_SCHEMA
        )
        res_one = engine.cluster_arrow_interval(one, opts)
        np.testing.assert_array_equal(res_one.labels, [0])
        assert len(res_one.representatives) == 1

        # All-singleton interval emits representatives ordered by cluster_id.
        rows = [
            spectrum_row("s:0", 500.0, [100.0, 200.0], [1.0, 1.0]),
            spectrum_row("s:1", 500.1, [400.0, 500.0], [1.0, 1.0]),
            spectrum_row("s:2", 500.2, [700.0, 800.0], [1.0, 1.0]),
        ]
        table = pa.Table.from_pylist(rows, SPECTRUM_SCHEMA)
        res_all = engine.cluster_arrow_interval(table, opts)
        assert [int(r.cluster_id) for r in res_all.representatives] == [
            0,
            1,
            2,
        ]

        opts_off = engine.ClusteringOptions(
            clustering=da_clustering_factory(), compute_representatives=False
        )
        res_off = engine.cluster_arrow_interval(table, opts_off)
        np.testing.assert_array_equal(res_off.labels, res_all.labels)
        assert res_off.representatives == []

    def test_giant_memmap_store_and_multi_tile_distance(
        self, lance_dataset, spectrum_row, tmp_path
    ):
        rows = [
            spectrum_row(
                f"s:{i}",
                500.0 + 0.01 * i,
                [100.0 + 5.0 * i, 200.0, 300.0],
                [0.6, 0.5, 0.6],
            )
            for i in range(6)
        ]
        ds = lance_dataset(rows)
        row_ids = np.arange(6, dtype=np.int64)
        full_table = ds.take(
            indices=pa.array(row_ids), columns=["mz", "intensity"]
        )
        expected = distance.compute_condensed_distance_matrix_csr(
            *distance.extract_arrow_csr(full_table), 0.05, 0
        )

        with engine.GiantMemmapStore(temp_dir=tmp_path) as store:
            mm_path = store.allocate(0, 6)
            assert Path(mm_path).is_file()
            for r0, r1 in engine.interval_row_bounds(6, 2):
                engine.tile_distance(
                    engine.TileTask(
                        giant_idx=0,
                        dataset=ds,
                        row_ids=row_ids,
                        r0=r0,
                        r1=r1,
                        memmap_path=mm_path,
                        tile_cost=1.0,
                        tile_spectra=1.0,
                        fragment_tol=0.05,
                        min_matched_peaks=0,
                    )
                )
            filled = np.load(mm_path)
            assert np.array_equal(filled, expected)
            store.release(0)
            assert not Path(mm_path).exists()

        # Context manager cleans up on exception too.
        leaked = ""

        def _alloc_and_fail() -> None:
            nonlocal leaked
            with engine.GiantMemmapStore(temp_dir=tmp_path) as store:
                leaked = store.allocate(0, 4)
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _alloc_and_fail()
        assert not Path(leaked).exists()

    def test_cluster_chunk_and_finalize_bucket_unordered_splits(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        consensus_factory,
    ):
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.0),
                spectrum_row("s:1", 500.05),
                spectrum_row("s:2", 700.0, mz=[400.0, 500.0, 600.0]),
                spectrum_row("s:3", 700.05, mz=[400.0, 500.0, 600.0]),
            ]
        )
        opts = engine.ClusteringOptions(clustering=da_clustering_factory())
        chunk = (
            engine.IntervalSpec(0, np.array([0, 1], dtype=np.int64)),
            engine.IntervalSpec(1, np.array([2, 3], dtype=np.int64)),
        )
        res_all = engine.cluster_chunk(chunk, ds, opts)
        res_batched = engine.cluster_chunk(chunk, ds, opts, read_batch_size=2)
        for (_, r1), (_, r2) in zip(res_all, res_batched, strict=True):
            np.testing.assert_array_equal(r1.labels, r2.labels)

        # finalize_bucket reconciles out-of-order task IDs and offsets labels.
        assembled = engine.finalize_bucket(
            4,
            [0, 2, 4],
            [
                (
                    1,
                    ClusterResult(
                        np.array([0, 1], np.int32),
                        [
                            consensus_factory(cluster_id=0),
                            consensus_factory(cluster_id=1),
                        ],
                    ),
                ),
                (
                    0,
                    ClusterResult(
                        np.array([0, 0], np.int32),
                        [consensus_factory(cluster_id=0)],
                    ),
                ),
            ],
            "2",
        )
        assert assembled.labels.tolist() == [0, 0, 1, 2]
        assert [int(r.cluster_id) for r in assembled.representatives] == [
            0,
            1,
            2,
        ]

    def test_randomized_row_shuffle_permutation_invariance(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        rng = np.random.default_rng(123)
        templates = [
            ([100.0, 200.0, 300.0, 400.0], [1.0, 0.8, 0.6, 0.4]),
            ([150.0, 250.0, 350.0, 450.0], [0.5, 1.0, 0.7, 0.3]),
            ([180.0, 280.0, 380.0, 480.0], [0.9, 0.4, 0.8, 0.5]),
        ]
        base_rows = [
            spectrum_row(
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
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 1)
        clust = da_clustering_factory(rt_tol=3.0, batch_size=20)
        res_orig = engine.generate_clusters(
            lance_dataset(base_rows, name="orig.lance"), clust
        )
        perm = rng.permutation(len(base_rows))
        res_shuf = engine.generate_clusters(
            lance_dataset(
                [base_rows[int(i)] for i in perm], name="shuf.lance"
            ),
            clust,
        )
        np.testing.assert_array_equal(res_shuf.labels, res_orig.labels)
        assert [int(r.cluster_id) for r in res_shuf.representatives] == [
            int(r.cluster_id) for r in res_orig.representatives
        ]
