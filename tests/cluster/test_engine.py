"""Unit tests for falcon.cluster.engine and falcon.cluster exports."""

import os

import numpy as np
import pytest

import falcon.cluster as cluster_pkg
from falcon.cluster import distance, engine, interval, scheduling, tiling
from falcon.config import MedoidConsensusConfig
from falcon.types import ClusterResult


class TestClusterPackageExports:
    def test_lazy_getattr_dir_and_unknown_rejection(self):
        assert cluster_pkg.generate_clusters is engine.generate_clusters
        assert (
            cluster_pkg.generate_clusters_multi
            is engine.generate_clusters_multi
        )
        assert engine.cluster_mz_interval is interval.cluster_mz_interval
        names = dir(cluster_pkg)
        assert "generate_clusters" in names
        assert "generate_clusters_multi" in names
        missing = "nonexistent_clustering_symbol"
        with pytest.raises(AttributeError, match="no attribute"):
            getattr(cluster_pkg, missing)


class TestChunkAndGiantWorkers:
    def test_cluster_chunk_empty_batched_reads_and_skips_charge_without_reps(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        peaks_b = [400.0, 500.0, 600.0]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.0, retention_time=10.0),
                spectrum_row("s:1", 500.05, retention_time=11.0),
                spectrum_row("s:2", 700.0, mz=peaks_b, retention_time=12.0),
                spectrum_row("s:3", 700.05, mz=peaks_b, retention_time=13.0),
            ]
        )
        options_on = engine.ClusteringOptions(
            clustering=da_clustering_factory(batch_size=2),
            consensus=MedoidConsensusConfig(),
            compute_representatives=True,
        )
        assert engine.cluster_chunk([], ds, options_on) == []

        chunk = (
            scheduling.IntervalSpec(
                task_id=0, row_ids=np.array([0, 1], dtype=np.int64)
            ),
            scheduling.IntervalSpec(
                task_id=1, row_ids=np.array([2, 3], dtype=np.int64)
            ),
        )
        res_all = engine.cluster_chunk(
            chunk, ds, options_on, read_batch_size=None
        )
        res_batched = engine.cluster_chunk(
            chunk, ds, options_on, read_batch_size=2
        )
        assert [tid for tid, _ in res_all] == [0, 1]
        for (_, r1), (_, r2) in zip(res_all, res_batched, strict=True):
            assert isinstance(r1, ClusterResult)
            assert isinstance(r2, ClusterResult)
            np.testing.assert_array_equal(r1.labels, r2.labels)

        seen_columns: list[list[str]] = []
        orig_take = ds.take

        def _spy_take(*args, **kwargs):
            seen_columns.append(list(kwargs.get("columns", [])))
            return orig_take(*args, **kwargs)

        monkeypatch.setattr(ds, "take", _spy_take)
        options_off = engine.ClusteringOptions(
            clustering=da_clustering_factory(batch_size=2),
            consensus=MedoidConsensusConfig(),
            compute_representatives=False,
        )
        res_no_reps = engine.cluster_chunk(chunk, ds, options_off)
        assert seen_columns == [
            ["precursor_mz", "retention_time", "mz", "intensity"]
        ]
        for (_, r_on), (_, r_off) in zip(res_all, res_no_reps, strict=True):
            np.testing.assert_array_equal(r_off.labels, r_on.labels)
            assert r_off.representatives == []

    def test_finalize_giant_skips_peaks_when_representatives_disabled(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        peaks_b = [400.0, 500.0, 600.0]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.00, retention_time=10.0),
                spectrum_row("s:1", 500.01, retention_time=11.0),
                spectrum_row("s:2", 500.02, mz=peaks_b, retention_time=12.0),
                spectrum_row("s:3", 500.03, mz=peaks_b, retention_time=13.0),
            ]
        )
        clust = da_clustering_factory(batch_size=8)
        spec = scheduling.IntervalSpec(
            task_id=0, row_ids=np.arange(4, dtype=np.int64)
        )
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 2)
        with tiling.GiantMemmapStore() as store:
            paths, tiles = tiling.allocate_giant_tiles(
                [(ds, spec)],
                clust.fragment_tol,
                clust.min_matched_peaks,
                n_workers=2,
                memmap_store=store,
            )
            for tile in tiles:
                tiling.tile_distance(tile)

            giant_on = engine.GiantInterval(
                giant_idx=0,
                bucket_id=0,
                dataset=ds,
                spec=spec,
                memmap_path=paths[0],
                options=engine.ClusteringOptions(
                    clustering=clust,
                    consensus=MedoidConsensusConfig(),
                    compute_representatives=True,
                ),
            )
            done_on = engine.finalize_giant(giant_on)
            assert len(done_on.result.representatives) == 2

            seen_columns: list[list[str]] = []
            orig_take = ds.take

            def _spy_take(*args, **kwargs):
                seen_columns.append(list(kwargs.get("columns", [])))
                return orig_take(*args, **kwargs)

            monkeypatch.setattr(ds, "take", _spy_take)
            giant_off = engine.GiantInterval(
                giant_idx=0,
                bucket_id=0,
                dataset=ds,
                spec=spec,
                memmap_path=paths[0],
                options=engine.ClusteringOptions(
                    clustering=clust,
                    consensus=MedoidConsensusConfig(),
                    compute_representatives=False,
                ),
            )
            done_off = engine.finalize_giant(giant_off)
            assert seen_columns == [["precursor_mz", "retention_time"]]
            assert done_off.result.representatives == []
            np.testing.assert_array_equal(
                done_off.result.labels, done_on.result.labels
            )


class TestPass2RamBudget:
    def test_pass2_worker_limit_respects_ram_budget_and_unavailable_fallback(
        self, lance_dataset, da_clustering_factory, monkeypatch
    ):
        ds = lance_dataset([], charge=2)
        options = engine.ClusteringOptions(clustering=da_clustering_factory())

        def _giant(idx: int, n: int) -> engine.GiantInterval:
            return engine.GiantInterval(
                giant_idx=idx,
                bucket_id=0,
                dataset=ds,
                spec=scheduling.IntervalSpec(
                    task_id=idx, row_ids=np.arange(n, dtype=np.int64)
                ),
                memmap_path="",
                options=options,
            )

        assert engine._pass2_worker_limit([], n_workers=8) == 1
        assert engine._pass2_worker_limit([_giant(0, 100)], n_workers=8) == 1
        giants = [_giant(i, 1000) for i in range(4)]
        assert engine._pass2_worker_limit(giants, n_workers=1) == 1

        # Per-giant RAM for n=1000: 16 * 1000 * 999 // 2 = 7_992_000.
        per_giant = 16 * 1000 * 999 // 2
        ram_for_two = int(2 * per_giant / 0.7) + 100
        monkeypatch.setattr(
            engine, "effective_memory_bytes", lambda: ram_for_two
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 2

        ram_for_one = int(1.5 * per_giant / 0.7)
        monkeypatch.setattr(
            engine, "effective_memory_bytes", lambda: ram_for_one
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 1

        monkeypatch.setattr(
            engine, "effective_memory_bytes", lambda: per_giant // 2
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 1

        monkeypatch.setattr(
            engine, "effective_memory_bytes", lambda: 100 * per_giant
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 4

        monkeypatch.setattr(engine, "effective_memory_bytes", lambda: None)
        assert engine._pass2_worker_limit(giants, n_workers=8) == 4


class TestPass1JointScheduling:
    def test_pass1_joint_lpt_ordering_and_unexpected_done_rejection(
        self, lance_dataset, da_clustering_factory, monkeypatch, caplog
    ):
        ds = lance_dataset([], charge=2)
        opts = engine.ClusteringOptions(clustering=da_clustering_factory())
        light_tile = tiling.TileTask(
            giant_idx=0,
            dataset=ds,
            row_ids=np.arange(4, dtype=np.int64),
            r0=0,
            r1=2,
            memmap_path="",
            tile_cost=10.0,
            tile_spectra=2.0,
            fragment_tol=0.05,
            min_matched_peaks=0,
        )
        heavy_chunk = engine.ChunkTask(
            bucket_id=0,
            cost=500,
            n_spectra=20,
            dataset=ds,
            intervals=(),
        )
        submitted: list[object] = []

        def _fake_pool(_n_jobs):
            def _run(jobs):
                for _fn, args, _kw in jobs:
                    submitted.append(args[0])
                # Complete light tile first (~2% < 5%) to exercise
                # progress-log throttling before final 100% log.
                yield tiling.TileDone(
                    giant_idx=0, tile_cost=10.0, tile_spectra=2.0
                )
                yield engine.ChunkDone(
                    bucket_id=0, cost=500, n_spectra=20, results=[]
                )

            return _run

        monkeypatch.setattr(engine, "loky_unordered_pool", _fake_pool)
        with caplog.at_level("INFO", logger="falcon.cluster.engine"):
            out = engine._run_pass1(
                [heavy_chunk],
                [light_tile],
                opts,
                n_buckets=1,
                n_workers=2,
                total_spectra=22,
            )
        assert submitted == [heavy_chunk, light_tile]
        assert out == {0: []}
        progress_logs = [
            rec.message
            for rec in caplog.records
            if "Clustering progress:" in rec.message
        ]
        assert len(progress_logs) == 1
        assert "100%" in progress_logs[0]

        # Unexpected completion type fails loudly with TypeError.
        def _bad_pool(_n_jobs):
            def _run(_jobs):
                yield "bogus"

            return _run

        monkeypatch.setattr(engine, "loky_unordered_pool", _bad_pool)
        with pytest.raises(TypeError, match="Unexpected Pass 1 completion"):
            engine._run_pass1(
                [heavy_chunk],
                [],
                opts,
                n_buckets=1,
                n_workers=1,
                total_spectra=20,
            )


class TestGiantTilingAndMemmaps:
    def test_generate_clusters_multi_empty_and_mismatched_labels_fail_fast(
        self, lance_dataset, da_clustering_factory, monkeypatch
    ):
        assert (
            engine.generate_clusters_multi([], da_clustering_factory()) == []
        )
        ds = lance_dataset([], charge=2)

        def _forbid_prepare(*_args, **_kwargs):
            raise AssertionError("prepare_bucket_chunks must not run")

        monkeypatch.setattr(engine, "prepare_bucket_chunks", _forbid_prepare)
        with pytest.raises(ValueError, match="must match"):
            engine.generate_clusters_multi(
                [ds], da_clustering_factory(), bucket_labels=["a", "b"]
            )

    def test_relative_temp_dir_absolutized_for_warm_loky_workers(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
    ):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        rel_tmp = dir_b / "rel_memmaps"
        dir_a.mkdir()
        rel_tmp.mkdir(parents=True)

        ds = lance_dataset(
            [
                spectrum_row("s:0", 400.00, retention_time=1.0),
                spectrum_row("s:1", 400.01, retention_time=2.0),
                spectrum_row("s:2", 700.00, retention_time=3.0),
                spectrum_row("s:3", 700.01, retention_time=4.0),
            ]
        )
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 2)

        seen_temp_dirs: list[str | None] = []
        real_pass1 = engine._run_pass1

        def _spy_pass1(chunk_tasks, tile_tasks, options, *args, **kwargs):
            seen_temp_dirs.append(options.temp_dir)
            return real_pass1(
                chunk_tasks, tile_tasks, options, *args, **kwargs
            )

        monkeypatch.setattr(engine, "_run_pass1", _spy_pass1)

        monkeypatch.chdir(dir_a)
        _ = engine.generate_clusters(
            ds, da_clustering_factory(batch_size=10), temp_dir=dir_a
        )
        monkeypatch.chdir(dir_b)
        res = engine.generate_clusters(
            ds,
            da_clustering_factory(batch_size=10),
            temp_dir="rel_memmaps",
        )
        assert seen_temp_dirs == [str(dir_a), str(rel_tmp)]
        assert os.path.isabs(str(seen_temp_dirs[-1]))
        assert len(res.labels) == 4
        assert len(np.unique(res.labels)) == 2
        assert list(rel_tmp.glob("*.npy")) == []

    def test_normal_interval_memmap_honors_temp_dir(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
    ):
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, retention_time=i)
                for i in range(4)
            ]
        )
        seen_dirs: list[str | None] = []
        real_open = distance.open_shared_condensed

        def _spy_open(n, *, temp_dir=None):
            seen_dirs.append(None if temp_dir is None else os.fspath(temp_dir))
            return real_open(n, temp_dir=temp_dir)

        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 2)
        monkeypatch.setattr(distance, "open_shared_condensed", _spy_open)
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 1)

        res = engine.generate_clusters(
            ds, da_clustering_factory(batch_size=10), temp_dir=tmp_path
        )
        assert seen_dirs == [str(tmp_path)]
        assert len(res.labels) == 4
        assert list(tmp_path.glob("*.npy")) == []

    @pytest.mark.parametrize("fail_stage", ["pass1", "pass2"])
    def test_giant_memmaps_cleaned_up_on_worker_error(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
        fail_stage,
    ):
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, retention_time=i)
                for i in range(8)
            ]
        )
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 3)
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 2)

        def _boom(_arg):
            raise RuntimeError(f"{fail_stage} worker boom")

        if fail_stage == "pass1":
            monkeypatch.setattr(engine, "tile_distance", _boom)
        else:
            monkeypatch.setattr(engine, "finalize_giant", _boom)

        with pytest.raises(RuntimeError, match=f"{fail_stage} worker boom"):
            engine.generate_clusters(
                ds, da_clustering_factory(batch_size=4), temp_dir=tmp_path
            )
        assert list(tmp_path.glob("*.npy")) == []

    def test_single_worker_skips_tiling(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, retention_time=i)
                for i in range(6)
            ]
        )
        seen_giants: list[int] = []
        real_alloc = engine.allocate_giant_tiles

        def _spy_alloc(giants, *args, **kwargs):
            seen_giants.append(len(giants))
            return real_alloc(giants, *args, **kwargs)

        monkeypatch.setattr(engine, "allocate_giant_tiles", _spy_alloc)
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 2)
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 1)

        res = engine.generate_clusters(ds, da_clustering_factory(batch_size=8))
        assert seen_giants == [0]
        assert len(res.labels) == 6
        assert len(np.unique(res.labels)) == 1

    def test_pass2_falls_back_to_sequential_when_ram_caps_workers_to_one(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
    ):
        peaks_b = [450.0, 550.0, 650.0]
        ds = lance_dataset(
            [
                spectrum_row(
                    f"s:{i}",
                    500.0 + 0.01 * i,
                    mz=[150.0, 250.0, 350.0] if i % 2 == 0 else peaks_b,
                    retention_time=float(i),
                )
                for i in range(8)
            ]
        )
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 3)
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 2)
        monkeypatch.setattr(
            engine, "_pass2_worker_limit", lambda _giants, _workers: 1
        )
        pool_calls = 0
        real_pool = engine.loky_unordered_pool

        def _spy_pool(n_jobs):
            nonlocal pool_calls
            pool_calls += 1
            return real_pool(n_jobs)

        monkeypatch.setattr(engine, "loky_unordered_pool", _spy_pool)
        res = engine.generate_clusters(
            ds, da_clustering_factory(batch_size=4), temp_dir=tmp_path
        )
        assert pool_calls == 1
        assert len(res.labels) == 8
        assert len(np.unique(res.labels)) == 4
        assert list(tmp_path.glob("*.npy")) == []


class TestFinalizeBucket:
    def test_offsets_unordered_splits_and_handles_no_reps(
        self, consensus_factory
    ):
        splits = np.array([0, 3, 6], dtype=np.int64)
        split0_reps = [
            consensus_factory(cluster_id=0),
            consensus_factory(cluster_id=1),
        ]
        split1_reps = [
            consensus_factory(cluster_id=0),
            consensus_factory(cluster_id=1),
        ]
        unordered = [
            (
                1,
                ClusterResult(
                    np.array([0, 1, 1], dtype=np.int32), split1_reps
                ),
            ),
            (
                0,
                ClusterResult(
                    np.array([0, 0, 1], dtype=np.int32), split0_reps
                ),
            ),
        ]
        res = engine.finalize_bucket(6, splits, unordered, "2")
        assert isinstance(res, ClusterResult)
        assert list(res.labels) == [0, 0, 1, 2, 3, 3]
        assert [int(r.cluster_id) for r in res.representatives] == [0, 1, 2, 3]

        no_reps = engine.finalize_bucket(
            6,
            splits,
            [
                (1, ClusterResult(np.array([0, 1, 1], dtype=np.int32), [])),
                (0, ClusterResult(np.array([0, 0, 1], dtype=np.int32), [])),
            ],
            "2",
        )
        assert list(no_reps.labels) == [0, 0, 1, 2, 3, 3]
        assert no_reps.representatives == []

    @pytest.mark.parametrize(
        ("n", "splits", "results", "match"),
        [
            (6, [0, 3], [], "Invalid splits"),
            (6, [0, 5, 3, 6], [], "Invalid splits"),
            (6, [0, 10, 6], [], "Invalid splits"),
            (6, np.array([[0, 6]]), [], "Invalid splits"),
            (
                6,
                [0, 3, 6],
                [(0, ClusterResult(np.array([0, 0, 1], dtype=np.int32), []))],
                "task IDs",
            ),
            (
                6,
                [0, 3, 6],
                [
                    (0, ClusterResult(np.array([0, 0], dtype=np.int32), [])),
                    (
                        1,
                        ClusterResult(np.array([0, 1, 1], dtype=np.int32), []),
                    ),
                ],
                "expected 3 label",
            ),
            (
                6,
                [0, 3, 6],
                [
                    (
                        0,
                        ClusterResult(
                            np.array([-1, 0, 0], dtype=np.int32), []
                        ),
                    ),
                    (
                        1,
                        ClusterResult(np.array([0, 1, 1], dtype=np.int32), []),
                    ),
                ],
                "negative cluster labels",
            ),
        ],
    )
    def test_validates_splits_lengths_and_non_negative_labels(
        self, n, splits, results, match
    ):
        with pytest.raises(ValueError, match=match):
            engine.finalize_bucket(n, splits, results, "2")


class TestGenerateClustersInvariants:
    def test_randomized_row_shuffle_permutation_invariance(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        # Verify shuffling Lance row order (including tied precursor m/z
        # and NaN RTs) yields identical cluster partitions by identifier
        # and identical representative spectra.
        rng = np.random.default_rng(123)
        peak_templates = [
            ([100.0, 200.0, 300.0, 400.0], [1.0, 0.8, 0.6, 0.4]),
            ([150.0, 250.0, 350.0, 450.0], [0.5, 1.0, 0.7, 0.3]),
            ([180.0, 280.0, 380.0, 480.0], [0.9, 0.4, 0.8, 0.5]),
        ]
        base_rows = []
        for idx in range(18):
            mz_base = [400.0, 400.0, 700.0][idx // 6] + (
                0.0 if idx % 3 == 0 else 0.02 * (idx % 3)
            )
            rt = np.nan if idx % 4 == 0 else float(10 + (idx % 5))
            p_mz, p_int = peak_templates[idx % len(peak_templates)]
            base_rows.append(
                spectrum_row(
                    f"spec:{idx:02d}",
                    mz_base,
                    mz=p_mz,
                    intensity=p_int,
                    precursor_charge=2,
                    retention_time=rt,
                )
            )

        monkeypatch.setattr(engine, "effective_n_workers", lambda: 1)
        clust = da_clustering_factory(rt_tol=3.0, batch_size=20)
        ds_orig = lance_dataset(base_rows, name="orig.lance")
        res_orig = engine.generate_clusters(ds_orig, clust)

        perm = rng.permutation(len(base_rows))
        ds_shuf = lance_dataset(
            [base_rows[int(i)] for i in perm], name="shuf.lance"
        )
        res_shuf = engine.generate_clusters(ds_shuf, clust)

        np.testing.assert_array_equal(res_shuf.labels, res_orig.labels)
        assert len(res_shuf.representatives) == len(res_orig.representatives)
        for rep_a, rep_b in zip(
            res_orig.representatives, res_shuf.representatives, strict=True
        ):
            assert int(rep_a.cluster_id) == int(rep_b.cluster_id)
            assert float(rep_a.precursor_mz) == pytest.approx(
                float(rep_b.precursor_mz)
            )
            np.testing.assert_allclose(rep_a.mz, rep_b.mz)
            np.testing.assert_allclose(rep_a.intensity, rep_b.intensity)
