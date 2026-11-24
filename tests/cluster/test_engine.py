"""Unit tests for falcon.cluster.engine and falcon.cluster exports."""

import gc
import math
import os

import fastcluster
import numpy as np
import pandas as pd
import pytest

import falcon.cluster as cluster_pkg
from falcon.cluster import distance, engine, scheduling, tiling
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.dataset import INTERVAL_COLUMNS
from falcon.types import ClusterResult


class TestClusterPackageExports:
    def test_lazy_getattr_dir_and_unknown_rejection(self):
        assert cluster_pkg.generate_clusters is engine.generate_clusters
        assert (
            cluster_pkg.generate_clusters_multi
            is engine.generate_clusters_multi
        )
        names = dir(cluster_pkg)
        assert "generate_clusters" in names
        assert "generate_clusters_multi" in names
        missing = "nonexistent_clustering_symbol"
        with pytest.raises(AttributeError, match="no attribute"):
            getattr(cluster_pkg, missing)


class TestIntervalAndChunkClustering:
    def test_interval_spectrum_tuples_sorts_canonically_with_nan_rt_ties(
        self, spectrum_row
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 1.0, 1.0]
        df = pd.DataFrame(
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
        tuples, mzs, rts = engine.interval_spectrum_tuples(df)
        assert [float(t.precursor_mz) for t in tuples] == [
            400.0,
            500.0,
            500.0,
            500.0,
        ]
        np.testing.assert_allclose(mzs, [400.0, 500.0, 500.0, 500.0])
        assert rts[0] == pytest.approx(99.0)
        assert rts[1] == pytest.approx(5.0)
        assert math.isnan(rts[2])
        assert math.isnan(rts[3])

    def test_unsorted_interval_aligns_precursor_mzs_during_refinement(
        self, spectrum_row, da_clustering_factory
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        # Intentionally unsorted DataFrame with two distinct precursor m/z
        # pairs that share identical fragment peaks.
        df = pd.DataFrame(
            [
                spectrum_row("s:hi_b", 600.05, peaks, inten, 2, 10.0),
                spectrum_row("s:lo_a", 500.00, peaks, inten, 2, 10.0),
                spectrum_row("s:hi_a", 600.00, peaks, inten, 2, 10.0),
                spectrum_row("s:lo_b", 500.05, peaks, inten, 2, 10.0),
            ]
        )
        options = engine.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
        )
        res = engine.cluster_mz_interval(df, options, already_sorted=False)
        # Canonical order is [500.00, 500.05, 600.00, 600.05].
        assert res.labels[0] == res.labels[1]
        assert res.labels[2] == res.labels[3]
        assert res.labels[0] != res.labels[2]

    def test_cluster_mz_interval_empty_returns_empty_cluster_result(
        self, da_clustering_factory
    ):
        empty_df = pd.DataFrame(columns=list(INTERVAL_COLUMNS))
        options = engine.ClusteringOptions(clustering=da_clustering_factory())
        res = engine.cluster_mz_interval(empty_df, options)
        assert isinstance(res, ClusterResult)
        assert res.labels.dtype == np.int32
        assert len(res.labels) == 0
        assert res.representatives == []

    @pytest.mark.parametrize("compute_reps", [True, False])
    def test_cluster_mz_interval_single_spectrum_and_all_singletons(
        self, spectrum_row, da_clustering_factory, compute_reps
    ):
        options = engine.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
            compute_representatives=compute_reps,
        )

        df1 = pd.DataFrame(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)]
        )
        res1 = engine.cluster_mz_interval(df1, options)
        assert isinstance(res1, ClusterResult)
        labels1, reps1 = res1
        assert list(labels1) == [0]
        assert len(reps1) == (1 if compute_reps else 0)

        df_orth = pd.DataFrame(
            [
                spectrum_row(
                    "s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
                spectrum_row(
                    "s:2", 500.1, [400.0, 500.0], [1.0, 1.0], 2, 20.0
                ),
            ]
        )
        res_o = engine.cluster_mz_interval(df_orth, options)
        labels_o, reps_o = res_o
        assert len(np.unique(labels_o)) == 2
        assert len(reps_o) == (2 if compute_reps else 0)

    def test_all_singleton_interval_emits_representatives_ordered_by_id(
        self, spectrum_row, da_clustering_factory
    ):
        # Spectra 0 and 2 share identical peaks (initially grouped across
        # index 1, forcing label_order to permute [0, 1, 2, 3]), then RT
        # refinement splits them so every spectrum becomes a singleton.
        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        peaks_c = [700.0, 800.0, 900.0]
        inten = [1.0, 0.8, 0.6]
        rows = [
            spectrum_row("s:0", 500.0, peaks_a, inten, 2, 10.0),
            spectrum_row("s:1", 500.1, peaks_b, inten, 2, 20.0),
            spectrum_row("s:2", 500.2, peaks_a, inten, 2, 100.0),
            spectrum_row("s:3", 500.3, peaks_c, inten, 2, 40.0),
        ]
        df = pd.DataFrame(rows)
        options = engine.ClusteringOptions(
            clustering=da_clustering_factory(rt_tol=5.0),
            consensus=MedoidConsensusConfig(),
        )

        res = engine.cluster_mz_interval(df, options, already_sorted=True)
        assert len(np.unique(res.labels)) == 4
        assert not np.array_equal(res.labels, np.arange(4, dtype=np.int32))
        assert [int(r.cluster_id) for r in res.representatives] == [0, 1, 2, 3]

        for row, label in zip(rows, res.labels, strict=True):
            rep = res.representatives[int(label)]
            assert int(rep.cluster_id) == int(label)
            assert float(rep.precursor_mz) == pytest.approx(
                float(row["precursor_mz"])
            )
            assert float(rep.retention_time) == pytest.approx(
                float(row["retention_time"])
            )
            np.testing.assert_allclose(rep.mz, row["mz"])

    def test_skipping_representatives_preserves_exact_labels_and_avoids_work(
        self, spectrum_row, da_clustering_factory, monkeypatch
    ):
        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        inten = [1.0, 0.8, 0.6]
        df = pd.DataFrame(
            [
                spectrum_row("s:1", 500.0, peaks_a, inten, 2, 10.0),
                spectrum_row("s:2", 500.05, peaks_b, inten, 2, 11.0),
                spectrum_row("s:3", 500.1, peaks_a, inten, 2, 12.0),
            ]
        )
        clust = da_clustering_factory()
        cons = AverageConsensusConfig()

        labels_on, reps_on = engine.cluster_mz_interval(
            df,
            engine.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=True,
            ),
        )
        assert len(reps_on) == 2

        def _fail_rep(*_args, **_kwargs):
            raise AssertionError("get_representative_spectra must not run")

        monkeypatch.setattr(engine, "get_representative_spectra", _fail_rep)
        labels_off, reps_off = engine.cluster_mz_interval(
            df,
            engine.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=False,
            ),
        )
        assert reps_off == []
        np.testing.assert_array_equal(labels_off, labels_on)

        # Precomputed pdist without peak columns also works when
        # compute_representatives=False (both sorted and unsorted).
        meta_df = df[["identifier", "precursor_mz", "retention_time"]].copy()
        pdist = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        labels_meta, reps_meta = engine.cluster_mz_interval(
            meta_df.iloc[::-1],
            engine.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=False,
            ),
            pdist=pdist,
            already_sorted=False,
        )
        assert reps_meta == []
        np.testing.assert_array_equal(labels_meta, labels_on)

    def test_linkage_single_average_complete_on_chain_and_min_matches(
        self, spectrum_row, da_clustering_factory
    ):
        # Pairwise distances: d(0,1)=d(1,2)=0.08, d(0,2)=0.1536.
        # Second merge: single=0.08, average=0.1168, complete=0.1536.
        c = 0.92
        s = math.sqrt(1.0 - c * c)
        mzs = [100.0, 200.0, 300.0]
        df = pd.DataFrame(
            [
                spectrum_row("s:0", 500.0, mzs, [c, s, 0.0], 2, 10.0),
                spectrum_row("s:1", 500.01, mzs, [1.0, 0.0, 0.0], 2, 11.0),
                spectrum_row("s:2", 500.02, mzs, [c, 0.0, s], 2, 12.0),
            ]
        )

        # At threshold 0.10: only single merges all three.
        for linkage_name, expected_n in (
            ("single", 1),
            ("average", 2),
            ("complete", 2),
        ):
            labels, _ = engine.cluster_mz_interval(
                df,
                engine.ClusteringOptions(
                    clustering=da_clustering_factory(
                        linkage=linkage_name, distance_threshold=0.10
                    )
                ),
            )
            assert len(np.unique(labels)) == expected_n

        # At threshold 0.13: single and average merge all three;
        # complete keeps two, separating average from both extremes.
        for linkage_name, expected_n in (
            ("single", 1),
            ("average", 1),
            ("complete", 2),
        ):
            labels, _ = engine.cluster_mz_interval(
                df,
                engine.ClusteringOptions(
                    clustering=da_clustering_factory(
                        linkage=linkage_name, distance_threshold=0.13
                    )
                ),
            )
            assert len(np.unique(labels)) == expected_n

        labels_gated, _ = engine.cluster_mz_interval(
            df,
            engine.ClusteringOptions(
                clustering=da_clustering_factory(
                    linkage="single", min_matched_peaks=4
                )
            ),
        )
        assert len(np.unique(labels_gated)) == 3

    def test_cluster_mz_interval_eager_memmap_and_gc_cleanup(
        self, spectrum_row, da_clustering_factory, monkeypatch
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        df = pd.DataFrame(
            [
                spectrum_row("s:0", 500.0, peaks, inten, 2, 10.0),
                spectrum_row("s:1", 500.01, peaks, inten, 2, 11.0),
                spectrum_row("s:2", 500.02, peaks, inten, 2, 12.0),
            ]
        )
        monkeypatch.setattr(engine, "_GC_COLLECT_MIN_SPECTRA", 2)
        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 2)

        closed: list[np.ndarray] = []
        gc_calls = 0
        real_close = engine.close_memmap
        real_gc = gc.collect

        def _spy_close(mm):
            closed.append(mm)
            real_close(mm)

        def _spy_gc():
            nonlocal gc_calls
            gc_calls += 1
            return real_gc()

        monkeypatch.setattr(engine, "close_memmap", _spy_close)
        monkeypatch.setattr(gc, "collect", _spy_gc)

        res = engine.cluster_mz_interval(
            df, engine.ClusteringOptions(clustering=da_clustering_factory())
        )
        assert len(np.unique(res.labels)) == 1
        assert len(closed) == 1
        assert gc_calls == 1

        # Cleanup also runs via finally if linkage raises mid-interval.
        def _fail_linkage(*_args, **_kwargs):
            raise RuntimeError("linkage boom")

        monkeypatch.setattr(fastcluster, "linkage", _fail_linkage)
        with pytest.raises(RuntimeError, match="linkage boom"):
            engine.cluster_mz_interval(
                df,
                engine.ClusteringOptions(clustering=da_clustering_factory()),
            )
        assert len(closed) == 2
        assert gc_calls == 2

    def test_cluster_chunk_empty_batched_reads_and_tagged_wrapper(
        self, lance_dataset, spectrum_row, da_clustering_factory
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
        options = engine.ClusteringOptions(
            clustering=da_clustering_factory(batch_size=2),
            consensus=MedoidConsensusConfig(),
        )

        assert engine.cluster_chunk([], ds, options) == []

        chunk = (
            scheduling.IntervalSpec(
                task_id=0, row_ids=np.array([0, 1], dtype=np.int64)
            ),
            scheduling.IntervalSpec(
                task_id=1, row_ids=np.array([2, 3], dtype=np.int64)
            ),
        )
        res_all = engine.cluster_chunk(
            chunk, ds, options, read_batch_size=None
        )
        res_batched = engine.cluster_chunk(
            chunk, ds, options, read_batch_size=2
        )
        assert [tid for tid, _ in res_all] == [0, 1]
        for (_, r1), (_, r2) in zip(res_all, res_batched, strict=True):
            assert isinstance(r1, ClusterResult)
            assert isinstance(r2, ClusterResult)
            np.testing.assert_array_equal(r1.labels, r2.labels)

        task = engine.ChunkTask(
            bucket_id=3, cost=8, n_spectra=4, dataset=ds, intervals=chunk
        )
        done = engine._cluster_chunk_tagged(task, options)
        assert isinstance(done, engine.ChunkDone)
        assert (done.bucket_id, done.cost, done.n_spectra) == (3, 8, 4)
        assert len(done.results) == 2

    def test_finalize_giant_skips_peaks_when_representatives_disabled(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.00, peaks_a, inten, 2, 10.0),
                spectrum_row("s:1", 500.01, peaks_a, inten, 2, 11.0),
                spectrum_row("s:2", 500.02, peaks_b, inten, 2, 12.0),
                spectrum_row("s:3", 500.03, peaks_b, inten, 2, 13.0),
            ]
        )
        clust = da_clustering_factory(batch_size=8)
        spec = scheduling.IntervalSpec(
            task_id=0, row_ids=np.arange(4, dtype=np.int64)
        )
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 2)
        with tiling.GiantMemmapStore() as store:
            paths, tiles = tiling.allocate_giant_tiles(
                [(0, ds, spec)],
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


class TestPass2WorkerLimitAndNoTilingGuard:
    def test_pass2_worker_limit_respects_ram_budget_and_sysconf_fallback(
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

        # Per-giant RAM for n=1000: 16 * 1000 * 999 // 2 = 7_992_000,
        # with 0.7 headroom budget -> total_ram = ceil(2 * per_giant / 0.7).
        per_giant = 16 * 1000 * 999 // 2
        monkeypatch.setattr(engine, "_cgroup_memory_limit", lambda: None)
        ram_for_two = int(2 * per_giant / 0.7) + 100
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 1 if name == "SC_PAGE_SIZE" else ram_for_two,
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 2

        # Cgroup limit below host sysconf RAM further caps concurrency.
        ram_for_one = int(1.5 * per_giant / 0.7)
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 1 if name == "SC_PAGE_SIZE" else 100 * per_giant,
        )
        monkeypatch.setattr(
            engine, "_cgroup_memory_limit", lambda: ram_for_one
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 1

        # Less RAM than a single giant still allows 1 sequential worker.
        monkeypatch.setattr(engine, "_cgroup_memory_limit", lambda: None)
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 1 if name == "SC_PAGE_SIZE" else per_giant // 2,
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 1

        # Abundant RAM caps at min(n_workers, len(giants)).
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 1 if name == "SC_PAGE_SIZE" else 100 * per_giant,
        )
        assert engine._pass2_worker_limit(giants, n_workers=8) == 4

        # Non-positive or failing sysconf falls back to min(workers, k).
        monkeypatch.setattr(os, "sysconf", lambda _name: -1)
        assert engine._pass2_worker_limit(giants, n_workers=8) == 4

        def _fail_sysconf(_name: str) -> int:
            raise ValueError("unsupported sysconf")

        monkeypatch.setattr(os, "sysconf", _fail_sysconf)
        assert engine._pass2_worker_limit(giants, n_workers=8) == 4

    def test_cgroup_memory_limit_parsing(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing"
        cg_max = tmp_path / "memory.max"
        cg_zero = tmp_path / "memory.zero"
        cg_num = tmp_path / "memory.num"
        cg_max.write_text("max\n", encoding="ascii")
        cg_zero.write_text("0\n", encoding="ascii")
        cg_num.write_text("1048576\n", encoding="ascii")

        # Fallback paths when /proc/self/cgroup is unreadable.
        monkeypatch.setattr(engine, "_PROC_CGROUP_PATH", str(missing))
        monkeypatch.setattr(
            engine, "_CGROUP_MEMORY_PATHS", (str(missing), str(cg_max))
        )
        assert engine._cgroup_memory_limit() is None

        monkeypatch.setattr(
            engine,
            "_CGROUP_MEMORY_PATHS",
            (str(missing), str(cg_max), str(cg_zero), str(cg_num)),
        )
        assert engine._cgroup_memory_limit() == 1048576

        # Non-root cgroup v2 & v1 traversal (e.g. Slurm / systemd) picks
        # the minimum finite limit across ancestors and hierarchies.
        v2_root = tmp_path / "cgv2"
        v2_job = v2_root / "system.slice" / "job_42"
        v2_step = v2_job / "step_0"
        v2_step.mkdir(parents=True)
        (v2_root / "memory.max").write_text("max\n", encoding="ascii")
        (v2_job / "memory.max").write_text("8388608\n", encoding="ascii")
        (v2_step / "memory.max").write_text("max\n", encoding="ascii")

        v1_root = tmp_path / "cgv1_mem"
        v1_job = v1_root / "slurm" / "job_42"
        v1_job.mkdir(parents=True)
        (v1_root / "memory.limit_in_bytes").write_text(
            "9223372036854771712\n", encoding="ascii"
        )
        (v1_job / "memory.limit_in_bytes").write_text(
            "4194304\n", encoding="ascii"
        )

        proc_cg = tmp_path / "proc_cgroup"
        proc_cg.write_text(
            "malformed_line\n2:cpu:/ignored\n0::/system.slice/job_42/step_0\n",
            encoding="ascii",
        )
        monkeypatch.setattr(engine, "_PROC_CGROUP_PATH", str(proc_cg))
        monkeypatch.setattr(engine, "_CGROUP_V2_ROOT", str(v2_root))
        monkeypatch.setattr(engine, "_CGROUP_V1_MEMORY_ROOT", str(v1_root))
        monkeypatch.setattr(engine, "_CGROUP_MEMORY_PATHS", ())
        assert engine._cgroup_memory_limit() == 8388608

        proc_cg.write_text(
            "0::/system.slice/job_42/step_0\n4:cpu,memory:/slurm/job_42\n",
            encoding="ascii",
        )
        assert engine._cgroup_memory_limit() == 4194304

        # Non-ASCII bytes in /proc/self/cgroup or a limit file never raise.
        (v2_step / "memory.max").write_bytes(b"\xff\xfe\n")
        proc_cg.write_bytes(
            b"3:cpu:/caf\xc3\xa9\n0::/system.slice/job_42/step_0\n"
        )
        assert engine._cgroup_memory_limit() == 8388608

    def test_generate_clusters_multi_empty_and_mismatched_labels(
        self, lance_dataset, da_clustering_factory
    ):
        assert (
            engine.generate_clusters_multi([], da_clustering_factory()) == []
        )
        ds = lance_dataset([], charge=2)
        with pytest.raises(ValueError, match="argument 2 is longer"):
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

        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        # Two disjoint m/z intervals ensure n_pass1 >= 2 so loky workers
        # actually spawn and warm in dir_a before chdir to dir_b.
        ds = lance_dataset(
            [
                spectrum_row("s:0", 400.00, peaks, inten, 2, 1.0),
                spectrum_row("s:1", 400.01, peaks, inten, 2, 2.0),
                spectrum_row("s:2", 700.00, peaks, inten, 2, 3.0),
                spectrum_row("s:3", 700.01, peaks, inten, 2, 4.0),
            ]
        )
        monkeypatch.setattr(engine, "effective_n_workers", lambda: 2)

        seen_temp_dirs: list[str | None] = []
        real_pass1 = engine._run_pass1

        def _spy_pass1(tile_tasks, chunk_tasks, options, *args, **kwargs):
            seen_temp_dirs.append(options.temp_dir)
            return real_pass1(
                tile_tasks, chunk_tasks, options, *args, **kwargs
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

    @pytest.mark.parametrize("min_pairs", [1, 2, 3, 5, 6, 8, 6_000_000])
    def test_effective_tile_threshold_exact_two_tile_boundary(
        self, min_pairs, monkeypatch
    ):
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", min_pairs)
        batch_size = 2
        thresh = engine._effective_tile_threshold(batch_size, n_workers=2)
        assert len(scheduling.plan_tile_bounds(thresh, n_workers=2)) >= 2
        assert len(scheduling.plan_tile_bounds(thresh - 1, n_workers=2)) == 1

    def test_cluster_mz_interval_rejects_invalid_external_pdist(
        self, da_clustering_factory
    ):
        import pandas as pd

        df = pd.DataFrame(
            {
                "identifier": ["a", "b", "c"],
                "precursor_charge": [2, 2, 2],
                "precursor_mz": [500.0, 500.01, 500.02],
                "retention_time": [1.0, 2.0, 3.0],
                "mz": [
                    np.array([100.0], dtype=np.float32),
                    np.array([100.0], dtype=np.float32),
                    np.array([100.0], dtype=np.float32),
                ],
                "intensity": [
                    np.array([1.0], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                ],
            }
        )
        opts = engine.ClusteringOptions(clustering=da_clustering_factory())
        with pytest.raises(ValueError, match="1D of length 3"):
            engine.cluster_mz_interval(
                df, opts, pdist=np.zeros((3, 3), dtype=np.float32)
            )
        with pytest.raises(ValueError, match="1D of length 3"):
            engine.cluster_mz_interval(
                df, opts, pdist=np.zeros(2, dtype=np.float32)
            )

    def test_normal_interval_memmap_honors_temp_dir(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
    ):
        from falcon.cluster import distance

        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, peaks, inten, 2, i)
                for i in range(4)
            ]
        )
        seen_dirs: list[str | None] = []
        real_open = distance.open_shared_condensed

        def _spy_open(n, *, temp_dir=None):
            seen_dirs.append(None if temp_dir is None else os.fspath(temp_dir))
            return real_open(n, temp_dir=temp_dir)

        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 2)
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
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, peaks, inten, 2, i)
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

    def test_single_worker_and_small_giant_do_not_tile(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        assert engine._effective_tile_threshold(batch_size=8, n_workers=1) == 9
        # With default MIN_PAIRS_PER_TILE=4_000_000, small batch_size=8
        # cannot split into >= 2 tiles so threshold stays well above 8.
        assert engine._effective_tile_threshold(batch_size=8, n_workers=4) > 8

        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, peaks, inten, 2, i)
                for i in range(6)
            ]
        )
        seen_giants: list[int] = []
        real_alloc = engine.allocate_giant_tiles

        def _spy_alloc(raw_giants, *args, **kwargs):
            seen_giants.append(len(raw_giants))
            return real_alloc(raw_giants, *args, **kwargs)

        monkeypatch.setattr(engine, "allocate_giant_tiles", _spy_alloc)
        # Even with tiny MIN_PAIRS_PER_TILE, n_workers=1 skips tiling.
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
        peaks_a = [150.0, 250.0, 350.0]
        peaks_b = [450.0, 550.0, 650.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row(
                    f"s:{i}",
                    500.0 + 0.01 * i,
                    peaks_a if i % 2 == 0 else peaks_b,
                    inten,
                    2,
                    float(i),
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
        real_pool = engine._parallel_pool

        def _spy_pool(n_jobs):
            nonlocal pool_calls
            pool_calls += 1
            return real_pool(n_jobs)

        monkeypatch.setattr(engine, "_parallel_pool", _spy_pool)
        res = engine.generate_clusters(
            ds, da_clustering_factory(batch_size=4), temp_dir=tmp_path
        )
        # Only Pass 1 uses a pool; Pass 2 runs sequentially in-process.
        assert pool_calls == 1
        assert len(res.labels) == 8
        assert len(np.unique(res.labels)) == 4
        assert list(tmp_path.glob("*.npy")) == []

    def test_finalize_bucket_offsets_unordered_splits_and_validates(
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

        with pytest.raises(ValueError, match="Invalid splits"):
            engine.finalize_bucket(6, [0, 3], [], "2")
        with pytest.raises(ValueError, match="task IDs"):
            engine.finalize_bucket(
                6,
                splits,
                [(0, ClusterResult(np.array([0, 0, 1], dtype=np.int32), []))],
                "2",
            )
