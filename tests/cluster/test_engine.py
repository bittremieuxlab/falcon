"""Unit tests for falcon.cluster.engine and falcon.cluster exports."""

import math
import os

import numpy as np
import pandas as pd
import pytest

import falcon.cluster as cluster_pkg
from falcon.cluster import engine, scheduling
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.dataset import INTERVAL_COLUMNS
from falcon.types import ClusterResult


class TestClusterPackageExports:
    def test_lazy_getattr_exposes_entry_points_and_rejects_unknown(self):
        assert cluster_pkg.generate_clusters is engine.generate_clusters
        assert (
            cluster_pkg.generate_clusters_multi
            is engine.generate_clusters_multi
        )
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
        tuples, rts = engine.interval_spectrum_tuples(df)
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

    def test_cluster_mz_interval_empty_returns_empty_cluster_result(
        self, da_clustering_factory
    ):
        empty_df = pd.DataFrame(columns=list(INTERVAL_COLUMNS))
        options = engine.ClusteringOptions(clustering=da_clustering_factory())
        res = engine.cluster_mz_interval(
            empty_df, np.empty(0, dtype=np.float32), options
        )
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
        res1 = engine.cluster_mz_interval(df1, np.array([500.0]), options)
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
        res_o = engine.cluster_mz_interval(
            df_orth, np.array([500.0, 500.1]), options
        )
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
        mzs = np.array([500.0, 500.1, 500.2, 500.3], dtype=np.float32)
        options = engine.ClusteringOptions(
            clustering=da_clustering_factory(rt_tol=5.0),
            consensus=MedoidConsensusConfig(),
        )

        res = engine.cluster_mz_interval(df, mzs, options, already_sorted=True)
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
        mzs = np.array([500.0, 500.05, 500.1])
        clust = da_clustering_factory()
        cons = AverageConsensusConfig()

        labels_on, reps_on = engine.cluster_mz_interval(
            df,
            mzs,
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
            mzs,
            engine.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=False,
            ),
        )
        assert reps_off == []
        np.testing.assert_array_equal(labels_off, labels_on)

    def test_linkage_single_vs_complete_on_chain_and_min_matches(
        self, spectrum_row, da_clustering_factory, monkeypatch
    ):
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
        prec_mzs = np.array([500.0, 500.01, 500.02])

        # Lower GC threshold to also exercise the eager cleanup branch.
        monkeypatch.setattr(engine, "_GC_COLLECT_MIN_SPECTRA", 2)
        labels_single, _ = engine.cluster_mz_interval(
            df,
            prec_mzs,
            engine.ClusteringOptions(
                clustering=da_clustering_factory(linkage="single")
            ),
        )
        labels_complete, _ = engine.cluster_mz_interval(
            df,
            prec_mzs,
            engine.ClusteringOptions(
                clustering=da_clustering_factory(linkage="complete")
            ),
        )
        assert len(np.unique(labels_single)) == 1
        assert len(np.unique(labels_complete)) == 2

        labels_gated, _ = engine.cluster_mz_interval(
            df,
            prec_mzs,
            engine.ClusteringOptions(
                clustering=da_clustering_factory(
                    linkage="single", min_matched_peaks=4
                )
            ),
        )
        assert len(np.unique(labels_gated)) == 3

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
                task_id=0,
                row_ids=[0, 1],
                precursor_mzs=np.array([500.0, 500.05]),
            ),
            scheduling.IntervalSpec(
                task_id=1,
                row_ids=[2, 3],
                precursor_mzs=np.array([700.0, 700.05]),
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

        task = scheduling.ChunkTask(
            bucket_id=3, cost=8, n_spectra=4, dataset=ds, intervals=chunk
        )
        tag, bid, cost, n_chunk, tagged_res = engine._cluster_chunk_tagged(
            task, options
        )
        assert (tag, bid, cost, n_chunk) == ("chunk", 3, 8, 4)
        assert len(tagged_res) == 2

    def test_in_process_giant_tiling_and_finalize_bucket_reordering(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
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
        clust = da_clustering_factory(batch_size=6)
        n, splits, normal_chunks, giant_specs = (
            scheduling.prepare_bucket_chunks(
                ds, "2", clust, num_chunks=4, threshold=3
            )
        )
        assert n == 5
        assert len(giant_specs) == 1
        assert len(normal_chunks) == 1

        options = engine.ClusteringOptions(
            clustering=clust, consensus=MedoidConsensusConfig()
        )
        memmap_paths: list[str] = []
        try:
            giants, tile_tasks = engine._allocate_giant_tiles(
                [(0, ds, giant_specs[0])],
                n_workers=4,
                memmap_paths=memmap_paths,
            )
            assert len(memmap_paths) == 1
            assert sum(t.tile_cost for t in tile_tasks) == pytest.approx(
                giant_specs[0].cost
            )
            for tile in tile_tasks:
                tag = engine._tile_distance(tile, 0.05, 0)
                assert tag == ("tile", tile.giant_idx, tile.tile_cost)

            bid, tid, giant_res = engine._finalize_giant(giants[0], options)
            assert (bid, tid) == (0, giant_specs[0].task_id)
            assert isinstance(giant_res, ClusterResult)
            normal_res = engine.cluster_chunk(normal_chunks[0], ds, options)

            # Supply split results out of task_id order.
            unordered = [*normal_res, (tid, giant_res)]
            bucket_res = engine.finalize_bucket(n, splits, unordered, "2")
            assert isinstance(bucket_res, ClusterResult)
            labels, reps = bucket_res
            assert len(labels) == 5
            assert len(reps) == 3
            assert [int(r.cluster_id) for r in reps] == [0, 1, 2]
        finally:
            for path in memmap_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def test_finalize_bucket_offsets_unordered_splits_cleanly(
        self, consensus_factory
    ):
        splits = [0, 3, 6]
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
