"""Unit tests for falcon.cluster.engine."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from falcon.cluster import engine, scheduling
from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    MedoidConsensusConfig,
    PrecursorTolerance,
)


def _default_clustering(**overrides) -> ClusteringConfig:
    defaults = {
        "precursor_tol": PrecursorTolerance(mass=0.5, mode="Da"),
        "fragment_tol": 0.05,
        "distance_threshold": 0.1,
    }
    defaults.update(overrides)
    return ClusteringConfig(**defaults)


class TestDefaultBucketLabel:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("/data/spectra/spectra_charge_2.lance", "2"),
            ("/data/spectra/spectra_charge_1_2.lance", "1, 2"),
            ("/data/spectra/spectra_charge_unknown.lance", "unknown"),
            ("/data/custom_dataset.lance", "0"),
        ],
    )
    def test_derives_label_from_uri_or_index_fallback(self, uri, expected):
        assert engine._default_bucket_label(SimpleNamespace(uri=uri), 0) == (
            expected
        )


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

    @pytest.mark.parametrize("compute_reps", [True, False])
    def test_cluster_mz_interval_single_spectrum_and_all_singletons(
        self, spectrum_row, compute_reps
    ):
        clust = _default_clustering()
        cons = MedoidConsensusConfig()

        df1 = pd.DataFrame(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)]
        )
        reps1, labels1 = engine.cluster_mz_interval(
            df1,
            np.array([500.0]),
            clust,
            cons,
            compute_representatives=compute_reps,
        )
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
        reps_o, labels_o = engine.cluster_mz_interval(
            df_orth,
            np.array([500.0, 500.1]),
            clust,
            cons,
            compute_representatives=compute_reps,
        )
        assert len(np.unique(labels_o)) == 2
        assert len(reps_o) == (2 if compute_reps else 0)

    def test_skipping_representatives_preserves_exact_labels_and_avoids_work(
        self, spectrum_row, monkeypatch
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
        clust = _default_clustering()
        cons = AverageConsensusConfig()

        reps_on, labels_on = engine.cluster_mz_interval(
            df, mzs, clust, cons, compute_representatives=True
        )
        assert len(reps_on) == 2

        def _fail_rep(*_args, **_kwargs):
            raise AssertionError("get_representative_spectra must not run")

        monkeypatch.setattr(engine, "get_representative_spectra", _fail_rep)
        reps_off, labels_off = engine.cluster_mz_interval(
            df, mzs, clust, cons, compute_representatives=False
        )
        assert reps_off == []
        np.testing.assert_array_equal(labels_off, labels_on)

    def test_linkage_single_vs_complete_on_chain_and_min_matches(
        self, spectrum_row
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
        cons = MedoidConsensusConfig()

        _, labels_single = engine.cluster_mz_interval(
            df, prec_mzs, _default_clustering(linkage="single"), cons
        )
        _, labels_complete = engine.cluster_mz_interval(
            df, prec_mzs, _default_clustering(linkage="complete"), cons
        )
        assert len(np.unique(labels_single)) == 1
        assert len(np.unique(labels_complete)) == 2

        _, labels_gated = engine.cluster_mz_interval(
            df,
            prec_mzs,
            _default_clustering(linkage="single", min_matched_peaks=4),
            cons,
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
        clust = _default_clustering(batch_size=2)
        cons = MedoidConsensusConfig()

        assert engine.cluster_chunk([], ds, clust, cons) == []

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
        res_all = engine.cluster_chunk(chunk, ds, clust, cons, batch_size=None)
        res_batched = engine.cluster_chunk(
            chunk, ds, clust, cons, batch_size=2
        )
        assert [tid for tid, _ in res_all] == [0, 1]
        for (_, (_, l1)), (_, (_, l2)) in zip(
            res_all, res_batched, strict=True
        ):
            np.testing.assert_array_equal(l1, l2)

        task = scheduling.ChunkTask(
            bucket_id=3, cost=8, n_spectra=4, dataset=ds, intervals=chunk
        )
        tag, bid, cost, n_chunk, tagged_res = engine._cluster_chunk_tagged(
            task, clust, cons, 101.0, 1500.0, True
        )
        assert (tag, bid, cost, n_chunk) == ("chunk", 3, 8, 4)
        assert len(tagged_res) == 2

    def test_in_process_giant_tiling_and_finalize_bucket_reordering(
        self, lance_dataset, spectrum_row
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
        n, splits, normal_chunks, giant_specs = (
            scheduling.prepare_bucket_chunks(
                ds,
                bucket_label="2",
                precursor_tol_mass=0.5,
                precursor_tol_mode="Da",
                batch_size=6,
                num_chunks=4,
                threshold=3,
            )
        )
        assert n == 5
        assert len(giant_specs) == 1
        assert len(normal_chunks) == 1

        clust = _default_clustering(batch_size=6)
        cons = MedoidConsensusConfig()
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

            bid, tid, giant_res = engine._finalize_giant(
                giants[0], clust, cons, 101.0, 1500.0, True
            )
            assert (bid, tid) == (0, giant_specs[0].task_id)
            normal_res = engine.cluster_chunk(
                normal_chunks[0], ds, clust, cons
            )

            # Supply split results out of task_id order.
            unordered = [*normal_res, (tid, giant_res)]
            labels, reps = engine.finalize_bucket(n, splits, unordered, "2")
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
            (1, (split1_reps, np.array([0, 1, 1], dtype=np.int32))),
            (0, (split0_reps, np.array([0, 0, 1], dtype=np.int32))),
        ]
        labels, reps = engine.finalize_bucket(6, splits, unordered, "2")
        assert list(labels) == [0, 0, 1, 2, 3, 3]
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2, 3]
