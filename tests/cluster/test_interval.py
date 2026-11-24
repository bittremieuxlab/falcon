"""Unit tests for falcon.cluster.interval (single-interval clustering)."""

import gc
import math

import fastcluster
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from falcon.cluster import distance, interval
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.dataset import INTERVAL_COLUMNS, SPECTRUM_SCHEMA
from falcon.types import ClusterResult


class TestIntervalSpectrumTuples:
    def test_sorts_canonically_with_nan_rt_ties(self, spectrum_row):
        df = pd.DataFrame(
            [
                spectrum_row("s:b", 500.0, retention_time=np.nan),
                spectrum_row("s:a", 500.0, retention_time=np.nan),
                spectrum_row("s:c", 500.0, retention_time=5.0),
                spectrum_row("s:d", 400.0, retention_time=99.0),
            ]
        )
        tuples, mzs, rts = interval.interval_spectrum_tuples(df)
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


class TestClusterMzInterval:
    def test_unsorted_interval_aligns_precursor_mzs_during_refinement(
        self, spectrum_row, da_clustering_factory
    ):
        df = pd.DataFrame(
            [
                spectrum_row("s:hi_b", 600.05, retention_time=10.0),
                spectrum_row("s:lo_a", 500.00, retention_time=10.0),
                spectrum_row("s:hi_a", 600.00, retention_time=10.0),
                spectrum_row("s:lo_b", 500.05, retention_time=10.0),
            ]
        )
        options = interval.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
        )
        res = interval.cluster_mz_interval(df, options, already_sorted=False)
        assert res.labels[0] == res.labels[1]
        assert res.labels[2] == res.labels[3]
        assert res.labels[0] != res.labels[2]

    def test_empty_returns_empty_cluster_result(self, da_clustering_factory):
        empty_df = pd.DataFrame(columns=list(INTERVAL_COLUMNS))
        options = interval.ClusteringOptions(
            clustering=da_clustering_factory()
        )
        res = interval.cluster_mz_interval(empty_df, options)
        assert isinstance(res, ClusterResult)
        assert res.labels.dtype == np.int32
        assert len(res.labels) == 0
        assert res.representatives == []

    @pytest.mark.parametrize("compute_reps", [True, False])
    def test_single_spectrum_and_all_singletons(
        self, spectrum_row, da_clustering_factory, compute_reps
    ):
        options = interval.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
            compute_representatives=compute_reps,
        )
        df1 = pd.DataFrame([spectrum_row("s:1", 500.0, retention_time=10.0)])
        res1 = interval.cluster_mz_interval(df1, options)
        assert isinstance(res1, ClusterResult)
        assert list(res1.labels) == [0]
        assert len(res1.representatives) == (1 if compute_reps else 0)

        # Singleton metadata-only path with precomputed empty pdist.
        if not compute_reps:
            meta1 = df1[["precursor_mz", "retention_time"]]
            res_meta1 = interval.cluster_mz_interval(
                meta1,
                options,
                pdist=np.empty(0, dtype=np.float32),
                already_sorted=True,
            )
            assert list(res_meta1.labels) == [0]
            assert res_meta1.representatives == []

        df_orth = pd.DataFrame(
            [
                spectrum_row("s:1", 500.0, retention_time=10.0),
                spectrum_row(
                    "s:2", 500.1, mz=[400.0, 500.0, 600.0], retention_time=20.0
                ),
            ]
        )
        res_o = interval.cluster_mz_interval(df_orth, options)
        assert len(np.unique(res_o.labels)) == 2
        assert len(res_o.representatives) == (2 if compute_reps else 0)

    def test_all_singleton_interval_emits_representatives_ordered_by_id(
        self, spectrum_row, da_clustering_factory
    ):
        rows = [
            spectrum_row("s:0", 500.0, retention_time=10.0),
            spectrum_row(
                "s:1", 500.1, mz=[400.0, 500.0, 600.0], retention_time=20.0
            ),
            spectrum_row("s:2", 500.2, retention_time=100.0),
            spectrum_row(
                "s:3", 500.3, mz=[700.0, 800.0, 900.0], retention_time=40.0
            ),
        ]
        df = pd.DataFrame(rows)
        options = interval.ClusteringOptions(
            clustering=da_clustering_factory(rt_tol=5.0),
            consensus=MedoidConsensusConfig(),
        )
        res = interval.cluster_mz_interval(df, options, already_sorted=True)
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

    def test_skipping_representatives_preserves_labels_and_avoids_work(
        self, spectrum_row, da_clustering_factory, monkeypatch
    ):
        df = pd.DataFrame(
            [
                spectrum_row("s:1", 500.0, retention_time=10.0),
                spectrum_row(
                    "s:2",
                    500.05,
                    mz=[400.0, 500.0, 600.0],
                    retention_time=11.0,
                ),
                spectrum_row("s:3", 500.1, retention_time=12.0),
            ]
        )
        clust = da_clustering_factory()
        cons = AverageConsensusConfig()

        labels_on, reps_on = interval.cluster_mz_interval(
            df,
            interval.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=True,
            ),
        )
        assert len(reps_on) == 2

        def _fail_rep(*_args, **_kwargs):
            raise AssertionError("get_representative_spectra must not run")

        monkeypatch.setattr(interval, "get_representative_spectra", _fail_rep)
        labels_off, reps_off = interval.cluster_mz_interval(
            df,
            interval.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=False,
            ),
        )
        assert reps_off == []
        np.testing.assert_array_equal(labels_off, labels_on)

        meta_df = df[["identifier", "precursor_mz", "retention_time"]].copy()
        pdist = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        labels_meta, reps_meta = interval.cluster_mz_interval(
            meta_df.iloc[::-1],
            interval.ClusteringOptions(
                clustering=clust,
                consensus=cons,
                compute_representatives=False,
            ),
            pdist=pdist,
            already_sorted=False,
        )
        assert reps_meta == []
        np.testing.assert_array_equal(labels_meta, labels_on)

    @pytest.mark.parametrize(
        ("linkage_name", "threshold", "min_matches", "expected_n"),
        [
            ("single", 0.10, 0, 1),
            ("average", 0.10, 0, 2),
            ("complete", 0.10, 0, 2),
            ("single", 0.13, 0, 1),
            ("average", 0.13, 0, 1),
            ("complete", 0.13, 0, 2),
            ("single", 0.10, 4, 3),
        ],
    )
    def test_linkage_modes_and_min_matched_peaks_gate(
        self,
        spectrum_row,
        da_clustering_factory,
        linkage_name,
        threshold,
        min_matches,
        expected_n,
    ):
        # d(0,1)=d(1,2)=0.08, d(0,2)=0.1536.
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
        labels, _ = interval.cluster_mz_interval(
            df,
            interval.ClusteringOptions(
                clustering=da_clustering_factory(
                    linkage=linkage_name,
                    distance_threshold=threshold,
                    min_matched_peaks=min_matches,
                )
            ),
        )
        assert len(np.unique(labels)) == expected_n

    def test_rejects_invalid_external_pdist_including_small_intervals(
        self, spectrum_row, da_clustering_factory
    ):
        opts = interval.ClusteringOptions(clustering=da_clustering_factory())
        df3 = pd.DataFrame(
            [spectrum_row(f"s:{i}", 500.0 + 0.01 * i) for i in range(3)]
        )
        with pytest.raises(ValueError, match="1D of length 3"):
            interval.cluster_mz_interval(
                df3, opts, pdist=np.zeros((3, 3), dtype=np.float32)
            )
        with pytest.raises(ValueError, match="1D of length 3"):
            interval.cluster_mz_interval(
                df3, opts, pdist=np.zeros(2, dtype=np.float32)
            )
        df1 = df3.iloc[:1]
        with pytest.raises(ValueError, match="1D of length 0"):
            interval.cluster_mz_interval(
                df1, opts, pdist=np.zeros(1, dtype=np.float32)
            )

    @pytest.mark.parametrize(
        ("consensus_cfg", "compute_reps", "expect_early_release"),
        [
            (MedoidConsensusConfig(), True, False),
            (MedoidConsensusConfig(), False, True),
            (AverageConsensusConfig(), True, True),
        ],
    )
    def test_eager_memmap_release_timing_and_error_cleanup(
        self,
        spectrum_row,
        da_clustering_factory,
        monkeypatch,
        consensus_cfg,
        compute_reps,
        expect_early_release,
    ):
        # Exact threshold n == MEMMAP_MIN_SPECTRA (3) triggers memmap+GC.
        df = pd.DataFrame(
            [
                spectrum_row(f"s:{i}", 500.0 + 0.01 * i, retention_time=10 + i)
                for i in range(3)
            ]
        )
        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 3)

        events: list[str] = []
        real_close = interval.close_memmap
        real_refine = interval.refine_clusters
        real_gc = gc.collect

        def _spy_close(mm):
            events.append("close")
            real_close(mm)

        def _spy_refine(*args, **kwargs):
            events.append("refine")
            return real_refine(*args, **kwargs)

        def _spy_gc():
            events.append("gc")
            return real_gc()

        monkeypatch.setattr(interval, "close_memmap", _spy_close)
        monkeypatch.setattr(interval, "refine_clusters", _spy_refine)
        monkeypatch.setattr(gc, "collect", _spy_gc)

        opts = interval.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=consensus_cfg,
            compute_representatives=compute_reps,
        )
        res = interval.cluster_mz_interval(df, opts)
        assert len(np.unique(res.labels)) == 1
        assert events.count("close") == 1
        assert events.count("gc") == 1
        if expect_early_release:
            assert events.index("close") < events.index("refine")
        else:
            assert events.index("refine") < events.index("close")

        # Cleanup also runs via finally if linkage raises mid-interval.
        events.clear()

        def _fail_linkage(*_args, **_kwargs):
            raise RuntimeError("linkage boom")

        monkeypatch.setattr(fastcluster, "linkage", _fail_linkage)
        with pytest.raises(RuntimeError, match="linkage boom"):
            interval.cluster_mz_interval(df, opts)
        assert events == ["close", "gc"]


class TestClusterArrowInterval:
    @pytest.mark.parametrize("compute_reps", [True, False])
    @pytest.mark.parametrize("charge", [2, None])
    def test_matches_dataframe_path_and_copies_representative_peaks(
        self, spectrum_row, da_clustering_factory, compute_reps, charge
    ):
        rows = [
            spectrum_row(
                "s:0", 500.00, precursor_charge=charge, retention_time=10.0
            ),
            spectrum_row(
                "s:1", 500.01, precursor_charge=charge, retention_time=11.0
            ),
            spectrum_row(
                "s:2",
                500.02,
                mz=[400.0, 500.0, 600.0],
                precursor_charge=charge,
                retention_time=12.0,
            ),
        ]
        table = pa.Table.from_pylist(rows, SPECTRUM_SCHEMA)
        if not compute_reps:
            table = table.select(
                ["precursor_mz", "retention_time", "mz", "intensity"]
            )
        opts = interval.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
            compute_representatives=compute_reps,
        )
        arrow_res = interval.cluster_arrow_interval(table, opts)
        df_res = interval.cluster_mz_interval(
            pd.DataFrame(rows), opts, already_sorted=True
        )
        np.testing.assert_array_equal(arrow_res.labels, df_res.labels)
        assert len(arrow_res.representatives) == len(df_res.representatives)
        if compute_reps:
            mz_flat, _, _ = distance.extract_arrow_csr(table)
            for rep_a, rep_d in zip(
                arrow_res.representatives, df_res.representatives, strict=True
            ):
                assert int(rep_a.cluster_id) == int(rep_d.cluster_id)
                np.testing.assert_allclose(rep_a.mz, rep_d.mz)
                assert not np.shares_memory(rep_a.mz, mz_flat)

    def test_empty_and_singleton_arrow_intervals(
        self, spectrum_row, da_clustering_factory
    ):
        empty = pa.Table.from_pylist([], SPECTRUM_SCHEMA)
        opts_on = interval.ClusteringOptions(
            clustering=da_clustering_factory(), compute_representatives=True
        )
        opts_off = interval.ClusteringOptions(
            clustering=da_clustering_factory(), compute_representatives=False
        )
        res_empty = interval.cluster_arrow_interval(empty, opts_on)
        assert len(res_empty.labels) == 0
        assert res_empty.representatives == []

        one = pa.Table.from_pylist(
            [spectrum_row("s:0", 500.0)], SPECTRUM_SCHEMA
        )
        res_one_off = interval.cluster_arrow_interval(one, opts_off)
        assert list(res_one_off.labels) == [0]
        assert res_one_off.representatives == []

        res_one_on = interval.cluster_arrow_interval(one, opts_on)
        assert list(res_one_on.labels) == [0]
        assert len(res_one_on.representatives) == 1

        with pytest.raises(ValueError, match="pdist must be 1D of length 0"):
            interval.cluster_arrow_interval(
                one, opts_off, pdist=np.array([0.0], dtype=np.float32)
            )

    @pytest.mark.parametrize("compute_reps", [True, False])
    def test_arrow_interval_with_precomputed_pdist(
        self, spectrum_row, da_clustering_factory, compute_reps
    ):
        rows = [
            spectrum_row("s:0", 500.00, retention_time=10.0),
            spectrum_row("s:1", 500.01, retention_time=11.0),
            spectrum_row(
                "s:2", 500.02, mz=[400.0, 500.0, 600.0], retention_time=12.0
            ),
        ]
        full_table = pa.Table.from_pylist(rows, SPECTRUM_SCHEMA)
        mz_flat, int_flat, offsets = distance.extract_arrow_csr(full_table)
        pdist = distance.compute_condensed_distance_matrix_csr(
            mz_flat, int_flat, offsets, fragment_tol=0.05, min_matched_peaks=0
        )
        table = (
            full_table
            if compute_reps
            else full_table.select(["precursor_mz", "retention_time"])
        )
        opts = interval.ClusteringOptions(
            clustering=da_clustering_factory(),
            consensus=MedoidConsensusConfig(),
            compute_representatives=compute_reps,
        )
        res_pdist = interval.cluster_arrow_interval(table, opts, pdist=pdist)
        res_direct = interval.cluster_arrow_interval(full_table, opts)
        np.testing.assert_array_equal(res_pdist.labels, res_direct.labels)
        assert len(res_pdist.representatives) == len(
            res_direct.representatives
        )
