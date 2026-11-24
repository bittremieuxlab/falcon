"""Integration tests for Lance-backed clustering on toy datasets."""

import math
import tempfile

import joblib
import numpy as np
import pytest

from falcon.cluster import ClusterResult, engine, generate_clusters_multi
from falcon.config import ConsensusConfig
from falcon.staging import CANONICAL_SORT_COLUMNS, read_canonical_metadata
from tests.helpers import make_da_clustering, make_spectrum_row

pytestmark = pytest.mark.integration


def _attach_labels(dataset, labels):
    meta = read_canonical_metadata(dataset, CANONICAL_SORT_COLUMNS)
    return dict(zip(meta["identifier"], np.asarray(labels), strict=True))


def _spy_giant_allocations(monkeypatch) -> list[int]:
    counts: list[int] = []
    real_open = engine.open_shared_condensed

    def _spy(n, *, temp_dir=None):
        counts.append(n)
        return real_open(n, temp_dir=temp_dir)

    monkeypatch.setattr(engine, "open_shared_condensed", _spy)
    return counts


class TestClusteringPipelineIntegration:
    def test_label_alignment_under_sort_key_ties_and_isolated_split(
        self, lance_dataset
    ):
        peaks_a = [100.0, 200.0, 300.0, 400.0, 500.0]
        peaks_b = [600.0, 700.0, 800.0, 900.0, 1000.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            make_spectrum_row("f:scan:B1", 100.0, peaks_b, inten, 2, 10.0),
            make_spectrum_row("f:scan:A2", 100.0, peaks_a, inten, 2, 10.0),
            make_spectrum_row("f:scan:B2", 100.0, peaks_b, inten, 2, 10.0),
            make_spectrum_row("f:scan:A1", 100.0, peaks_a, inten, 2, 10.0),
            make_spectrum_row("f:scan:ISO", 900.0, peaks_a, inten, 2, 10.0),
        ]
        dataset = lance_dataset(rows)
        res = generate_clusters_multi(
            [dataset], make_da_clustering(), bucket_labels=["2"]
        )[0]
        assert isinstance(res, ClusterResult)
        labels, reps = res
        assert labels.dtype == np.int32
        assert (labels >= 0).all()
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2]

        by_id = _attach_labels(dataset, labels)
        assert by_id["f:scan:A1"] == by_id["f:scan:A2"]
        assert by_id["f:scan:B1"] == by_id["f:scan:B2"]
        assert (
            len({by_id["f:scan:A1"], by_id["f:scan:B1"], by_id["f:scan:ISO"]})
            == 3
        )

        res_no_rep = generate_clusters_multi(
            [dataset],
            make_da_clustering(),
            compute_representatives=False,
        )[0]
        np.testing.assert_array_equal(res_no_rep.labels, labels)
        assert res_no_rep.representatives == []

    def test_giant_tiled_path_matches_normal_path_and_cleans_memmaps(
        self, lance_dataset, tmp_path, monkeypatch
    ):
        peaks_a = [150.0, 250.0, 350.0, 450.0, 550.0]
        peaks_b = [180.0, 280.0, 380.0, 480.0, 580.0]
        inten_a = [1.0, 0.8, 0.6, 0.4, 0.2]
        inten_b = [0.2, 0.4, 0.6, 0.8, 1.0]
        rows = [
            make_spectrum_row("s:0", 500.00, peaks_a, inten_a, 2, 10.0),
            make_spectrum_row("s:1", 500.01, peaks_a, inten_a, 2, 11.0),
            make_spectrum_row("s:2", 500.02, peaks_a, inten_a, 2, 12.0),
            make_spectrum_row("s:3", 500.03, peaks_b, inten_b, 2, 13.0),
            make_spectrum_row("s:4", 500.04, peaks_b, inten_b, 2, 14.0),
            make_spectrum_row("s:5", 500.05, peaks_b, inten_b, 2, 15.0),
            make_spectrum_row("s:6", 800.00, peaks_a, inten_a, 2, 20.0),
            make_spectrum_row("s:7", 800.01, peaks_a, inten_a, 2, 21.0),
        ]
        dataset = lance_dataset(rows)
        temp_sub = tmp_path / "memmaps"
        temp_sub.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_sub))
        monkeypatch.setattr(engine, "MIN_PAIRS_PER_TILE", 4)
        monkeypatch.setattr(joblib, "effective_n_jobs", lambda _n=-1: 4)

        seen_giants = _spy_giant_allocations(monkeypatch)
        for consensus_cfg, memmap_dir in (
            (ConsensusConfig(method="medoid"), temp_sub),
            (ConsensusConfig(method="average"), None),
        ):
            before = len(seen_giants)
            res_giant = generate_clusters_multi(
                [dataset],
                make_da_clustering(batch_size=8),
                consensus_cfg,
                min_mz=100.0,
                max_mz=700.0,
                temp_dir=memmap_dir,
            )[0]
            assert len(seen_giants) == before + 1
            res_normal = generate_clusters_multi(
                [dataset],
                make_da_clustering(batch_size=100),
                consensus_cfg,
                min_mz=100.0,
                max_mz=700.0,
            )[0]
            assert len(seen_giants) == before + 1
            np.testing.assert_array_equal(res_giant.labels, res_normal.labels)
            assert (
                len(res_giant.representatives)
                == len(res_normal.representatives)
                == 3
            )
            for r_g, r_n in zip(
                res_giant.representatives,
                res_normal.representatives,
                strict=True,
            ):
                assert r_g.cluster_id == r_n.cluster_id
                np.testing.assert_allclose(r_g.mz, r_n.mz)
                np.testing.assert_allclose(r_g.intensity, r_n.intensity)
        assert list(temp_sub.glob("*.npy")) == []

    def test_multi_bucket_rt_filtering_and_unknown_charge_average(
        self, lance_dataset
    ):
        peaks = [100.0, 200.0, 300.0, 400.0, 500.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        ds_ch2 = lance_dataset(
            [
                make_spectrum_row("c2:e1", 500.0, peaks, inten, 2, 10.0),
                make_spectrum_row("c2:e2", 500.01, peaks, inten, 2, 10.5),
                make_spectrum_row("c2:l1", 500.02, peaks, inten, 2, 100.0),
                make_spectrum_row("c2:l2", 500.03, peaks, inten, 2, 100.5),
                make_spectrum_row("c2:n1", 500.04, peaks, inten, 2, np.nan),
                make_spectrum_row("c2:n2", 500.05, peaks, inten, 2, np.nan),
            ],
            charge=2,
        )
        ds_unk = lance_dataset(
            [
                make_spectrum_row("u:1", 400.00, peaks, [0.5] * 5, None, 5.0),
                make_spectrum_row("u:2", 400.01, peaks, [0.5] * 5, None, 5.5),
                make_spectrum_row("u:3", 400.02, peaks, [0.5] * 5, None, 6.0),
                make_spectrum_row(
                    "u:4",
                    400.03,
                    peaks,
                    [0.51, 0.5, 0.5, 0.5, 0.5],
                    None,
                    6.5,
                ),
            ],
            charge="unknown",
        )
        results = generate_clusters_multi(
            [ds_ch2, ds_unk],
            make_da_clustering(rt_tol=5.0),
            ConsensusConfig(method="average"),
            min_mz=100.0,
            max_mz=600.0,
        )
        by_id_ch2 = _attach_labels(ds_ch2, results[0].labels)
        assert by_id_ch2["c2:e1"] == by_id_ch2["c2:e2"]
        assert by_id_ch2["c2:l1"] == by_id_ch2["c2:l2"]
        assert by_id_ch2["c2:n1"] == by_id_ch2["c2:n2"]
        assert (
            len({by_id_ch2["c2:e1"], by_id_ch2["c2:l1"], by_id_ch2["c2:n1"]})
            == 3
        )
        assert len(results[0].representatives) == 3

        labels_unk, reps_unk = results[1]
        assert len(np.unique(labels_unk)) == 1
        assert len(reps_unk) == 1
        assert math.isnan(reps_unk[0].precursor_charge)
        assert len(reps_unk[0].mz) == 5

    def test_empty_dataset_and_parallel_pass2_forced_batch_split(
        self, lance_dataset, monkeypatch
    ):
        empty_res = generate_clusters_multi(
            [lance_dataset([], charge=2)], make_da_clustering()
        )[0]
        assert len(empty_res.labels) == 0
        assert empty_res.representatives == []

        peaks_a = [150.0, 250.0, 350.0, 450.0, 550.0]
        peaks_b = [180.0, 280.0, 380.0, 480.0, 580.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            make_spectrum_row(
                f"s:{i}",
                500.0 + 0.01 * i,
                peaks_a if i % 2 == 0 else peaks_b,
                inten,
                2,
                10.0 + i,
            )
            for i in range(8)
        ]
        ds = lance_dataset(rows)
        monkeypatch.setattr(engine, "MIN_PAIRS_PER_TILE", 3)
        monkeypatch.setattr(joblib, "effective_n_jobs", lambda _n=-1: 2)
        seen_giants = _spy_giant_allocations(monkeypatch)
        res = generate_clusters_multi(
            [ds], make_da_clustering(batch_size=4), ConsensusConfig()
        )[0]
        assert seen_giants == [4, 4]
        assert len(res.labels) == 8
        assert len(np.unique(res.labels)) == 4
        assert [int(r.cluster_id) for r in res.representatives] == [0, 1, 2, 3]
