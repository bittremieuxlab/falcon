"""Integration tests for Lance-backed clustering on toy datasets."""

import math
import tempfile

import numpy as np
import pytest

from falcon.cluster import generate_clusters, generate_clusters_multi
from falcon.config import AverageConsensusConfig, MedoidConsensusConfig
from falcon.dataset import CANONICAL_SORT_COLUMNS
from falcon.types import ClusterResult

pytestmark = pytest.mark.integration


def _attach_labels(dataset, labels):
    """Map spectrum identifier to label using canonical ordering."""
    meta = dataset.to_table(columns=list(CANONICAL_SORT_COLUMNS)).to_pandas()
    meta = meta.sort_values(list(CANONICAL_SORT_COLUMNS)).reset_index(
        drop=True
    )
    meta["cluster"] = np.asarray(labels)
    return dict(zip(meta["identifier"], meta["cluster"], strict=True))


class TestClusteringPipelineIntegration:
    def test_label_alignment_under_sort_key_ties_and_isolated_split(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        peaks_a = [100.0, 200.0, 300.0, 400.0, 500.0]
        peaks_b = [600.0, 700.0, 800.0, 900.0, 1000.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            spectrum_row("f:scan:B1", 100.0, peaks_b, inten, 2, 10.0),
            spectrum_row("f:scan:A2", 100.0, peaks_a, inten, 2, 10.0),
            spectrum_row("f:scan:B2", 100.0, peaks_b, inten, 2, 10.0),
            spectrum_row("f:scan:A1", 100.0, peaks_a, inten, 2, 10.0),
            spectrum_row("f:scan:ISO", 900.0, peaks_a, inten, 2, 10.0),
        ]
        dataset = lance_dataset(rows)
        res = generate_clusters(
            dataset, da_clustering_factory(), bucket_label="2"
        )
        assert isinstance(res, ClusterResult)
        labels, reps = res
        assert res.labels is labels
        assert res.representatives is reps
        assert labels.dtype == np.int32
        assert (labels >= 0).all()
        assert len(reps) == 3
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2]

        by_id = _attach_labels(dataset, labels)
        assert by_id["f:scan:A1"] == by_id["f:scan:A2"]
        assert by_id["f:scan:B1"] == by_id["f:scan:B2"]
        assert (
            len({by_id["f:scan:A1"], by_id["f:scan:B1"], by_id["f:scan:ISO"]})
            == 3
        )

        # Skipping representatives returns bit-identical labels and [].
        res_no_rep = generate_clusters(
            dataset,
            da_clustering_factory(),
            compute_representatives=False,
        )
        assert isinstance(res_no_rep, ClusterResult)
        labels_no_rep, reps_empty = res_no_rep
        np.testing.assert_array_equal(labels_no_rep, labels)
        assert reps_empty == []

    def test_dissimilar_spectra_become_singletons(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        rows = [
            spectrum_row(
                "f:scan:1",
                100.0,
                [100.0, 200.0, 300.0, 400.0, 500.0],
                [1.0] * 5,
                2,
                10.0,
            ),
            spectrum_row(
                "f:scan:2",
                100.1,
                [600.0, 700.0, 800.0, 900.0, 1000.0],
                [1.0] * 5,
                2,
                20.0,
            ),
            spectrum_row(
                "f:scan:3",
                100.2,
                [1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
                [1.0] * 5,
                2,
                30.0,
            ),
        ]
        dataset = lance_dataset(rows)
        res = generate_clusters(dataset, da_clustering_factory())
        assert isinstance(res, ClusterResult)
        labels, reps = res
        assert len(np.unique(labels)) == 3
        assert len(reps) == 3
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2]

    def test_giant_tiled_path_matches_normal_path_and_cleans_memmaps(
        self,
        lance_dataset,
        spectrum_row,
        da_clustering_factory,
        tmp_path,
        monkeypatch,
    ):
        peaks_a = [150.0, 250.0, 350.0, 450.0, 550.0]
        peaks_b = [180.0, 280.0, 380.0, 480.0, 580.0]
        inten_a = [1.0, 0.8, 0.6, 0.4, 0.2]
        inten_b = [0.2, 0.4, 0.6, 0.8, 1.0]
        rows = [
            spectrum_row("s:0", 500.00, peaks_a, inten_a, 2, 10.0),
            spectrum_row("s:1", 500.01, peaks_a, inten_a, 2, 11.0),
            spectrum_row("s:2", 500.02, peaks_a, inten_a, 2, 12.0),
            spectrum_row("s:3", 500.03, peaks_b, inten_b, 2, 13.0),
            spectrum_row("s:4", 500.04, peaks_b, inten_b, 2, 14.0),
            spectrum_row("s:5", 500.05, peaks_b, inten_b, 2, 15.0),
            spectrum_row("s:6", 800.00, peaks_a, inten_a, 2, 20.0),
            spectrum_row("s:7", 800.01, peaks_a, inten_a, 2, 21.0),
        ]
        dataset = lance_dataset(rows)
        temp_sub = tmp_path / "memmaps"
        temp_sub.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_sub))

        for consensus_cfg in (
            None,
            MedoidConsensusConfig(),
            AverageConsensusConfig(),
        ):
            res_giant = generate_clusters(
                dataset,
                da_clustering_factory(batch_size=8),
                consensus_cfg,
                min_mz=100.0,
                max_mz=700.0,
            )
            res_normal = generate_clusters(
                dataset,
                da_clustering_factory(batch_size=100),
                consensus_cfg,
                min_mz=100.0,
                max_mz=700.0,
            )
            assert isinstance(res_giant, ClusterResult)
            assert isinstance(res_normal, ClusterResult)
            labels_giant, reps_giant = res_giant
            labels_normal, reps_normal = res_normal
            np.testing.assert_array_equal(labels_giant, labels_normal)
            assert len(reps_giant) == len(reps_normal) == 3
            for r_g, r_n in zip(reps_giant, reps_normal, strict=True):
                assert r_g.cluster_id == r_n.cluster_id
                np.testing.assert_allclose(r_g.mz, r_n.mz)
                np.testing.assert_allclose(r_g.intensity, r_n.intensity)
        assert list(temp_sub.glob("*.npy")) == []

    def test_multi_bucket_rt_filtering_and_unknown_charge_average(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        peaks = [100.0, 200.0, 300.0, 400.0, 500.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        ds_ch2 = lance_dataset(
            [
                spectrum_row("c2:e1", 500.0, peaks, inten, 2, 10.0),
                spectrum_row("c2:e2", 500.01, peaks, inten, 2, 10.5),
                spectrum_row("c2:l1", 500.02, peaks, inten, 2, 100.0),
                spectrum_row("c2:l2", 500.03, peaks, inten, 2, 100.5),
                spectrum_row("c2:n1", 500.04, peaks, inten, 2, np.nan),
                spectrum_row("c2:n2", 500.05, peaks, inten, 2, np.nan),
            ],
            charge=2,
        )
        ds_unk = lance_dataset(
            [
                spectrum_row(
                    "u:1", 400.00, peaks, [0.5, 0.5, 0.5, 0.5, 0.5], None, 5.0
                ),
                spectrum_row(
                    "u:2", 400.01, peaks, [0.5, 0.5, 0.5, 0.5, 0.5], None, 5.5
                ),
                spectrum_row(
                    "u:3", 400.02, peaks, [0.5, 0.5, 0.5, 0.5, 0.5], None, 6.0
                ),
                spectrum_row(
                    "u:4", 400.03, peaks, [0.51, 0.5, 0.5, 0.5, 0.5], None, 6.5
                ),
            ],
            charge="unknown",
        )
        results = generate_clusters_multi(
            [ds_ch2, ds_unk],
            da_clustering_factory(rt_tol=5.0),
            AverageConsensusConfig(),
            min_mz=100.0,
            max_mz=600.0,
        )
        assert len(results) == 2
        assert all(isinstance(res, ClusterResult) for res in results)

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

    def test_empty_dataset_returns_empty_labels_and_reps(
        self, lance_dataset, da_clustering_factory
    ):
        ds = lance_dataset([], charge=2)
        res = generate_clusters(ds, da_clustering_factory())
        assert isinstance(res, ClusterResult)
        assert len(res.labels) == 0
        assert res.representatives == []
