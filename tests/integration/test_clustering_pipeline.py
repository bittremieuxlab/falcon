"""Integration tests for Lance-backed clustering on toy datasets."""

import math
import tempfile

import numpy as np

from falcon.cluster import cluster


def _attach_labels(dataset, labels):
    """Map spectrum identifier to label using canonical ordering."""
    meta = dataset.to_table(
        columns=["identifier", "precursor_mz", "retention_time"]
    ).to_pandas()
    meta = meta.sort_values(
        ["precursor_mz", "retention_time", "identifier"]
    ).reset_index(drop=True)
    meta["cluster"] = np.asarray(labels)
    return dict(zip(meta["identifier"], meta["cluster"], strict=True))


class TestClusteringPipelineIntegration:
    def test_label_alignment_under_sort_key_ties_and_isolated_split(
        self, lance_dataset, spectrum_row
    ):
        """Canonical tie-breaking aligns labels and isolates far m/z."""
        peaks_a = [100.0, 200.0, 300.0, 400.0, 500.0]
        peaks_b = [600.0, 700.0, 800.0, 900.0, 1000.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            spectrum_row("f:scan:B1", 100.0, peaks_b, inten, 2, 10.0),
            spectrum_row("f:scan:A2", 100.0, peaks_a, inten, 2, 10.0),
            spectrum_row("f:scan:B2", 100.0, peaks_b, inten, 2, 10.0),
            spectrum_row("f:scan:A1", 100.0, peaks_a, inten, 2, 10.0),
            # Isolated precursor m/z -> own single-spectrum split.
            spectrum_row("f:scan:ISO", 900.0, peaks_a, inten, 2, 10.0),
        ]
        dataset = lance_dataset(rows)
        labels, reps = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        labels_arr = np.asarray(labels)
        assert (labels_arr >= 0).all()
        assert len(reps) == 3
        by_id = _attach_labels(dataset, labels_arr)
        assert by_id["f:scan:A1"] == by_id["f:scan:A2"]
        assert by_id["f:scan:B1"] == by_id["f:scan:B2"]
        assert (
            len({by_id["f:scan:A1"], by_id["f:scan:B1"], by_id["f:scan:ISO"]})
            == 3
        )

    def test_dissimilar_spectra_become_singletons(
        self, lance_dataset, spectrum_row
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
        labels, reps = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        assert len(np.unique(np.asarray(labels))) == 3
        assert len(reps) == 3

    def test_giant_tiled_path_matches_normal_path_and_cleans_memmaps(
        self, lance_dataset, spectrum_row, tmp_path, monkeypatch
    ):
        """Force the multiprocess giant-interval path (batch_size=8,
        tile_threshold=4) and verify identical labels, representatives,
        and temp-file cleanup for both medoid and average consensus.
        """
        peaks_a = [150.0, 250.0, 350.0, 450.0, 550.0]
        peaks_b = [180.0, 280.0, 380.0, 480.0, 580.0]
        inten_a = [1.0, 0.8, 0.6, 0.4, 0.2]
        inten_b = [0.2, 0.4, 0.6, 0.8, 1.0]
        # 6 close spectra (giant interval >= tile_threshold=4) + 2 far
        # spectra (normal chunk < tile_threshold=4).
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

        avg_params = {
            "min_mz": 100.0,
            "max_mz": 700.0,
            "bin_size": 0.1,
            "outlier_cutoff_lower": 1.5,
            "outlier_cutoff_upper": 1.5,
        }
        for method, params in [("medoid", {}), ("average", avg_params)]:
            labels_giant, reps_giant = cluster.generate_clusters(
                dataset,
                "complete",
                0.1,
                0,
                0.5,
                "Da",
                None,
                0.05,
                batch_size=8,
                consensus_method=method,
                consensus_params=params,
            )
            labels_normal, reps_normal = cluster.generate_clusters(
                dataset,
                "complete",
                0.1,
                0,
                0.5,
                "Da",
                None,
                0.05,
                batch_size=100,
                consensus_method=method,
                consensus_params=params,
            )
            np.testing.assert_array_equal(labels_giant, labels_normal)
            assert len(reps_giant) == len(reps_normal) == 3
            for r_g, r_n in zip(reps_giant, reps_normal, strict=True):
                assert r_g.cluster_id == r_n.cluster_id
                np.testing.assert_allclose(r_g.mz, r_n.mz)
                np.testing.assert_allclose(r_g.intensity, r_n.intensity)
        # No shared distance memmap files left behind after completion.
        assert list(temp_sub.glob("*.npy")) == []

    def test_multi_bucket_rt_filtering_and_unknown_charge_average(
        self, lance_dataset, spectrum_row
    ):
        """Multi-bucket clustering with RT split and missing charge."""
        peaks = [100.0, 200.0, 300.0, 400.0, 500.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        # Charge 2: identical peaks at early RT, late RT, and NaN RT.
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
        # Unknown charge: 4 spectra averaged into one consensus.
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
        avg_params = {
            "min_mz": 100.0,
            "max_mz": 600.0,
            "bin_size": 0.1,
            "outlier_cutoff_lower": 1.5,
            "outlier_cutoff_upper": 1.5,
        }
        results = cluster.generate_clusters_multi(
            [ds_ch2, ds_unk],
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            rt_tol=5.0,
            fragment_tol=0.05,
            batch_size=2**15,
            consensus_method="average",
            consensus_params=avg_params,
        )
        assert len(results) == 2

        by_id_ch2 = _attach_labels(ds_ch2, results[0][0])
        assert by_id_ch2["c2:e1"] == by_id_ch2["c2:e2"]
        assert by_id_ch2["c2:l1"] == by_id_ch2["c2:l2"]
        assert by_id_ch2["c2:n1"] == by_id_ch2["c2:n2"]
        assert (
            len({by_id_ch2["c2:e1"], by_id_ch2["c2:l1"], by_id_ch2["c2:n1"]})
            == 3
        )

        labels_unk, reps_unk = results[1]
        assert len(np.unique(np.asarray(labels_unk))) == 1
        assert len(reps_unk) == 1
        assert math.isnan(reps_unk[0].precursor_charge)
        assert len(reps_unk[0].mz) == 5

    def test_empty_dataset_returns_empty_labels_and_reps(self, lance_dataset):
        ds = lance_dataset([], charge=2)
        labels, reps = cluster.generate_clusters(
            ds,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        assert len(labels) == 0
        assert reps == []
