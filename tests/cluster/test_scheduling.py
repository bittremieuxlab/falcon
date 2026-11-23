"""Unit tests for falcon.cluster.scheduling."""

import itertools

import numba as nb
import numba.typed
import numpy as np
import pytest

from falcon.cluster import scheduling


def _make_spec(task_id: int, n: int) -> scheduling.IntervalSpec:
    """Create a synthetic IntervalSpec with ``n`` spectra."""
    return scheduling.IntervalSpec(
        task_id=task_id,
        row_ids=list(range(n)),
        precursor_mzs=np.full(n, 500.0 + task_id, dtype=np.float32),
    )


class TestTileThreshold:
    @pytest.mark.parametrize(
        ("batch_size", "expected"),
        [(1, 2), (2, 2), (4, 2), (10, 5), (32768, 16384)],
    )
    def test_scales_with_batch_size_and_floors_at_two(
        self, batch_size, expected
    ):
        assert scheduling.tile_threshold(batch_size) == expected


class TestPrecursorMzSplits:
    def test_da_single_and_multiple_splits_with_exact_boundary(self):
        mzs = np.array([100.0, 100.5, 101.0, 101.5])
        assert scheduling.precursor_mz_splits(
            mzs, 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 4])

        mzs_split = np.array([100.0, 100.5, 101.5, 102.0])
        assert scheduling.precursor_mz_splits(
            mzs_split, 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 2, 4])

    def test_ppm_single_and_multiple_splits(self):
        mzs_one = np.array([100.0, 100.002, 100.003, 100.005])
        assert scheduling.precursor_mz_splits(
            mzs_one, 50, "ppm", batch_size=100
        ) == nb.typed.List([0, 4])

        mzs_two = np.array([100.0, 100.004, 101.0, 101.004])
        assert scheduling.precursor_mz_splits(
            mzs_two, 50, "ppm", batch_size=100
        ) == nb.typed.List([0, 2, 4])

    def test_true_even_chunking_for_internal_and_trailing_oversized_blocks(
        self,
    ):
        # Continuous block of 4 with batch_size=2 -> [0, 2, 4].
        mzs = np.array([100.0, 100.2, 100.4, 100.5])
        assert scheduling.precursor_mz_splits(
            mzs, 0.5, "Da", batch_size=2
        ) == nb.typed.List([0, 2, 4])

        # Internal contiguous block of 4 exceeds batch_size=3 -> evenly
        # split into (2, 2) rather than greedy (3, 1).
        mzs2 = np.array([100.0, 101.0, 102.0, 103.0, 106.0, 107.0])
        assert scheduling.precursor_mz_splits(
            mzs2, 1.0, "Da", batch_size=3
        ) == nb.typed.List([0, 2, 4, 6])

        # Trailing contiguous block of 7 exceeds batch_size=5 -> evenly
        # split into (4, 3) rather than greedy (5, 2).
        mzs_tail = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6])
        assert scheduling.precursor_mz_splits(
            mzs_tail, 0.5, "Da", batch_size=5
        ) == nb.typed.List([0, 4, 7])

        # Gap coincides with block_size == batch_size (3).
        mzs3 = np.array([100.0, 100.1, 100.2, 200.0, 200.1])
        assert scheduling.precursor_mz_splits(
            mzs3, 0.5, "Da", batch_size=3
        ) == nb.typed.List([0, 3, 5])

        # batch_size=1 splits every element into a singleton interval.
        assert scheduling.precursor_mz_splits(
            np.array([100.0, 100.0, 100.0]), 0.5, "Da", batch_size=1
        ) == nb.typed.List([0, 1, 2, 3])

    def test_empty_and_single_value(self):
        assert scheduling.precursor_mz_splits(
            np.array([]), 0.5, "Da", batch_size=100
        ) == nb.typed.List([0])
        assert scheduling.precursor_mz_splits(
            np.array([100.0]), 0.5, "Da", batch_size=100
        ) == nb.typed.List([0, 1])


class TestIntervalRowBounds:
    @pytest.mark.parametrize("n", [2, 3, 10, 100])
    @pytest.mark.parametrize("n_tiles", [1, 2, 4, 20])
    def test_exact_partition_of_rows(self, n, n_tiles):
        bounds = scheduling.interval_row_bounds(n, n_tiles)
        assert bounds
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        for r0, r1 in bounds:
            assert r0 < r1
        for prev, curr in itertools.pairwise(bounds):
            assert prev[1] == curr[0]

    def test_balances_pair_work_across_tiles(self):
        n = 400
        bounds = scheduling.interval_row_bounds(n, 4)
        assert len(bounds) == 4
        pair_counts = [
            (2 * n - 1 - r0 - r1) * (r1 - r0) // 2 for r0, r1 in bounds
        ]
        assert sum(pair_counts) == n * (n - 1) // 2
        ideal = n * (n - 1) // 8
        assert all(abs(p - ideal) / ideal < 0.15 for p in pair_counts)


class TestCostBasedChunking:
    def test_lpt_balancing_and_task_preservation(self):
        specs = [
            _make_spec(0, 100),
            _make_spec(1, 200),
            _make_spec(2, 200),
            _make_spec(3, 100),
        ]
        chunks = scheduling.cost_based_chunking(specs, 2)
        assert len(chunks) == 2
        assigned = sorted(
            (spec.task_id, spec.n_spectra)
            for chunk in chunks
            for spec in chunk
        )
        assert assigned == [(0, 100), (1, 200), (2, 200), (3, 100)]
        costs = [sum(spec.cost for spec in chunk) for chunk in chunks]
        assert costs == [50_000, 50_000]

    def test_dominant_task_isolated(self):
        specs = [
            _make_spec(0, 100),
            _make_spec(1, 1),
            _make_spec(2, 1),
            _make_spec(3, 1),
        ]
        chunks = scheduling.cost_based_chunking(specs, 2)
        assert sorted(len(chunk) for chunk in chunks) == [1, 3]

    def test_edge_cases_empty_single_excess_and_invalid_num_chunks(self):
        assert scheduling.cost_based_chunking([], 4) == []
        two_specs = [_make_spec(0, 10), _make_spec(1, 10)]
        assert len(scheduling.cost_based_chunking(two_specs, 1)) == 1
        assert len(scheduling.cost_based_chunking(two_specs, 5)) == 2

        for invalid in (0, -2):
            with pytest.raises(
                ValueError, match="num_chunks must be positive"
            ):
                scheduling.cost_based_chunking(two_specs, invalid)


class TestPrepareBucketChunks:
    def test_canonical_order_and_exact_giant_threshold_separation(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        # Insert out of canonical order so row_ids must remap cleanly.
        rows = [
            spectrum_row("s:g2", 700.02, peaks, inten, 2, 12.0),
            spectrum_row("s:n1", 500.00, peaks, inten, 2, 10.0),
            spectrum_row("s:g0", 700.00, peaks, inten, 2, 10.0),
            spectrum_row("s:n2", 500.01, peaks, inten, 2, 11.0),
            spectrum_row("s:g1", 700.01, peaks, inten, 2, 11.0),
        ]
        ds = lance_dataset(rows)
        n, splits, normal_chunks, giant_specs = (
            scheduling.prepare_bucket_chunks(
                ds,
                "2",
                da_clustering_factory(batch_size=10),
                num_chunks=4,
                threshold=3,
            )
        )
        assert n == 5
        assert list(splits) == [0, 2, 5]
        assert len(normal_chunks) == 1
        assert len(normal_chunks[0]) == 1
        assert normal_chunks[0][0].n_spectra == 2
        assert len(giant_specs) == 1
        assert giant_specs[0].n_spectra == 3

        # Taking by row_ids must yield rows in canonical order.
        giant_ids = (
            ds.take(indices=giant_specs[0].row_ids, columns=["identifier"])
            .to_pandas()["identifier"]
            .tolist()
        )
        assert giant_ids == ["s:g0", "s:g1", "s:g2"]

    def test_giant_interval_does_not_skew_normal_chunk_lpt_balance(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        rows = []
        # 1 giant interval (10 spectra, cost 100) at m/z ~100 Da.
        for i in range(10):
            rows.append(
                spectrum_row(
                    f"g:{i}", 100.0 + 0.01 * i, peaks, inten, 2, float(i)
                )
            )
        # 4 normal intervals (4 spectra each, cost 16 each; sum 64 < 100).
        for group_idx, base_mz in enumerate((200.0, 300.0, 400.0, 500.0)):
            for i in range(4):
                rows.append(
                    spectrum_row(
                        f"n:{group_idx}:{i}",
                        base_mz + 0.01 * i,
                        peaks,
                        inten,
                        2,
                        float(i),
                    )
                )
        ds = lance_dataset(rows)

        n, splits, normal_chunks, giant_specs = (
            scheduling.prepare_bucket_chunks(
                ds,
                "2",
                da_clustering_factory(batch_size=20),
                num_chunks=2,
                threshold=8,
            )
        )
        assert n == 26
        assert list(splits) == [0, 10, 14, 18, 22, 26]
        assert len(giant_specs) == 1
        assert giant_specs[0].n_spectra == 10
        assert len(normal_chunks) == 2
        assert [len(chunk) for chunk in normal_chunks] == [2, 2]
        assert [sum(s.cost for s in chunk) for chunk in normal_chunks] == [
            32,
            32,
        ]

    def test_empty_and_all_giant_buckets(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        clust = da_clustering_factory(batch_size=10)
        ds_empty = lance_dataset([], name="empty.lance")
        n_empty, splits_empty, chunks_empty, giants_empty = (
            scheduling.prepare_bucket_chunks(
                ds_empty, "2", clust, num_chunks=2, threshold=4
            )
        )
        assert n_empty == 0
        assert list(splits_empty) == [0]
        assert chunks_empty == []
        assert giants_empty == []

        peaks = [100.0, 200.0, 300.0]
        inten = [1.0, 0.8, 0.6]
        ds_giant = lance_dataset(
            [
                spectrum_row(f"g:{i}", 500.0 + 0.01 * i, peaks, inten, 2, i)
                for i in range(4)
            ],
            name="giant.lance",
        )
        n_g, splits_g, chunks_g, giants_g = scheduling.prepare_bucket_chunks(
            ds_giant, "2", clust, num_chunks=2, threshold=4
        )
        assert n_g == 4
        assert list(splits_g) == [0, 4]
        assert chunks_g == []
        assert len(giants_g) == 1
        assert giants_g[0].n_spectra == 4
