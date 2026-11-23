"""Unit tests for falcon.cluster.scheduling."""

import itertools

import numba as nb
import numpy as np
import pytest

from falcon.cluster import scheduling


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
        splits = nb.typed.List([0, 100, 300, 500, 600])
        chunks = scheduling.cost_based_chunking(splits, 2)
        assert len(chunks) == 2
        assigned = sorted(t for chunk in chunks for t in chunk)
        assert assigned == [
            (0, (0, 100)),
            (1, (100, 300)),
            (2, (300, 500)),
            (3, (500, 600)),
        ]
        costs = [
            sum((stop - start) ** 2 for _, (start, stop) in c) for c in chunks
        ]
        assert costs == [50_000, 50_000]

    def test_dominant_task_isolated(self):
        splits = nb.typed.List([0, 100, 101, 102, 103])
        chunks = scheduling.cost_based_chunking(splits, 2)
        assert sorted(len(c) for c in chunks) == [1, 3]

    def test_edge_cases_empty_single_worker_and_excess_workers(self):
        assert scheduling.cost_based_chunking(nb.typed.List([0]), 4) == []
        assert (
            len(scheduling.cost_based_chunking(nb.typed.List([0, 10, 20]), 1))
            == 1
        )
        assert (
            len(scheduling.cost_based_chunking(nb.typed.List([0, 10, 20]), 5))
            == 2
        )
