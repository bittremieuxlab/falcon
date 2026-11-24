"""Unit tests for falcon.cluster.scheduling."""

import itertools

import numpy as np
import pytest

from falcon.cluster import refine, scheduling


def _make_spec(task_id: int, n: int) -> scheduling.IntervalSpec:
    """Create a synthetic IntervalSpec with ``n`` spectra."""
    return scheduling.IntervalSpec(
        task_id=task_id,
        row_ids=np.arange(n, dtype=np.int64),
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

    @pytest.mark.parametrize("bad_batch_size", [0, -1])
    def test_rejects_non_positive_batch_size(self, bad_batch_size):
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            scheduling.tile_threshold(bad_batch_size)

    @pytest.mark.parametrize("min_pairs", [1, 2, 3, 5, 6, 8, 6_000_000])
    def test_effective_tile_threshold_exact_two_tile_boundary(
        self, min_pairs, monkeypatch
    ):
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", min_pairs)
        thresh = scheduling.effective_tile_threshold(batch_size=2, n_workers=2)
        assert len(scheduling.plan_tile_bounds(thresh, n_workers=2)) >= 2
        assert len(scheduling.plan_tile_bounds(thresh - 1, n_workers=2)) == 1

    def test_effective_tile_threshold_disables_tiling_for_single_worker(self):
        assert scheduling.effective_tile_threshold(8, n_workers=1) == 9
        assert scheduling.effective_tile_threshold(8, n_workers=0) == 9
        # With default MIN_PAIRS_PER_TILE, small batch_size cannot split.
        assert scheduling.effective_tile_threshold(8, n_workers=4) > 8


class TestPrecursorMzSplits:
    def test_da_single_and_multiple_splits_with_exact_boundary(self):
        mzs = np.array([100.0, 100.5, 101.0, 101.5])
        splits = scheduling.precursor_mz_splits(mzs, 0.5, "Da", batch_size=100)
        assert isinstance(splits, np.ndarray)
        assert splits.dtype == np.int64
        np.testing.assert_array_equal(splits, [0, 4])

        mzs_split = np.array([100.0, 100.5, 101.5, 102.0])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(
                mzs_split, 0.5, "Da", batch_size=100
            ),
            [0, 2, 4],
        )

    def test_ppm_single_and_multiple_splits(self):
        mzs_one = np.array([100.0, 100.002, 100.003, 100.005])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs_one, 50, "ppm", batch_size=100),
            [0, 4],
        )

        mzs_two = np.array([100.0, 100.004, 101.0, 101.004])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs_two, 50, "ppm", batch_size=100),
            [0, 2, 4],
        )

    def test_true_even_chunking_for_internal_and_trailing_oversized_blocks(
        self,
    ):
        mzs = np.array([100.0, 100.2, 100.4, 100.5])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs, 0.5, "Da", batch_size=2),
            [0, 2, 4],
        )

        mzs2 = np.array([100.0, 101.0, 102.0, 103.0, 106.0, 107.0])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs2, 1.0, "Da", batch_size=3),
            [0, 2, 4, 6],
        )

        mzs_tail = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs_tail, 0.5, "Da", batch_size=5),
            [0, 4, 7],
        )

        mzs3 = np.array([100.0, 100.1, 100.2, 200.0, 200.1])
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(mzs3, 0.5, "Da", batch_size=3),
            [0, 3, 5],
        )

        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(
                np.array([100.0, 100.0, 100.0]), 0.5, "Da", batch_size=1
            ),
            [0, 1, 2, 3],
        )

    def test_empty_single_value_and_invalid_arguments(self):
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(
                np.array([]), 0.5, "Da", batch_size=100
            ),
            [0],
        )
        np.testing.assert_array_equal(
            scheduling.precursor_mz_splits(
                np.array([100.0]), 0.5, "Da", batch_size=100
            ),
            [0, 1],
        )
        with pytest.raises(ValueError, match="batch_size"):
            scheduling.precursor_mz_splits(
                np.array([100.0]), 0.5, "Da", batch_size=0
            )
        with pytest.raises(ValueError, match="precursor_tol_mode"):
            scheduling.precursor_mz_splits(
                np.array([100.0, 101.0]), 0.5, "da", batch_size=10
            )
        with pytest.raises(ValueError, match="precursor_mzs must be 1D"):
            scheduling.precursor_mz_splits(
                np.zeros((2, 2)), 0.5, "Da", batch_size=10
            )
        with pytest.raises(ValueError, match="strictly positive"):
            scheduling.precursor_mz_splits(
                np.array([0.0, 100.0]), 20.0, "ppm", batch_size=10
            )
        for bad_ppm in (
            np.array([100.0, np.nan]),
            np.array([100.0, np.inf]),
        ):
            with pytest.raises(ValueError, match="finite strictly positive"):
                scheduling.precursor_mz_splits(
                    bad_ppm, 20.0, "ppm", batch_size=10
                )

    def test_ppm_float32_input_matches_float64_linkage_1d_boundary(self):
        mzs_f32 = np.array([300.0, 300.00592], dtype=np.float32)
        ppm_f32 = float(
            np.float32((mzs_f32[1] - mzs_f32[0]) / mzs_f32[0])
            * np.float32(1e6)
        )
        ppm_f64 = float(refine.linkage_1d(mzs_f32, "ppm")[0, 2])
        assert ppm_f32 < ppm_f64
        tol = 0.5 * (ppm_f32 + ppm_f64)

        splits = scheduling.precursor_mz_splits(
            mzs_f32, tol, "ppm", batch_size=10
        )
        np.testing.assert_array_equal(splits, [0, 1, 2])
        assert (
            len(
                np.unique(
                    refine.cut_linkage(refine.linkage_1d(mzs_f32, "ppm"), tol)
                )
            )
            == 2
        )

    @pytest.mark.parametrize("mode", ["Da", "ppm"])
    @pytest.mark.parametrize("batch_size", [1, 3, 7, 25])
    def test_randomized_invariants_balance_and_gap_soundness(
        self, mode, batch_size
    ):
        rng = np.random.default_rng(42)
        tol = 0.5 if mode == "Da" else 50.0
        for _ in range(10):
            # Mix clustered regions, exact duplicates, and wide gaps.
            centers = rng.uniform(150.0, 1200.0, size=6)
            jitter_scale = 0.25 if mode == "Da" else 0.015
            vals = np.concatenate(
                [
                    c + rng.uniform(0.0, jitter_scale, size=rng.integers(1, 9))
                    for c in centers
                ]
            )
            vals = np.sort(np.concatenate([vals, vals[:3]]))
            n = len(vals)

            splits = scheduling.precursor_mz_splits(
                vals, tol, mode, batch_size
            )
            assert splits.dtype == np.int64
            assert splits[0] == 0
            assert splits[-1] == n
            lengths = np.diff(splits)
            assert np.all((lengths >= 1) & (lengths <= batch_size))

            # Independent reference: contiguous natural blocks separated
            # by gaps strictly exceeding tol.
            diffs = (
                np.diff(vals)
                if mode == "Da"
                else np.diff(vals) / vals[:-1] * 1e6
            )
            gap_indices = np.flatnonzero(diffs > tol) + 1
            block_bounds = [0, *gap_indices.tolist(), n]

            # Every natural gap must appear as a split boundary, and any
            # sub-splits inside a block must be balanced within <= 1.
            split_set = set(splits.tolist())
            assert set(block_bounds).issubset(split_set)
            for b_start, b_end in itertools.pairwise(block_bounds):
                sub = splits[(splits >= b_start) & (splits <= b_end)]
                sub_lens = np.diff(sub)
                expected_chunks = (
                    (b_end - b_start) + batch_size - 1
                ) // batch_size
                assert len(sub_lens) == expected_chunks
                assert int(sub_lens.max() - sub_lens.min()) <= 1


class TestIntervalRowBoundsAndTilePlanning:
    @pytest.mark.parametrize("n", [2, 3, 4, 10, 100])
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

    def test_small_interval_splits_without_off_by_one_collapse(self):
        assert scheduling.interval_row_bounds(4, 2) == [(0, 1), (1, 4)]
        assert scheduling.interval_row_bounds(6, 2) == [(0, 2), (2, 6)]

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

    def test_validates_bounds_and_plan_tile_bounds_conserves_cost(
        self, monkeypatch
    ):
        with pytest.raises(ValueError, match="n must be at least 2"):
            scheduling.interval_row_bounds(1, 1)
        with pytest.raises(ValueError, match="n_tiles must be at least 1"):
            scheduling.interval_row_bounds(4, 0)
        with pytest.raises(ValueError, match="n must be at least 2"):
            scheduling.plan_tile_bounds(1, 2)
        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            scheduling.plan_tile_bounds(4, 0)

        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 3)
        planned = scheduling.plan_tile_bounds(6, n_workers=4)
        assert len(planned) >= 2
        assert sum(cost for _, _, cost in planned) == pytest.approx(36.0)


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
        rows = [
            spectrum_row("s:g2", 700.02, retention_time=12.0),
            spectrum_row("s:n1", 500.00, retention_time=10.0),
            spectrum_row("s:g0", 700.00, retention_time=10.0),
            spectrum_row("s:n2", 500.01, retention_time=11.0),
            spectrum_row("s:g1", 700.01, retention_time=11.0),
        ]
        ds = lance_dataset(rows)
        plan = scheduling.prepare_bucket_chunks(
            ds,
            "2",
            da_clustering_factory(batch_size=10),
            num_chunks=4,
            threshold=3,
        )
        assert isinstance(plan, scheduling.BucketPlan)
        assert plan.n_spectra == 5
        assert list(plan.splits) == [0, 2, 5]
        assert len(plan.normal_chunks) == 1
        assert len(plan.normal_chunks[0]) == 1
        assert plan.normal_chunks[0][0].n_spectra == 2
        assert len(plan.giant_specs) == 1
        assert plan.giant_specs[0].n_spectra == 3
        assert isinstance(plan.giant_specs[0].row_ids, np.ndarray)

        giant_ids = (
            ds.take(
                indices=plan.giant_specs[0].row_ids.tolist(),
                columns=["identifier"],
            )
            .to_pandas()["identifier"]
            .tolist()
        )
        assert giant_ids == ["s:g0", "s:g1", "s:g2"]

    def test_giant_interval_does_not_skew_normal_chunk_lpt_balance(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        rows = [
            spectrum_row(f"g:{i}", 100.0 + 0.01 * i, retention_time=float(i))
            for i in range(10)
        ]
        for group_idx, base_mz in enumerate((200.0, 300.0, 400.0, 500.0)):
            for i in range(4):
                rows.append(
                    spectrum_row(
                        f"n:{group_idx}:{i}",
                        base_mz + 0.01 * i,
                        retention_time=float(i),
                    )
                )
        ds = lance_dataset(rows)

        plan = scheduling.prepare_bucket_chunks(
            ds,
            "2",
            da_clustering_factory(batch_size=20),
            num_chunks=2,
            threshold=8,
        )
        assert plan.n_spectra == 26
        assert list(plan.splits) == [0, 10, 14, 18, 22, 26]
        assert len(plan.giant_specs) == 1
        assert plan.giant_specs[0].n_spectra == 10
        assert len(plan.normal_chunks) == 2
        assert [len(chunk) for chunk in plan.normal_chunks] == [2, 2]
        assert [
            sum(s.cost for s in chunk) for chunk in plan.normal_chunks
        ] == [32, 32]

    def test_empty_and_all_giant_buckets(
        self, lance_dataset, spectrum_row, da_clustering_factory
    ):
        clust = da_clustering_factory(batch_size=10)
        ds_empty = lance_dataset([], name="empty.lance")
        plan_empty = scheduling.prepare_bucket_chunks(
            ds_empty, "2", clust, num_chunks=2, threshold=4
        )
        assert plan_empty.n_spectra == 0
        assert list(plan_empty.splits) == [0]
        assert plan_empty.normal_chunks == []
        assert plan_empty.giant_specs == []

        ds_giant = lance_dataset(
            [
                spectrum_row(f"g:{i}", 500.0 + 0.01 * i, retention_time=i)
                for i in range(4)
            ],
            name="giant.lance",
        )
        plan_g = scheduling.prepare_bucket_chunks(
            ds_giant, "2", clust, num_chunks=2, threshold=4
        )
        assert plan_g.n_spectra == 4
        assert list(plan_g.splits) == [0, 4]
        assert plan_g.normal_chunks == []
        assert len(plan_g.giant_specs) == 1
        assert plan_g.giant_specs[0].n_spectra == 4
