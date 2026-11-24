"""Unit tests for falcon.cluster.tiling."""

import os

import numpy as np
import pyarrow as pa
import pytest

from falcon.cluster import distance, interval, scheduling, tiling


class TestGiantMemmapStore:
    def test_allocate_eager_release_and_context_exit_cleanup(self):
        with tiling.GiantMemmapStore() as store:
            p0 = store.allocate(0, 4)
            p1 = store.allocate(1, 4)
            assert os.path.isfile(p0)
            assert os.path.isfile(p1)

            with pytest.raises(ValueError, match="already allocated"):
                store.allocate(0, 4)

            store.release(0)
            assert not os.path.exists(p0)
            assert os.path.isfile(p1)
            # Releasing again is an idempotent no-op.
            store.release(0)
        assert not os.path.exists(p1)

    def test_temp_dir_and_registers_path_before_close_memmap(
        self, tmp_path, monkeypatch
    ):
        with tiling.GiantMemmapStore(temp_dir=tmp_path) as store:
            path = store.allocate(0, 4)
            assert os.path.dirname(path) == str(tmp_path)
            assert os.path.isfile(path)
        assert not os.path.exists(path)
        assert list(tmp_path.iterdir()) == []

        real_close = tiling.close_memmap

        def _fail_close(mm):
            real_close(mm)
            raise RuntimeError("close boom")

        monkeypatch.setattr(tiling, "close_memmap", _fail_close)
        with (
            pytest.raises(RuntimeError, match="close boom"),
            tiling.GiantMemmapStore(temp_dir=tmp_path) as store,
        ):
            store.allocate(0, 4)
        # Path was registered before close_memmap so __exit__ unlinked.
        assert list(tmp_path.iterdir()) == []

    def test_cleans_up_on_exception(self):
        created: list[str] = []

        def _allocate_and_fail() -> None:
            with tiling.GiantMemmapStore() as store:
                path = store.allocate(0, 4)
                created.append(path)
                assert os.path.isfile(path)
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _allocate_and_fail()
        assert len(created) == 1
        assert not os.path.exists(created[0])


class TestGiantTiling:
    def test_multi_tile_distance_populates_exact_memmap_and_decoupled_exports(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        assert "GiantInterval" not in tiling.__all__
        assert "finalize_giant" not in tiling.__all__

        peaks_b = [400.0, 500.0, 600.0]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.00, retention_time=10.0),
                spectrum_row("s:1", 500.01, retention_time=11.0),
                spectrum_row("s:2", 500.02, mz=peaks_b, retention_time=12.0),
                spectrum_row("s:3", 500.03, mz=peaks_b, retention_time=13.0),
                spectrum_row("s:4", 500.04, retention_time=14.0),
                spectrum_row("s:5", 800.00, retention_time=15.0),
            ]
        )
        clust = da_clustering_factory(batch_size=8)
        plan = scheduling.prepare_bucket_chunks(
            ds, "2", clust, num_chunks=4, threshold=4
        )
        assert plan.n_spectra == 6
        assert len(plan.giant_specs) == 1

        # Lower MIN_PAIRS_PER_TILE so n=5 splits into >= 2 disjoint tiles.
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 3)
        with tiling.GiantMemmapStore() as store:
            memmap_paths, tile_tasks = tiling.allocate_giant_tiles(
                [(ds, plan.giant_specs[0])],
                clust.fragment_tol,
                clust.min_matched_peaks,
                n_workers=4,
                memmap_store=store,
            )
            assert len(memmap_paths) == 1
            assert len(tile_tasks) >= 2
            assert sum(t.tile_cost for t in tile_tasks) == pytest.approx(
                plan.giant_specs[0].cost
            )
            assert sum(t.tile_spectra for t in tile_tasks) == pytest.approx(
                plan.giant_specs[0].n_spectra
            )
            for tile in tile_tasks:
                done = tiling.tile_distance(tile)
                assert isinstance(done, tiling.TileDone)
                assert done.giant_idx == tile.giant_idx
                assert done.tile_cost == pytest.approx(tile.tile_cost)

            giant_df = ds.take(
                indices=pa.array(plan.giant_specs[0].row_ids)
            ).to_pandas()
            spec_tuples, _, _ = interval.interval_spectrum_tuples(
                giant_df, already_sorted=True
            )
            expected_pdist = distance.compute_condensed_distance_matrix(
                spec_tuples, clust.fragment_tol, clust.min_matched_peaks
            )
            tiled_pdist = np.lib.format.open_memmap(memmap_paths[0], mode="r")
            try:
                assert np.array_equal(tiled_pdist, expected_pdist)
            finally:
                distance.close_memmap(tiled_pdist)
