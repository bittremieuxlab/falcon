"""Unit tests for falcon.cluster.tiling."""

import os

import numpy as np
import pyarrow as pa
import pytest

from falcon.cluster import distance, engine, scheduling, tiling
from falcon.config import MedoidConsensusConfig
from falcon.types import ClusterResult


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


class TestExtractArrowCsr:
    def test_extracts_int64_offsets_and_validates_columns(self):
        table = pa.table(
            {
                "mz": pa.array(
                    [[100.0, 200.0], [300.0]], pa.list_(pa.float32())
                ),
                "intensity": pa.array(
                    [[0.6, 0.8], [1.0]], pa.list_(pa.float32())
                ),
            }
        )
        mz_flat, int_flat, offsets = tiling._extract_arrow_csr(table)
        assert mz_flat.dtype == np.float32
        assert int_flat.dtype == np.float32
        assert offsets.dtype == np.int64
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])

        bad_type = pa.table(
            {"mz": pa.array([1.0]), "intensity": pa.array([1.0])}
        )
        with pytest.raises(TypeError, match="list arrays"):
            tiling._extract_arrow_csr(bad_type)

        mismatch = pa.table(
            {
                "mz": pa.array([[100.0, 200.0]], pa.list_(pa.float32())),
                "intensity": pa.array([[1.0]], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(
            ValueError, match="Mismatched PyArrow list offsets"
        ):
            tiling._extract_arrow_csr(mismatch)

    def test_rejects_list_level_and_element_level_nulls(self):
        # Masked list-level null with offsets matching the sibling [].
        null_mz_list = pa.table(
            {
                "mz": pa.array([[100.0], None], pa.list_(pa.float32())),
                "intensity": pa.array([[1.0], []], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(ValueError, match="Null lists"):
            tiling._extract_arrow_csr(null_mz_list)

        null_int_list = pa.table(
            {
                "mz": pa.array([[100.0], []], pa.list_(pa.float32())),
                "intensity": pa.array([[1.0], None], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(ValueError, match="Null lists"):
            tiling._extract_arrow_csr(null_int_list)

        null_mz_elem = pa.table(
            {
                "mz": pa.array([[100.0, None]], pa.list_(pa.float32())),
                "intensity": pa.array([[0.6, 0.8]], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(ValueError, match="Null values"):
            tiling._extract_arrow_csr(null_mz_elem)

        null_int_elem = pa.table(
            {
                "mz": pa.array([[100.0, 200.0]], pa.list_(pa.float32())),
                "intensity": pa.array([[0.6, None]], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(ValueError, match="Null values"):
            tiling._extract_arrow_csr(null_int_elem)

    def test_handles_sliced_list_arrays_with_distinct_base_offsets(self):
        # Slice mz from index 1 (offset 2) and intensity from index 2
        # (offset 3), with a parent null outside the sliced region.
        mz_parent = pa.array(
            [[None, 50.0], [100.0, 200.0], [300.0]], pa.list_(pa.float32())
        )
        int_parent = pa.array(
            [[0.1], [0.2, 0.3], [0.6, 0.8], [1.0]], pa.list_(pa.float32())
        )
        table = pa.table(
            {
                "mz": mz_parent.slice(1, 2),
                "intensity": int_parent.slice(2, 2),
            }
        )
        mz_flat, int_flat, offsets = tiling._extract_arrow_csr(table)
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])
        assert np.shares_memory(int_parent.values.to_numpy(), int_flat)

        # Multi-chunk columns combine cleanly too.
        chunked = pa.table(
            {
                "mz": pa.chunked_array(
                    [
                        pa.array([[100.0]], pa.list_(pa.float32())),
                        pa.array([[200.0, 300.0]], pa.list_(pa.float32())),
                    ]
                ),
                "intensity": pa.chunked_array(
                    [
                        pa.array([[1.0]], pa.list_(pa.float32())),
                        pa.array([[0.6, 0.8]], pa.list_(pa.float32())),
                    ]
                ),
            }
        )
        c_mz, c_int, c_off = tiling._extract_arrow_csr(chunked)
        np.testing.assert_allclose(c_mz, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(c_int, [1.0, 0.6, 0.8])
        np.testing.assert_array_equal(c_off, [0, 1, 3])


class TestGiantTiling:
    def test_multi_tile_distance_populates_memmap_and_has_no_engine_exports(
        self, lance_dataset, spectrum_row, da_clustering_factory, monkeypatch
    ):
        assert "GiantInterval" not in tiling.__all__
        assert "finalize_giant" not in tiling.__all__

        peaks_a = [100.0, 200.0, 300.0]
        peaks_b = [400.0, 500.0, 600.0]
        inten = [1.0, 0.8, 0.6]
        ds = lance_dataset(
            [
                spectrum_row("s:0", 500.00, peaks_a, inten, 2, 10.0),
                spectrum_row("s:1", 500.01, peaks_a, inten, 2, 11.0),
                spectrum_row("s:2", 500.02, peaks_b, inten, 2, 12.0),
                spectrum_row("s:3", 500.03, peaks_b, inten, 2, 13.0),
                spectrum_row("s:4", 500.04, peaks_a, inten, 2, 14.0),
                spectrum_row("s:5", 800.00, peaks_a, inten, 2, 15.0),
            ]
        )
        clust = da_clustering_factory(batch_size=8)
        plan = scheduling.prepare_bucket_chunks(
            ds, "2", clust, num_chunks=4, threshold=4
        )
        assert plan.n_spectra == 6
        assert len(plan.giant_specs) == 1
        assert len(plan.normal_chunks) == 1

        # Lower MIN_PAIRS_PER_TILE so n=5 splits into >= 2 disjoint tiles.
        monkeypatch.setattr(scheduling, "MIN_PAIRS_PER_TILE", 3)
        options_on = engine.ClusteringOptions(
            clustering=clust,
            consensus=MedoidConsensusConfig(),
            compute_representatives=True,
        )
        with tiling.GiantMemmapStore() as store:
            memmap_paths, tile_tasks = tiling.allocate_giant_tiles(
                [(0, ds, plan.giant_specs[0])],
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

            giant = engine.GiantInterval(
                giant_idx=0,
                bucket_id=0,
                dataset=ds,
                spec=plan.giant_specs[0],
                memmap_path=memmap_paths[0],
                options=options_on,
            )
            giant_done = engine.finalize_giant(giant)
            assert isinstance(giant_done, engine.GiantDone)
            assert (giant_done.bucket_id, giant_done.task_id) == (
                0,
                plan.giant_specs[0].task_id,
            )
            assert isinstance(giant_done.result, ClusterResult)
            assert len(giant_done.result.representatives) == 2

            # Verify tiled memmap matches direct distance calculation.
            giant_df = ds.take(
                indices=pa.array(plan.giant_specs[0].row_ids)
            ).to_pandas()
            spec_tuples, _, _ = engine.interval_spectrum_tuples(
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

            normal_res = engine.cluster_chunk(
                plan.normal_chunks[0], ds, options_on
            )
            bucket_res = engine.finalize_bucket(
                plan.n_spectra,
                plan.splits,
                [*normal_res, (giant_done.task_id, giant_done.result)],
                "2",
            )
            assert len(bucket_res.labels) == 6
            assert [int(r.cluster_id) for r in bucket_res.representatives] == [
                0,
                1,
                2,
            ]
