"""Unit tests for falcon.cluster.distance."""

import os
import tempfile

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from falcon.cluster import distance, scheduling


def _random_spectra(n, rng, spec_factory, n_peaks=30, mz_max=200.0):
    """Generate random L2-normalized SpectrumTuples via spec_factory."""
    return [
        spec_factory(
            np.sort(rng.uniform(0.0, mz_max, n_peaks)),
            rng.random(n_peaks),
            precursor_mz=float(rng.uniform(100.0, 1000.0)),
            charge=1,
        )
        for _ in range(n)
    ]


class TestCondensedIndex:
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_matches_scipy_squareform_and_symmetry_exhaustively(self, n):
        expected_condensed = np.arange(n * (n - 1) // 2)
        square = squareform(expected_condensed)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                idx = distance.condensed_index(i, j, n)
                assert idx == square[i, j]
                assert idx == distance.condensed_index(j, i, n)

    def test_large_matrix_analytical_values(self):
        assert distance.condensed_index(10, 20, 100) == 954
        assert distance.condensed_index(20, 10, 100) == 954
        assert distance.condensed_index(50, 99, 100) == 3773
        assert distance.condensed_index(99, 50, 100) == 3773


class TestPackSpectraCsr:
    def test_handles_empty_and_variable_length_spectra(self, spec_factory):
        empty_mz, empty_int, empty_off = distance.pack_spectra_csr([])
        assert empty_mz.size == 0
        assert empty_int.size == 0
        np.testing.assert_array_equal(empty_off, [0])

        zero_mz, zero_int, zero_off = distance.pack_spectra_csr(
            [
                spec_factory([], [], normalize=False),
                spec_factory([], [], normalize=False),
            ]
        )
        assert zero_mz.size == 0
        assert zero_int.size == 0
        np.testing.assert_array_equal(zero_off, [0, 0, 0])

        mz_flat, int_flat, offsets = distance.pack_spectra_csr(
            [
                spec_factory([100.0, 200.0], [0.6, 0.8], normalize=False),
                spec_factory([300.0], [1.0], normalize=False),
            ]
        )
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])

    def test_rejects_mismatched_peak_lengths(self, spec_factory):
        bad = spec_factory([100.0, 200.0], [1.0], normalize=False)
        with pytest.raises(ValueError, match="mismatched mz"):
            distance.pack_spectra_csr([bad])


class TestComputeCondensedDistanceMatrix:
    def test_identical_orthogonal_and_intermediate_distances(
        self, spec_factory
    ):
        s0 = spec_factory(
            [100.0, 200.0, 300.0], [1.0, 0.0, 0.0], normalize=False
        )
        s1 = spec_factory(
            [100.0, 200.0, 300.0], [0.6, 0.8, 0.0], normalize=False
        )
        s2 = spec_factory(
            [600.0, 700.0, 800.0], [1.0, 0.0, 0.0], normalize=False
        )

        pdist = distance.compute_condensed_distance_matrix(
            [s0, s1, s2], fragment_tol=0.05, min_matched_peaks=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (3,)
        np.testing.assert_allclose(pdist, [0.4, 1.0, 1.0], atol=1e-6)

    def test_min_matched_peaks_exact_threshold(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        d_pass = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_tol=0.5, min_matched_peaks=3
        )
        d_fail = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_tol=0.5, min_matched_peaks=4
        )
        assert d_pass[0] == pytest.approx(0.0, abs=1e-5)
        assert d_fail[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("n", [0, 1])
    def test_empty_and_single_spectrum_return_empty_vector(
        self, mock_spectra, n
    ):
        pdist = distance.compute_condensed_distance_matrix(
            mock_spectra(n), fragment_tol=0.05, min_matched_peaks=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (0,)

    def test_small_interval_uses_in_memory_and_large_unlinks_memmap(
        self, mock_spectra, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        spec1, spec2 = mock_spectra(2, n_peaks=3)

        # Small interval stays in RAM with no backing temp file.
        pdist_ram = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_tol=0.5, min_matched_peaks=0
        )
        assert not isinstance(pdist_ram, np.memmap)
        assert list(tmp_path.iterdir()) == []
        assert pdist_ram[0] == pytest.approx(0.0, abs=1e-5)

        # At/above threshold uses a memmap unlinked immediately on POSIX.
        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 2)
        pdist_mm = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_tol=0.5, min_matched_peaks=0
        )
        assert isinstance(pdist_mm, np.memmap)
        assert list(tmp_path.iterdir()) == []
        assert pdist_mm[0] == pytest.approx(0.0, abs=1e-5)
        distance.close_memmap(pdist_mm)

    def test_close_memmap_on_plain_ndarray_and_unlink_failure_cleanup(
        self, mock_spectra, tmp_path, monkeypatch
    ):
        # Plain ndarrays lack _mmap; close_memmap is a safe no-op.
        distance.close_memmap(np.array([0.0, 1.0], dtype=np.float32))

        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 2)
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        closed: list[np.ndarray] = []
        real_close = distance.close_memmap
        created: list[str] = []
        real_unlink = os.unlink

        def _spy_close(mm):
            closed.append(mm)
            real_close(mm)

        def _fail_unlink(path):
            created.append(str(path))
            raise OSError("unlink denied")

        monkeypatch.setattr(distance, "close_memmap", _spy_close)
        monkeypatch.setattr(os, "unlink", _fail_unlink)
        try:
            with pytest.raises(OSError, match="unlink denied"):
                distance.compute_condensed_distance_matrix(
                    mock_spectra(2, n_peaks=3),
                    fragment_tol=0.5,
                    min_matched_peaks=0,
                )
            assert len(closed) == 1
        finally:
            for path in created:
                if os.path.exists(path):
                    real_unlink(path)

    def test_serial_and_parallel_dispatch_produce_identical_distances(
        self, spec_factory, monkeypatch
    ):
        import numba as nb

        rng = np.random.default_rng(11)
        spectra = _random_spectra(25, rng, spec_factory, n_peaks=15)
        monkeypatch.setattr(nb, "get_num_threads", lambda: 1)
        serial = distance.compute_condensed_distance_matrix(
            spectra, fragment_tol=0.05, min_matched_peaks=1
        )
        monkeypatch.setattr(nb, "get_num_threads", lambda: 4)
        parallel = distance.compute_condensed_distance_matrix(
            spectra, fragment_tol=0.05, min_matched_peaks=1
        )
        assert np.array_equal(serial, parallel)

    def test_memmap_honors_temp_dir_and_closes_on_kernel_error(
        self, mock_spectra, tmp_path, monkeypatch
    ):
        specs = mock_spectra(4, n_peaks=3)
        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 2)

        seen_dirs: list[str | None] = []
        real_open = distance.open_shared_condensed

        def _spy_open(n, *, temp_dir=None):
            seen_dirs.append(None if temp_dir is None else os.fspath(temp_dir))
            return real_open(n, temp_dir=temp_dir)

        monkeypatch.setattr(distance, "open_shared_condensed", _spy_open)
        pdist = distance.compute_condensed_distance_matrix(
            specs, fragment_tol=0.05, min_matched_peaks=0, temp_dir=tmp_path
        )
        try:
            assert seen_dirs == [str(tmp_path)]
            assert list(tmp_path.glob("*.npy")) == []
        finally:
            distance.close_memmap(pdist)

        closed: list[object] = []
        real_close = distance.close_memmap

        def _spy_close(arr):
            closed.append(arr)
            real_close(arr)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("kernel boom")

        monkeypatch.setattr(distance, "close_memmap", _spy_close)
        monkeypatch.setattr(distance, "_condensed_rows_csr", _boom)
        monkeypatch.setattr(
            distance, "_condensed_distance_matrix_parallel", _boom
        )
        with pytest.raises(RuntimeError, match="kernel boom"):
            distance.compute_condensed_distance_matrix(
                specs,
                fragment_tol=0.05,
                min_matched_peaks=0,
                temp_dir=tmp_path,
            )
        assert len(closed) == 1
        assert list(tmp_path.glob("*.npy")) == []

        # Non-memmap kernel failure re-raises without calling close_memmap.
        monkeypatch.setattr(distance, "_MEMMAP_MIN_SPECTRA", 100)
        with pytest.raises(RuntimeError, match="kernel boom"):
            distance.compute_condensed_distance_matrix(
                specs,
                fragment_tol=0.05,
                min_matched_peaks=0,
            )
        assert len(closed) == 1


class TestTiledCondensed:
    def test_open_shared_condensed_persists_until_caller_unlinks(
        self, tmp_path
    ):
        path, mm = distance.open_shared_condensed(5, temp_dir=tmp_path)
        try:
            assert os.path.dirname(path) == str(tmp_path)
            assert os.path.isfile(path)
            assert mm.dtype == np.float32
            assert mm.shape == (10,)
            mm[3] = 0.25
            mm.flush()
            distance.close_memmap(mm)

            reopened = np.lib.format.open_memmap(path, mode="r")
            try:
                assert reopened[3] == pytest.approx(0.25)
            finally:
                distance.close_memmap(reopened)
        finally:
            os.unlink(path)

    def test_open_shared_condensed_validates_n_and_cleans_up_on_error(
        self, tmp_path, monkeypatch
    ):
        for bad_n in (-1, 0, 1):
            with pytest.raises(ValueError, match="at least 2 spectra"):
                distance.open_shared_condensed(bad_n)

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

        def _fail_open(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(np.lib.format, "open_memmap", _fail_open)
        with pytest.raises(OSError, match="disk full"):
            distance.open_shared_condensed(4)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("min_matched_peaks", [0, 3])
    @pytest.mark.parametrize("n_tiles", [1, 7])
    def test_tiled_rows_bit_identical_to_full_matrix(
        self, spec_factory, min_matched_peaks, n_tiles
    ):
        rng = np.random.default_rng(0)
        spectra = _random_spectra(200, rng, spec_factory)
        frag = 0.05
        n = len(spectra)

        ref = np.array(
            distance.compute_condensed_distance_matrix(
                spectra,
                fragment_tol=frag,
                min_matched_peaks=min_matched_peaks,
            )
        )
        assert (ref < 0.99).any()

        mz_flat, int_flat, offsets = distance.pack_spectra_csr(spectra)
        out = np.full(n * (n - 1) // 2, np.nan, np.float32)
        bounds = scheduling.interval_row_bounds(n, n_tiles)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        assert all(
            bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)
        )
        for r0, r1 in bounds:
            distance.fill_condensed_rows_csr(
                out,
                mz_flat,
                int_flat,
                offsets,
                r0,
                r1,
                fragment_tol=frag,
                min_matched_peaks=min_matched_peaks,
            )

        assert not np.isnan(out).any()
        assert np.array_equal(out, ref)

    def test_empty_row_band_leaves_buffer_untouched(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        out = np.array([9.0], dtype=np.float32)
        mz_flat, int_flat, offsets = distance.pack_spectra_csr([spec1, spec2])
        distance.fill_condensed_rows_csr(
            out, mz_flat, int_flat, offsets, 1, 1, 0.05, 0
        )
        assert out[0] == pytest.approx(9.0)

    def test_fill_condensed_rows_csr_validates_inputs(self, mock_spectra):
        specs = mock_spectra(3, n_peaks=2)
        mz_flat, int_flat, offsets = distance.pack_spectra_csr(specs)
        valid_buf = np.empty(3, dtype=np.float32)

        with pytest.raises(ValueError, match="out of bounds"):
            distance.fill_condensed_rows_csr(
                valid_buf, mz_flat, int_flat, offsets, -1, 2, 0.05, 0
            )
        with pytest.raises(ValueError, match="out of bounds"):
            distance.fill_condensed_rows_csr(
                valid_buf, mz_flat, int_flat, offsets, 0, 4, 0.05, 0
            )
        with pytest.raises(ValueError, match="1D with length"):
            distance.fill_condensed_rows_csr(
                np.empty(2, dtype=np.float32),
                mz_flat,
                int_flat,
                offsets,
                0,
                3,
                0.05,
                0,
            )
        with pytest.raises(ValueError, match="offsets must be 1D"):
            distance.fill_condensed_rows_csr(
                valid_buf, mz_flat, int_flat, np.empty(0), 0, 0, 0.05, 0
            )
        with pytest.raises(ValueError, match="integer dtype"):
            distance.fill_condensed_rows_csr(
                valid_buf,
                mz_flat,
                int_flat,
                offsets.astype(np.float64),
                0,
                3,
                0.05,
                0,
            )
        with pytest.raises(ValueError, match="monotonically"):
            distance.fill_condensed_rows_csr(
                valid_buf,
                mz_flat,
                int_flat,
                np.array([0, 4, 2, 6], dtype=np.int64),
                0,
                3,
                0.05,
                0,
            )
        with pytest.raises(ValueError, match="equal length"):
            distance.fill_condensed_rows_csr(
                valid_buf, mz_flat[:-1], int_flat, offsets, 0, 3, 0.05, 0
            )
        with pytest.raises(ValueError, match="exceed peak buffer"):
            distance.fill_condensed_rows_csr(
                valid_buf,
                mz_flat,
                int_flat,
                np.array([-1, 2, 4, 6], dtype=np.int64),
                0,
                3,
                0.05,
                0,
            )
        bad_offsets = np.array([0, 2, 4, 99], dtype=np.int64)
        with pytest.raises(ValueError, match="exceed peak buffer"):
            distance.fill_condensed_rows_csr(
                valid_buf, mz_flat, int_flat, bad_offsets, 0, 3, 0.05, 0
            )
