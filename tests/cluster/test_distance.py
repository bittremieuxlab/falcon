"""Unit tests for falcon.cluster.distance."""

import os
import tempfile

import numba as nb
import numpy as np
import pyarrow as pa
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


class TestPackAndExtractCsr:
    def test_pack_handles_empty_and_variable_length_spectra(
        self, spec_factory
    ):
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

    def test_pack_rejects_mismatched_peak_lengths(self, spec_factory):
        bad = spec_factory([100.0, 200.0], [1.0], normalize=False)
        with pytest.raises(ValueError, match="mismatched mz"):
            distance.pack_spectra_csr([bad])

    def test_extract_arrow_csr_validates_columns_and_types(self):
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
        mz_flat, int_flat, offsets = distance.extract_arrow_csr(table)
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
            distance.extract_arrow_csr(bad_type)

        mismatch = pa.table(
            {
                "mz": pa.array([[100.0, 200.0]], pa.list_(pa.float32())),
                "intensity": pa.array([[1.0]], pa.list_(pa.float32())),
            }
        )
        with pytest.raises(
            ValueError, match="Mismatched PyArrow list offsets"
        ):
            distance.extract_arrow_csr(mismatch)

    @pytest.mark.parametrize(
        ("mz_vals", "int_vals", "match"),
        [
            ([[100.0], None], [[1.0], []], "Null lists"),
            ([[100.0], []], [[1.0], None], "Null lists"),
            ([[100.0, None]], [[0.6, 0.8]], "Null values"),
            ([[100.0, 200.0]], [[0.6, None]], "Null values"),
        ],
    )
    def test_extract_arrow_csr_rejects_nulls(self, mz_vals, int_vals, match):
        table = pa.table(
            {
                "mz": pa.array(mz_vals, pa.list_(pa.float32())),
                "intensity": pa.array(int_vals, pa.list_(pa.float32())),
            }
        )
        with pytest.raises(ValueError, match=match):
            distance.extract_arrow_csr(table)

    def test_extract_arrow_csr_handles_sliced_and_chunked_arrays(self):
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
        mz_flat, int_flat, offsets = distance.extract_arrow_csr(table)
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])
        assert np.shares_memory(int_parent.values.to_numpy(), int_flat)

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
        c_mz, c_int, c_off = distance.extract_arrow_csr(chunked)
        np.testing.assert_allclose(c_mz, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(c_int, [1.0, 0.6, 0.8])
        np.testing.assert_array_equal(c_off, [0, 1, 3])


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

        mz_flat, int_flat, offsets = distance.pack_spectra_csr([s0, s1, s2])
        pdist_csr = distance.compute_condensed_distance_matrix_csr(
            mz_flat, int_flat, offsets, fragment_tol=0.05, min_matched_peaks=0
        )
        assert np.array_equal(pdist_csr, pdist)

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
        specs = mock_spectra(n)
        pdist = distance.compute_condensed_distance_matrix(
            specs, fragment_tol=0.05, min_matched_peaks=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (0,)

        mz_flat, int_flat, offsets = distance.pack_spectra_csr(specs)
        pdist_csr = distance.compute_condensed_distance_matrix_csr(
            mz_flat, int_flat, offsets, fragment_tol=0.05, min_matched_peaks=0
        )
        assert pdist_csr.dtype == np.float32
        assert pdist_csr.shape == (0,)

    def test_small_interval_uses_in_memory_and_large_unlinks_memmap(
        self, mock_spectra, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        spec1, spec2 = mock_spectra(2, n_peaks=3)

        pdist_ram = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_tol=0.5, min_matched_peaks=0
        )
        assert not isinstance(pdist_ram, np.memmap)
        assert list(tmp_path.iterdir()) == []
        assert pdist_ram[0] == pytest.approx(0.0, abs=1e-5)

        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 2)
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
        distance.close_memmap(np.array([0.0, 1.0], dtype=np.float32))

        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 2)
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
        monkeypatch.setattr(distance.os, "unlink", _fail_unlink)
        try:
            with pytest.raises(OSError, match="unlink denied"):
                distance.compute_condensed_distance_matrix(
                    mock_spectra(2, n_peaks=3),
                    fragment_tol=0.5,
                    min_matched_peaks=0,
                    temp_dir=tmp_path,
                )
            assert len(closed) == 1
        finally:
            for path in created:
                if os.path.exists(path):
                    real_unlink(path)

    def test_serial_and_parallel_dispatch_produce_identical_distances(
        self, spec_factory, monkeypatch
    ):
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
        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 2)

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

        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 100)
        with pytest.raises(RuntimeError, match="kernel boom"):
            distance.compute_condensed_distance_matrix(
                specs,
                fragment_tol=0.05,
                min_matched_peaks=0,
            )
        assert len(closed) == 1

    @pytest.mark.parametrize(
        ("mz_flat", "int_flat", "offsets", "match"),
        [
            (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
                "offsets must be 1D",
            ),
            (
                np.array([100.0, 200.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                "integer dtype",
            ),
            (
                np.array([100.0, 200.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([0, 2, 1], dtype=np.int64),
                "monotonically non-decreasing",
            ),
            (
                np.array([100.0, 200.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([0, 1, 2], dtype=np.int64),
                "equal length",
            ),
            (
                np.array([100.0, 200.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([0, 1, 5], dtype=np.int64),
                "exceed peak buffer",
            ),
        ],
    )
    def test_compute_condensed_csr_rejects_invalid_buffers(
        self, mz_flat, int_flat, offsets, match
    ):
        with pytest.raises(ValueError, match=match):
            distance.compute_condensed_distance_matrix_csr(
                mz_flat,
                int_flat,
                offsets,
                fragment_tol=0.05,
                min_matched_peaks=0,
            )


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

    @pytest.mark.parametrize("bad_n", [-1, 0, 1])
    def test_open_shared_condensed_rejects_small_n(self, bad_n):
        with pytest.raises(ValueError, match="at least 2 spectra"):
            distance.open_shared_condensed(bad_n)

    def test_open_shared_condensed_cleans_up_on_open_error(
        self, tmp_path, monkeypatch
    ):
        def _fail_open(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(np.lib.format, "open_memmap", _fail_open)
        with pytest.raises(OSError, match="disk full"):
            distance.open_shared_condensed(4, temp_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("min_matched_peaks", [0, 3])
    def test_tiled_rows_bit_identical_to_full_matrix(
        self, spec_factory, min_matched_peaks
    ):
        rng = np.random.default_rng(0)
        spectra = _random_spectra(60, rng, spec_factory)
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
        for n_tiles in (1, 7):
            out = np.full(n * (n - 1) // 2, np.nan, np.float32)
            bounds = scheduling.interval_row_bounds(n, n_tiles)
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n
            assert all(
                bounds[i][1] == bounds[i + 1][0]
                for i in range(len(bounds) - 1)
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

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("neg_r0", "out of bounds"),
            ("overshoot_r1", "out of bounds"),
            ("reversed_band", "out of bounds"),
            ("int_buf", "floating dtype"),
            ("short_buf", "1D with length"),
            ("empty_offsets", "offsets must be 1D"),
            ("float_offsets", "integer dtype"),
            ("non_monotonic_offsets", "monotonically"),
            ("mismatched_peaks", "equal length"),
            ("neg_offset", "exceed peak buffer"),
            ("overshoot_offset", "exceed peak buffer"),
        ],
    )
    def test_fill_condensed_rows_csr_validates_inputs(
        self, mock_spectra, case, match
    ):
        specs = mock_spectra(3, n_peaks=2)
        mz_flat, int_flat, offsets = distance.pack_spectra_csr(specs)
        buf = np.empty(3, dtype=np.float32)
        r0, r1 = 0, 3

        if case == "neg_r0":
            r0, r1 = -1, 2
        elif case == "overshoot_r1":
            r0, r1 = 0, 4
        elif case == "reversed_band":
            r0, r1 = 2, 1
        elif case == "int_buf":
            buf = np.empty(3, dtype=np.int32)
        elif case == "short_buf":
            buf = np.empty(2, dtype=np.float32)
        elif case == "empty_offsets":
            offsets = np.empty(0, dtype=np.int64)
            r1 = 0
        elif case == "float_offsets":
            offsets = offsets.astype(np.float64)
        elif case == "non_monotonic_offsets":
            offsets = np.array([0, 4, 2, 6], dtype=np.int64)
        elif case == "mismatched_peaks":
            mz_flat = mz_flat[:-1]
        elif case == "neg_offset":
            offsets = np.array([-1, 2, 4, 6], dtype=np.int64)
        elif case == "overshoot_offset":
            offsets = np.array([0, 2, 4, 99], dtype=np.int64)

        with pytest.raises(ValueError, match=match):
            distance.fill_condensed_rows_csr(
                buf, mz_flat, int_flat, offsets, r0, r1, 0.05, 0
            )
