"""Unit tests for falcon.cluster.distance."""

import os
import tempfile

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from falcon.cluster import distance, scheduling
from falcon.types import SpectrumTuple


def _random_spectra(n, rng, n_peaks=30, mz_max=200.0):
    """Generate random L2-normalized SpectrumTuples."""
    specs = []
    for _ in range(n):
        mz = np.sort(rng.uniform(0.0, mz_max, n_peaks)).astype(np.float32)
        intensity = rng.random(n_peaks).astype(np.float32)
        intensity /= np.linalg.norm(intensity)
        specs.append(
            SpectrumTuple(float(rng.uniform(100, 1000)), 1, mz, intensity)
        )
    return specs


class TestCondensedIndex:
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_matches_scipy_squareform_exhaustively(self, n):
        expected_condensed = np.arange(n * (n - 1) // 2)
        square = squareform(expected_condensed)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert distance.condensed_index(i, j, n) == square[i, j]

    def test_large_matrix_analytical_values(self):
        assert distance.condensed_index(10, 20, 100) == 954
        assert distance.condensed_index(50, 99, 100) == 3773

    @pytest.mark.parametrize(
        ("i", "j", "n", "match"),
        [
            (0, 1, 0, "greater than 0"),
            (0, 1, -1, "greater than 0"),
            (0, 0, 1, "No diagonal elements"),
            (2, 2, 5, "No diagonal elements"),
            (-1, 2, 5, "non-negative"),
            (2, -1, 5, "non-negative"),
            (2, 5, 5, "out of bounds"),
            (5, 2, 5, "out of bounds"),
        ],
    )
    def test_invalid_inputs_raise_value_error(self, i, j, n, match):
        with pytest.raises(ValueError, match=match):
            distance.condensed_index(i, j, n)


class TestPackSpectraCsr:
    def test_handles_empty_spectra_and_variable_lengths(self, spec_factory):
        empty_mz, empty_int, empty_off = distance.pack_spectra_csr(
            [
                spec_factory([], [], normalize=False),
                spec_factory([], [], normalize=False),
            ]
        )
        assert empty_mz.size == 0
        assert empty_int.size == 0
        np.testing.assert_array_equal(empty_off, [0, 0, 0])

        mz_flat, int_flat, offsets = distance.pack_spectra_csr(
            [
                spec_factory([100.0, 200.0], [0.6, 0.8], normalize=False),
                spec_factory([300.0], [1.0], normalize=False),
            ]
        )
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])


class TestComputeCondensedDistanceMatrix:
    def test_identical_orthogonal_and_intermediate_distances(self):
        i_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        i_b = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mz = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        s0 = SpectrumTuple(500.0, 2, mz, i_a)
        s1 = SpectrumTuple(500.0, 2, mz, i_b)
        s2 = SpectrumTuple(
            500.0,
            2,
            np.array([600.0, 700.0, 800.0], dtype=np.float32),
            i_a,
        )

        pdist = distance.compute_condensed_distance_matrix(
            [s0, s1, s2], fragment_mz_tol=0.05, min_matches=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (3,)
        np.testing.assert_allclose(pdist, [0.4, 1.0, 1.0], atol=1e-6)

    def test_min_matches_exact_threshold(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        d_pass = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=3
        )
        d_fail = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=4
        )
        assert d_pass[0] == pytest.approx(0.0, abs=1e-5)
        assert d_fail[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("n", [0, 1])
    def test_empty_and_single_spectrum_return_empty_vector(
        self, mock_spectra, n
    ):
        pdist = distance.compute_condensed_distance_matrix(
            mock_spectra(n), fragment_mz_tol=0.05, min_matches=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (0,)

    def test_backing_temp_file_unlinked_immediately(
        self, mock_spectra, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        pdist = distance.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=0
        )
        assert list(tmp_path.iterdir()) == []
        assert pdist[0] == pytest.approx(0.0, abs=1e-5)


class TestTiledCondensed:
    def test_open_shared_condensed_persists_until_caller_unlinks(self):
        path, mm = distance.open_shared_condensed(5)
        try:
            assert os.path.isfile(path)
            assert mm.dtype == np.float32
            assert mm.shape == (10,)
            mm[3] = 0.25
            mm.flush()
            del mm

            reopened = np.lib.format.open_memmap(path, mode="r")
            assert reopened[3] == pytest.approx(0.25)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("min_matches", [0, 3])
    @pytest.mark.parametrize("n_tiles", [1, 7])
    def test_tiled_rows_bit_identical_to_full_matrix(
        self, min_matches, n_tiles
    ):
        rng = np.random.default_rng(0)
        spectra = _random_spectra(200, rng)
        frag = 0.05
        n = len(spectra)

        ref = np.array(
            distance.compute_condensed_distance_matrix(
                spectra, frag, min_matches
            )
        )
        assert (ref < 0.99).any()

        out = np.full(n * (n - 1) // 2, np.nan, np.float32)
        bounds = scheduling.interval_row_bounds(n, n_tiles)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        assert all(
            bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)
        )
        for r0, r1 in bounds:
            distance.fill_condensed_rows(
                out, spectra, r0, r1, frag, min_matches
            )

        assert not np.isnan(out).any()
        assert np.array_equal(out, ref)

    def test_empty_row_band_leaves_buffer_untouched(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        out = np.array([9.0], dtype=np.float32)
        distance.fill_condensed_rows(out, [spec1, spec2], 1, 1, 0.05, 0)
        assert out[0] == pytest.approx(9.0)
