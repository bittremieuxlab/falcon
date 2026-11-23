"""Unit tests for falcon.cluster.distance_matrix."""

import os
import tempfile

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from falcon.cluster import cluster, distance_matrix, similarity


def _random_spectra(n, rng, n_peaks=30, mz_max=200.0):
    """Generate random L2-normalized SpectrumTuples."""
    specs = []
    for _ in range(n):
        mz = np.sort(rng.uniform(0.0, mz_max, n_peaks)).astype(np.float32)
        intensity = rng.random(n_peaks).astype(np.float32)
        intensity /= np.linalg.norm(intensity)
        specs.append(
            similarity.SpectrumTuple(
                float(rng.uniform(100, 1000)), 1, mz, intensity
            )
        )
    return specs


# ----------------------------------------------------------------------
# condensed_index
# ----------------------------------------------------------------------


class TestCondensedIndex:
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_matches_scipy_squareform_exhaustively(self, n):
        """Condensed index matches SciPy's upper-triangle layout."""
        expected_condensed = np.arange(n * (n - 1) // 2)
        square = squareform(expected_condensed)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                assert distance_matrix.condensed_index(i, j, n) == square[i, j]

    def test_large_matrix_analytical_values(self):
        assert distance_matrix.condensed_index(10, 20, 100) == 954
        assert distance_matrix.condensed_index(50, 99, 100) == 3773

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
            distance_matrix.condensed_index(i, j, n)


# ----------------------------------------------------------------------
# compute_condensed_distance_matrix
# ----------------------------------------------------------------------


class TestComputeCondensedDistanceMatrix:
    def test_identical_orthogonal_and_intermediate_distances(self):
        i_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        i_b = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        mz = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        s0 = similarity.SpectrumTuple(500.0, 2, mz, i_a)
        s1 = similarity.SpectrumTuple(500.0, 2, mz, i_b)
        s2 = similarity.SpectrumTuple(
            500.0,
            2,
            np.array([600.0, 700.0, 800.0], dtype=np.float32),
            i_a,
        )

        pdist = distance_matrix.compute_condensed_distance_matrix(
            [s0, s1, s2], fragment_mz_tol=0.05, min_matches=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (3,)
        # (0, 1): dot = 0.6 -> dist = 0.4; (0, 2) and (1, 2) -> 1.0.
        np.testing.assert_allclose(pdist, [0.4, 1.0, 1.0], atol=1e-6)

    def test_min_matches_exact_threshold(self, spectrum_tuple_pair):
        spec1, spec2 = spectrum_tuple_pair  # 3 matching peaks
        d_pass = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=3
        )
        d_fail = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=4
        )
        assert d_pass[0] == pytest.approx(0.0, abs=1e-5)
        assert d_fail[0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("n", [0, 1])
    def test_empty_and_single_spectrum_return_empty_vector(
        self, mock_spectra, n
    ):
        pdist = distance_matrix.compute_condensed_distance_matrix(
            mock_spectra(n), fragment_mz_tol=0.05, min_matches=0
        )
        assert pdist.dtype == np.float32
        assert pdist.shape == (0,)

    def test_backing_temp_file_unlinked_immediately(
        self, spectrum_tuple_pair, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        spec1, spec2 = spectrum_tuple_pair
        pdist = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=0
        )
        assert list(tmp_path.iterdir()) == []
        assert pdist[0] == pytest.approx(0.0, abs=1e-5)


# ----------------------------------------------------------------------
# open_shared_condensed & _condensed_rows (tiled kernel)
# ----------------------------------------------------------------------


class TestTiledCondensed:
    def test_open_shared_condensed_persists_until_caller_unlinks(self):
        path, mm = distance_matrix.open_shared_condensed(5)
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
            distance_matrix.compute_condensed_distance_matrix(
                spectra, frag, min_matches
            )
        )
        assert (ref < 0.99).any()

        out = np.full(n * (n - 1) // 2, np.nan, np.float32)
        bounds = cluster._interval_row_bounds(n, n_tiles)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        assert all(
            bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)
        )
        for r0, r1 in bounds:
            distance_matrix._condensed_rows(
                out, spectra, r0, r1, frag, min_matches
            )

        assert not np.isnan(out).any()
        assert np.array_equal(out, ref)

    def test_empty_row_band_leaves_buffer_untouched(self, spectrum_tuple_pair):
        spec1, spec2 = spectrum_tuple_pair
        out = np.array([9.0], dtype=np.float32)
        distance_matrix._condensed_rows(out, [spec1, spec2], 1, 1, 0.05, 0)
        assert out[0] == pytest.approx(9.0)
