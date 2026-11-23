"""Unit tests for falcon.cluster.similarity."""

import math

import numpy as np
import pandas as pd
import pytest

from falcon.cluster import similarity


def _spec(mz, intensity, normalize=True, precursor_mz=500.0, charge=2):
    mz_arr = np.asarray(mz, dtype=np.float32)
    int_arr = np.asarray(intensity, dtype=np.float32)
    if normalize and int_arr.size > 0:
        int_arr = int_arr / np.linalg.norm(int_arr)
    return similarity.SpectrumTuple(
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mz_arr,
        intensity=int_arr,
    )


class TestCosineFast:
    def test_identical_spectra(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        score, matched = similarity.cosine_fast(spec1, spec2, 0.5)
        assert score == pytest.approx(1.0)
        assert matched == 3

    def test_hand_computed_different_intensities(self):
        a = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        b = np.array([0.7, 0.6, 0.5], dtype=np.float32)
        expected = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
        spec1 = _spec([100.0, 101.0, 102.0], a)
        spec2 = _spec([100.0, 101.0, 102.0], b)
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 3
        assert score == pytest.approx(expected, rel=1e-5)

    def test_partial_overlap(self):
        spec1 = _spec([100.0, 200.0, 300.0], [1.0, 1.0, 1.0])
        spec2 = _spec([100.0, 200.0, 999.0], [1.0, 1.0, 1.0])
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 2
        assert score == pytest.approx(2.0 / 3.0, rel=1e-5)

    def test_tolerance_boundary_is_inclusive_and_zero_tol_exact(self):
        # Use binary-exact values (0.25 Da difference).
        spec1 = _spec([100.0], [1.0])
        spec2 = _spec([100.25], [1.0])

        assert similarity.cosine_fast(spec1, spec2, 0.25) == (
            pytest.approx(1.0),
            1,
        )
        assert similarity.cosine_fast(spec1, spec2, 0.24) == (0.0, 0)
        assert similarity.cosine_fast(spec1, spec1, 0.0) == (
            pytest.approx(1.0),
            1,
        )
        assert similarity.cosine_fast(spec1, spec2, 0.0) == (0.0, 0)

    def test_greedy_one_to_one_many_to_one_selects_strongest_peak(self):
        """Two competing peaks match only the stronger partner."""
        # Normalized: spec1 intensities are [0.6, 0.8], spec2 is [1.0].
        spec1 = _spec([100.0, 100.05], [3.0, 4.0])
        spec2 = _spec([100.02], [1.0])
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 1
        assert score == pytest.approx(0.8, rel=1e-5)

    def test_greedy_one_to_one_cross_competing_pairs(self):
        """Greedy picks highest product first without reusing peaks."""
        # spec1: m/z [100.0, 100.1], norm intensities [0.8, 0.6]
        # spec2: m/z [100.05, 100.15], norm intensities [0.8, 0.6]
        # With tol=0.06:
        #   pairs within tol:
        #     (0, 0): 0.8 * 0.8 = 0.64  <- picked first
        #     (1, 0): 0.6 * 0.8 = 0.48  <- blocked (spec2[0] used)
        #     (1, 1): 0.6 * 0.6 = 0.36  <- picked second
        # Total = 0.64 + 0.36 = 1.00, 2 matches.
        spec1 = _spec([100.0, 100.1], [4.0, 3.0])
        spec2 = _spec([100.05, 100.15], [4.0, 3.0])
        score, matched = similarity.cosine_fast(spec1, spec2, 0.06)
        assert matched == 2
        assert score == pytest.approx(1.0, rel=1e-5)

        # Now make the middle cross-pair (1, 0) steal the only match of
        # spec1[0] and spec2[1] when spec2[0] has high intensity:
        # spec1: [100.0, 100.06], int [0.6, 0.8]
        # spec2: [100.03, 100.09], int [0.9, sqrt(0.19)]
        # With tol=0.04:
        #   (0, 0): 0.6 * 0.9 = 0.54
        #   (1, 0): 0.8 * 0.9 = 0.72  <- picked first
        #   (1, 1): 0.8 * sqrt(0.19)  <- blocked (spec1[1] used)
        #   (0, 1): |100.0 - 100.09| = 0.09 > 0.04 (outside tol)
        # Leaves spec1[0] and spec2[1] unmatched -> score 0.72, 1.
        spec_a = _spec([100.0, 100.06], [0.6, 0.8], normalize=False)
        spec_b = _spec(
            [100.03, 100.09], [0.9, math.sqrt(0.19)], normalize=False
        )
        score, matched = similarity.cosine_fast(spec_a, spec_b, 0.04)
        assert matched == 1
        assert score == pytest.approx(0.72, rel=1e-5)

    def test_symmetry_on_random_spectra(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            mz1 = np.sort(rng.uniform(100.0, 110.0, 12)).astype(np.float32)
            mz2 = np.sort(rng.uniform(100.0, 110.0, 15)).astype(np.float32)
            i1 = rng.uniform(0.1, 1.0, 12).astype(np.float32)
            i2 = rng.uniform(0.1, 1.0, 15).astype(np.float32)
            s1 = _spec(mz1, i1)
            s2 = _spec(mz2, i2)
            score_ab, match_ab = similarity.cosine_fast(s1, s2, 0.2)
            score_ba, match_ba = similarity.cosine_fast(s2, s1, 0.2)
            assert match_ab == match_ba
            assert score_ab == pytest.approx(score_ba, abs=1e-6)

    def test_empty_and_disjoint_spectra(self):
        non_empty = _spec([100.0, 200.0], [0.6, 0.8], normalize=False)
        empty = _spec([], [], normalize=False)
        higher = _spec([500.0, 600.0], [0.6, 0.8], normalize=False)

        assert similarity.cosine_fast(non_empty, empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(empty, non_empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(empty, empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(non_empty, higher, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(higher, non_empty, 0.1) == (0.0, 0)

    def test_score_clamped_to_unit_interval(self):
        s1 = _spec([100.0, 200.0], [2.0, 2.0], normalize=False)
        score, matched = similarity.cosine_fast(s1, s1, 0.1)
        assert matched == 2
        assert score == 1.0


class TestDfRowToSpectrumTuple:
    def test_converts_rows_including_nan_charge(self):
        i1 = np.array([0.6, 0.8], dtype=np.float32)
        i2 = np.array([0.8, 0.6], dtype=np.float32)
        df = pd.DataFrame(
            {
                "precursor_mz": [100.0, 200.0],
                "precursor_charge": [2, np.nan],
                "mz": [
                    np.array([100.0, 150.0], dtype=np.float32),
                    np.array([200.0, 250.0], dtype=np.float32),
                ],
                "intensity": [i1, i2],
            }
        )
        tuples = df.apply(similarity.df_row_to_spectrum_tuple, axis=1)

        assert tuples[0].precursor_mz == 100.0
        assert tuples[0].precursor_charge == 2
        np.testing.assert_array_equal(tuples[0].mz, [100.0, 150.0])
        np.testing.assert_array_equal(tuples[0].intensity, i1)

        assert tuples[1].precursor_mz == 200.0
        assert math.isnan(tuples[1].precursor_charge)
        np.testing.assert_array_equal(tuples[1].mz, [200.0, 250.0])
        np.testing.assert_array_equal(tuples[1].intensity, i2)
