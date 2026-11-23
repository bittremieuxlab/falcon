"""Unit tests for falcon.cluster.similarity."""

import math

import numpy as np
import pytest

from falcon.cluster import similarity


class TestCosineFast:
    def test_identical_spectra(self, mock_spectra):
        spec1, spec2 = mock_spectra(2, n_peaks=3)
        score, matched = similarity.cosine_fast(spec1, spec2, 0.5)
        assert score == pytest.approx(1.0)
        assert matched == 3

    def test_hand_computed_different_intensities(self, spec_factory):
        a = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        b = np.array([0.7, 0.6, 0.5], dtype=np.float32)
        expected = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
        spec1 = spec_factory([100.0, 101.0, 102.0], a)
        spec2 = spec_factory([100.0, 101.0, 102.0], b)
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert score == pytest.approx(expected, rel=1e-5)
        assert matched == 3

    def test_tolerance_boundary_inclusive(self, spec_factory):
        spec1 = spec_factory([100.0], [1.0], normalize=False)
        spec2 = spec_factory([100.5], [1.0], normalize=False)
        assert similarity.cosine_fast(spec1, spec2, 0.5) == (1.0, 1)
        assert similarity.cosine_fast(spec1, spec2, 0.499) == (0.0, 0)

    def test_one_to_one_greedy_highest_product_wins(self, spec_factory):
        spec1 = spec_factory([100.0], [1.0], normalize=False)
        spec2 = spec_factory([100.01, 100.02], [0.6, 0.8], normalize=False)
        score, matched = similarity.cosine_fast(spec1, spec2, 0.05)
        assert matched == 1
        assert score == pytest.approx(0.8, rel=1e-5)

        spec_a = spec_factory([100.0, 100.06], [0.6, 0.8], normalize=False)
        spec_b = spec_factory(
            [100.03, 100.09], [0.9, math.sqrt(0.19)], normalize=False
        )
        score2, matched2 = similarity.cosine_fast(spec_a, spec_b, 0.04)
        assert matched2 == 1
        assert score2 == pytest.approx(0.72, rel=1e-5)

    def test_symmetry_on_random_spectra(self, spec_factory):
        rng = np.random.default_rng(7)
        for _ in range(20):
            mz1 = np.sort(rng.uniform(100.0, 110.0, 12)).astype(np.float32)
            mz2 = np.sort(rng.uniform(100.0, 110.0, 15)).astype(np.float32)
            i1 = rng.uniform(0.1, 1.0, 12).astype(np.float32)
            i2 = rng.uniform(0.1, 1.0, 15).astype(np.float32)
            s1 = spec_factory(mz1, i1)
            s2 = spec_factory(mz2, i2)
            score_ab, match_ab = similarity.cosine_fast(s1, s2, 0.2)
            score_ba, match_ba = similarity.cosine_fast(s2, s1, 0.2)
            assert match_ab == match_ba
            assert score_ab == pytest.approx(score_ba, abs=1e-6)

    def test_empty_and_disjoint_spectra(self, spec_factory):
        non_empty = spec_factory([100.0, 200.0], [0.6, 0.8], normalize=False)
        empty = spec_factory([], [], normalize=False)
        higher = spec_factory([500.0, 600.0], [0.6, 0.8], normalize=False)

        assert similarity.cosine_fast(non_empty, empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(empty, non_empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(empty, empty, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(non_empty, higher, 0.1) == (0.0, 0)
        assert similarity.cosine_fast(higher, non_empty, 0.1) == (0.0, 0)

    def test_score_clamped_to_unit_interval(self, spec_factory):
        s1 = spec_factory([100.0, 200.0], [2.0, 2.0], normalize=False)
        score, matched = similarity.cosine_fast(s1, s1, 0.1)
        assert matched == 2
        assert score == 1.0
