"""Unit tests for falcon.cluster.distance (similarity & CSR matrices)."""

import math
import tempfile

import numpy as np
import pyarrow as pa
import pytest
from scipy.spatial.distance import squareform

from falcon.cluster import distance, engine


def _random_spectra(n, rng, spec_factory, n_peaks=25, mz_max=200.0):
    return [
        spec_factory(
            np.sort(rng.uniform(0.0, mz_max, n_peaks)),
            rng.random(n_peaks),
            precursor_mz=float(rng.uniform(100.0, 1000.0)),
            charge=1,
        )
        for _ in range(n)
    ]


class TestCosineFastArrays:
    def test_identical_boundary_greedy_and_clamping(self, spec_factory):
        s1 = spec_factory([100.0], [1.0], normalize=False)
        s2 = spec_factory([100.5], [1.0], normalize=False)
        assert distance.cosine_fast_arrays(
            s1.mz, s1.intensity, s2.mz, s2.intensity, 0.5
        ) == (1.0, 1)
        assert distance.cosine_fast_arrays(
            s1.mz, s1.intensity, s2.mz, s2.intensity, 0.499
        ) == (0.0, 0)

        # Greedy one-to-one picks highest product pair first.
        sa = spec_factory([100.0, 100.06], [0.6, 0.8], normalize=False)
        sb = spec_factory(
            [100.03, 100.09], [0.9, math.sqrt(0.19)], normalize=False
        )
        score, matched = distance.cosine_fast_arrays(
            sa.mz, sa.intensity, sb.mz, sb.intensity, 0.04
        )
        assert matched == 1
        assert score == pytest.approx(0.72, rel=1e-5)

        # Empty and disjoint inputs return (0.0, 0).
        empty = spec_factory([], [], normalize=False)
        assert distance.cosine_fast_arrays(
            sa.mz, sa.intensity, empty.mz, empty.intensity, 0.1
        ) == (0.0, 0)

        # Unnormalized inputs clamp to [0, 1].
        hi = spec_factory([100.0, 200.0], [2.0, 2.0], normalize=False)
        assert distance.cosine_fast_arrays(
            hi.mz, hi.intensity, hi.mz, hi.intensity, 0.1
        ) == (1.0, 2)

    def test_matches_independent_brute_force_reference_and_symmetry(
        self, spec_factory
    ):
        rng = np.random.default_rng(99)
        tol = 0.15
        for _ in range(25):
            mz1 = np.sort(rng.uniform(100.0, 105.0, 10)).astype(np.float32)
            mz2 = np.sort(rng.uniform(100.0, 105.0, 12)).astype(np.float32)
            s1 = spec_factory(
                mz1, rng.uniform(0.1, 1.0, 10).astype(np.float32)
            )
            s2 = spec_factory(
                mz2, rng.uniform(0.1, 1.0, 12).astype(np.float32)
            )

            pairs = sorted(
                (-(float(s1.intensity[i]) * float(s2.intensity[j])), i, j)
                for i in range(len(s1.mz))
                for j in range(len(s2.mz))
                if abs(float(s1.mz[i]) - float(s2.mz[j])) <= tol
            )
            used1: set[int] = set()
            used2: set[int] = set()
            ref_score, ref_matches = 0.0, 0
            for neg_prod, i, j in pairs:
                if i not in used1 and j not in used2:
                    used1.add(i)
                    used2.add(j)
                    ref_score -= neg_prod
                    ref_matches += 1
            ref_score = max(0.0, min(ref_score, 1.0))

            score_ab, m_ab = distance.cosine_fast_arrays(
                s1.mz, s1.intensity, s2.mz, s2.intensity, tol
            )
            score_ba, m_ba = distance.cosine_fast_arrays(
                s2.mz, s2.intensity, s1.mz, s1.intensity, tol
            )
            assert m_ab == m_ba == ref_matches
            assert score_ab == pytest.approx(ref_score, abs=1e-6)
            assert score_ba == pytest.approx(ref_score, abs=1e-6)

    def test_lower_boundary_rounding_never_drops_exact_match(self):
        rng = np.random.default_rng(123)
        tol = 0.05
        ones = np.array([1.0], dtype=np.float64)
        two_ones = np.array([0.0, 1.0], dtype=np.float64)
        for peak in rng.uniform(100.0, 1500.0, size=100):
            score, matched = distance.cosine_fast_arrays(
                np.array([peak]),
                ones,
                np.array([peak - tol, peak]),
                two_ones,
                tol,
            )
            assert matched == 1
            assert score == pytest.approx(1.0)


class TestCondensedDistanceAndCsr:
    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_condensed_index_matches_scipy_squareform(self, n):
        square = squareform(np.arange(n * (n - 1) // 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    idx = distance.condensed_index(i, j, n)
                    assert idx == square[i, j]
                    assert idx == distance.condensed_index(j, i, n)

    def test_extract_arrow_csr_sliced_chunked_and_mismatch(self):
        mz_parent = pa.array(
            [[50.0], [100.0, 200.0], [300.0]], pa.list_(pa.float32())
        )
        int_parent = pa.array(
            [[0.1], [0.2], [0.6, 0.8], [1.0]], pa.list_(pa.float32())
        )
        sliced = pa.table(
            {"mz": mz_parent.slice(1, 2), "intensity": int_parent.slice(2, 2)}
        )
        mz_flat, int_flat, offsets = distance.extract_arrow_csr(sliced)
        np.testing.assert_allclose(mz_flat, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(int_flat, [0.6, 0.8, 1.0])
        np.testing.assert_array_equal(offsets, [0, 2, 3])

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

    def test_condensed_matrix_distances_min_peaks_and_memmap_cleanup(
        self, spec_factory, tmp_path, monkeypatch
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
        csr = distance.pack_spectra_csr([s0, s1, s2])

        pdist = distance.compute_condensed_distance_matrix_csr(*csr, 0.05, 0)
        assert pdist.dtype == np.float32
        np.testing.assert_allclose(pdist, [0.4, 1.0, 1.0], atol=1e-6)

        # min_matched_peaks gate forces distance to 1.0 when unmet.
        gated = distance.compute_condensed_distance_matrix_csr(*csr, 0.05, 4)
        np.testing.assert_allclose(gated, [1.0, 1.0, 1.0], atol=1e-6)

        # Empty/single spectra return empty float32 vector.
        for n in (0, 1):
            empty_pdist = distance.compute_condensed_distance_matrix_csr(
                *distance.pack_spectra_csr([s0][:n]), 0.05, 0
            )
            assert empty_pdist.shape == (0,)

        # Memmap path unlinks backing file immediately on POSIX.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(distance, "MEMMAP_MIN_SPECTRA", 2)
        mm = distance.compute_condensed_distance_matrix_csr(*csr, 0.05, 0)
        assert isinstance(mm, np.memmap)
        assert list(tmp_path.iterdir()) == []
        np.testing.assert_allclose(mm, pdist)
        distance.close_memmap(mm)

    def test_tiled_rows_bit_identical_to_full_matrix(self, spec_factory):
        rng = np.random.default_rng(7)
        specs = _random_spectra(24, rng, spec_factory, n_peaks=20)
        mz_flat, int_flat, offsets = distance.pack_spectra_csr(specs)
        full = distance.compute_condensed_distance_matrix_csr(
            mz_flat, int_flat, offsets, 0.05, 2
        )

        tiled = np.empty_like(full)
        for r0, r1 in engine.interval_row_bounds(len(specs), 4):
            distance.fill_condensed_rows_csr(
                tiled, mz_flat, int_flat, offsets, r0, r1, 0.05, 2
            )
        assert np.array_equal(tiled, full)
