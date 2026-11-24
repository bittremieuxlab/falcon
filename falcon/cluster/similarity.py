"""Greedy fragment-peak matching and cosine similarity calculation."""

from __future__ import annotations

import numba as nb
import numpy as np

__all__ = ["cosine_fast_arrays"]


@nb.njit(cache=True)
def cosine_fast_arrays(
    mz1: np.ndarray,
    intensity1: np.ndarray,
    mz2: np.ndarray,
    intensity2: np.ndarray,
    fragment_tol: float,
) -> tuple[float, int]:
    """
    Compute greedy one-to-one cosine similarity over peak arrays.

    Assumes both ``mz`` arrays are sorted ascending and both
    ``intensity`` arrays are L2-normalized with matching length.

    Parameters
    ----------
    mz1, intensity1 : np.ndarray
        First spectrum's sorted fragment m/z and unit-norm intensities.
    mz2, intensity2 : np.ndarray
        Second spectrum's sorted fragment m/z and unit-norm intensities.
    fragment_tol : float
        Maximum absolute m/z difference (in Da, inclusive) allowed for
        two fragment peaks to match.

    Returns
    -------
    tuple[float, int]
        Cosine similarity clamped to ``[0.0, 1.0]`` and the number of
        matched peak pairs.
    """
    n1 = len(mz1)
    n2 = len(mz2)
    if n1 == 0 or n2 == 0:
        return 0.0, 0

    other_idx = 0
    peak_matches = []
    for i in range(n1):
        peak_mz = mz1[i]
        peak_intensity = intensity1[i]
        while other_idx < n2 and peak_mz - mz2[other_idx] > fragment_tol:
            other_idx += 1
        if other_idx == n2:
            break
        j = other_idx
        while j < n2 and mz2[j] - peak_mz <= fragment_tol:
            # Negate score so plain tuple sort orders by descending
            # score with ascending (i, j) tie-breaking and stays
            # Numba-cacheable (no key-lambda closure).
            peak_matches.append((-(peak_intensity * intensity2[j]), i, j))
            j += 1

    if len(peak_matches) == 0:
        return 0.0, 0

    # Greedily accumulate highest-scoring one-to-one peak matches.
    score = 0.0
    n_matches = 0
    peak_matches.sort()
    used1 = np.zeros(n1, np.bool_)
    used2 = np.zeros(n2, np.bool_)
    for neg_score, i, j in peak_matches:
        if not used1[i] and not used2[j]:
            score -= neg_score
            used1[i] = True
            used2[j] = True
            n_matches += 1

    score = max(0.0, min(score, 1.0))
    return score, n_matches
