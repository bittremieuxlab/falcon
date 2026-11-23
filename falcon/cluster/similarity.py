"""Greedy fragment-peak matching and cosine similarity calculation."""

import collections

import numba as nb
import numpy as np
import pandas as pd

SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)


@nb.njit(fastmath=True, boundscheck=False)
def cosine_fast(
    spec: SpectrumTuple,
    spec_other: SpectrumTuple,
    fragment_mz_tolerance: float,
) -> tuple[float, int]:
    """
    Compute greedy one-to-one cosine similarity between two spectra.

    Assumes both spectra have fragment ``mz`` arrays sorted ascending
    and L2-normalized ``intensity`` arrays of matching length. All peak
    pairs within ``fragment_mz_tolerance`` (inclusive) are ranked by
    intensity product and greedily matched without reusing either peak.

    Parameters
    ----------
    spec : SpectrumTuple
        First spectrum tuple.
    spec_other : SpectrumTuple
        Second spectrum tuple.
    fragment_mz_tolerance : float
        Maximum absolute m/z difference (in Da, inclusive) allowed for
        two fragment peaks to match.

    Returns
    -------
    tuple[float, int]
        Cosine similarity clamped to ``[0.0, 1.0]`` and the number of
        matched peak pairs.
    """
    # Enumerate candidate peak pairs within the fragment m/z window
    # using a sliding pointer over sorted m/z arrays.
    other_peak_index = 0
    peak_matches = []
    for peak_index, (peak_mz, peak_intensity) in enumerate(
        zip(spec.mz, spec.intensity)  # noqa: B905 (Numba lacks strict=)
    ):
        # Advance past peaks below the lower tolerance boundary.
        while other_peak_index < len(spec_other.mz) - 1 and (
            peak_mz - fragment_mz_tolerance > spec_other.mz[other_peak_index]
        ):
            other_peak_index += 1
        # Collect all peaks within +/- fragment_mz_tolerance.
        other_peak_i = other_peak_index
        while (
            other_peak_i < len(spec_other.mz)
            and abs(peak_mz - (spec_other.mz[other_peak_i]))
            <= fragment_mz_tolerance
        ):
            peak_matches.append(
                (
                    peak_index,
                    other_peak_i,
                    peak_intensity * spec_other.intensity[other_peak_i],
                )
            )
            other_peak_i += 1

    # Greedily accumulate highest-scoring one-to-one peak matches.
    score = 0.0
    peak_matches.sort(key=lambda x: x[2], reverse=True)
    spec_peaks_used = np.zeros(len(spec.mz), np.bool_)
    other_spec_peaks_used = np.zeros(len(spec_other.mz), np.bool_)
    for peak_index, other_peak_index, pair_score in peak_matches:
        if (
            not spec_peaks_used[peak_index]
            and not other_spec_peaks_used[other_peak_index]
        ):
            score += pair_score
            spec_peaks_used[peak_index] = True
            other_spec_peaks_used[other_peak_index] = True

    score = max(0.0, min(score, 1.0))

    return score, int(spec_peaks_used.sum())


def df_row_to_spectrum_tuple(row: pd.Series) -> SpectrumTuple:
    """
    Convert a DataFrame row into a ``SpectrumTuple``.

    Parameters
    ----------
    row : pd.Series
        Row containing ``precursor_mz``, ``precursor_charge``, ``mz``,
        and ``intensity`` fields.

    Returns
    -------
    SpectrumTuple
        Numba-compatible spectrum tuple.
    """
    return SpectrumTuple(
        row["precursor_mz"],
        row["precursor_charge"],
        row["mz"],
        row["intensity"],
    )
