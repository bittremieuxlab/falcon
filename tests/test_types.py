"""Unit tests for falcon.types shared records, charge buckets, and bins."""

import math

import numpy as np
import pytest

from falcon.types import (
    ConsensusTuple,
    SpectrumTuple,
    charge_sort_key,
    make_consensus_tuple,
    mz_bin_bounds,
    parse_charge_buckets,
)


class TestConsensusTupleFactory:
    @pytest.mark.parametrize("charge", [2, np.int8(3), None, float("nan")])
    def test_make_consensus_tuple_normalizes_dtypes_and_missing_charge(
        self, charge
    ):
        rep = make_consensus_tuple(
            precursor_mz=500.25,
            precursor_charge=charge,
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensity=np.array([0.6, 0.8], dtype=np.float64),
            retention_time=12.5,
            cluster_id=7,
        )
        assert isinstance(rep, ConsensusTuple)
        assert rep.precursor_mz.dtype == np.float32
        assert rep.mz.dtype == np.float32
        assert rep.intensity.dtype == np.float32
        assert rep.retention_time.dtype == np.float32
        assert rep.cluster_id == np.int32(7)
        if charge is None or math.isnan(float(charge)):
            assert math.isnan(rep.precursor_charge)
        else:
            assert rep.precursor_charge == np.int32(charge)

    def test_spectrum_tuple_is_namedtuple(self):
        spec = SpectrumTuple(
            500.0,
            2,
            np.array([100.0], np.float32),
            np.array([1.0], np.float32),
        )
        assert spec.precursor_mz == 500.0
        assert spec.precursor_charge == 2


class TestChargeBucketsAndSortKey:
    def test_charge_sort_key_orders_integers_before_unknown(self):
        ordered = sorted(["unknown", 3, -1, 1], key=charge_sort_key)
        assert ordered == [-1, 1, 3, "unknown"]

    def test_parse_charge_buckets_canonicalizes_and_validates(self):
        assert parse_charge_buckets(None) is None
        assert parse_charge_buckets("auto") is None
        assert parse_charge_buckets([["unknown", 2, 1], "other"]) == (
            (1, 2, "unknown"),
            "other",
        )
        with pytest.raises(ValueError, match="not be empty"):
            parse_charge_buckets([])
        with pytest.raises(ValueError, match="outside int8"):
            parse_charge_buckets([[128]])
        with pytest.raises(ValueError, match="more than one bucket"):
            parse_charge_buckets([[1, 2], [2, 3]])


class TestMzBinBounds:
    def test_exact_aligned_and_unaligned_bounds(self):
        n_bins, start, end = mz_bin_bounds(100.0, 200.0, 1.0)
        assert (n_bins, start, end) == (101, 100.0, 201.0)

        n_bins2, start2, end2 = mz_bin_bounds(101.03, 1500.03, 0.05)
        assert start2 <= 101.03
        assert end2 > 1500.03
        assert end2 - start2 == pytest.approx(n_bins2 * 0.05)

        n_bins3, start3, end3 = mz_bin_bounds(250.0, 250.0, 0.5)
        assert (n_bins3, start3, end3) == (1, 250.0, 250.5)

    def test_handles_float_rounding_and_rejects_invalid_inputs(self):
        # 0.29999999999999993 / 0.1 == 3.0 -> exercises start_idx adjustment.
        n_bins, start, end = mz_bin_bounds(0.29999999999999993, 1.0, 0.1)
        assert start <= 0.29999999999999993
        assert end > 1.0
        assert n_bins >= 1

        # 0.30000000000000004 / 0.1 == 3.0 -> exercises end_idx check.
        n2, s2, e2 = mz_bin_bounds(0.0, 1500.0, 0.05)
        assert s2 == 0.0
        assert e2 > 1500.0
        assert n2 == 30001

        with pytest.raises(ValueError, match="finite"):
            mz_bin_bounds(0.0, float("inf"), 0.05)
        with pytest.raises(ValueError, match="positive"):
            mz_bin_bounds(0.0, 100.0, 0.0)
        with pytest.raises(ValueError, match="must not exceed"):
            mz_bin_bounds(200.0, 100.0, 0.05)
        with pytest.raises(ValueError, match="non-finite"):
            mz_bin_bounds(0.0, 1e308, 1e-308)
        with pytest.raises(ValueError, match="non-finite"):
            mz_bin_bounds(0.0, 1e308, 1e308)
        with pytest.raises(ValueError, match="float64 precision"):
            mz_bin_bounds(0.0, 1e20, 1.0)
