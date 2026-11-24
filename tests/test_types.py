"""Unit tests for falcon.types records, charge buckets, and m/z bins."""

import math

import numpy as np
import pytest

from falcon.types import (
    AUTO_SENTINEL,
    NONE_SENTINEL,
    ChargeValue,
    ClusterResult,
    ConsensusTuple,
    SpectrumTuple,
    charge_sort_key,
    make_consensus_tuple,
    mz_bin_bounds,
    offset_cluster_result,
    parse_charge_bucket_list,
    parse_optional_charge_buckets,
)


class TestSpectrumAndConsensusRecords:
    @pytest.mark.parametrize(
        "charge", [2, np.int8(3), None, float("nan"), float("inf")]
    )
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
        if charge is None or not math.isfinite(float(charge)):
            assert math.isnan(rep.precursor_charge)
        else:
            assert rep.precursor_charge == np.int32(charge)

    @pytest.mark.parametrize("charge", [2, None])
    def test_consensus_tuple_from_spectrum(self, charge):
        spec = SpectrumTuple(
            precursor_mz=412.5,
            precursor_charge=charge,
            mz=np.array([150.0, 250.0], dtype=np.float64),
            intensity=np.array([0.5, 0.9], dtype=np.float64),
        )
        rep = ConsensusTuple.from_spectrum(
            spec, retention_time=45.0, cluster_id=3
        )
        assert isinstance(rep, ConsensusTuple)
        assert rep.precursor_mz == np.float32(412.5)
        assert rep.mz.dtype == np.float32
        assert rep.intensity.dtype == np.float32
        assert rep.retention_time == np.float32(45.0)
        assert rep.cluster_id == np.int32(3)
        if charge is None:
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


class TestClusterResultOffsetting:
    def test_offset_cluster_result_shifts_without_mutating_inputs(
        self, consensus_factory
    ):
        labels = np.array([0, 2, 1, 2], dtype=np.int32)
        reps = [
            consensus_factory(cluster_id=0),
            consensus_factory(cluster_id=1),
            consensus_factory(cluster_id=2),
        ]
        original = ClusterResult(labels=labels, representatives=reps)

        shifted, next_offset = offset_cluster_result(original, offset=10)

        assert isinstance(shifted, ClusterResult)
        assert shifted.labels.dtype == np.int32
        assert shifted.labels.tolist() == [10, 12, 11, 12]
        assert [int(r.cluster_id) for r in shifted.representatives] == [
            10,
            11,
            12,
        ]
        assert next_offset == 13
        # Input array and representative records must remain untouched.
        assert labels.tolist() == [0, 2, 1, 2]
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2]

    def test_offset_cluster_result_omits_representatives_when_disabled(
        self, consensus_factory
    ):
        result = ClusterResult(
            labels=np.array([0, 1], dtype=np.int32),
            representatives=[consensus_factory(cluster_id=0)],
        )

        shifted, next_offset = offset_cluster_result(
            result, offset=5, include_representatives=False
        )

        assert shifted.labels.tolist() == [5, 6]
        assert shifted.representatives == []
        assert next_offset == 7

    def test_offset_cluster_result_empty_input_preserves_offset(self):
        empty = ClusterResult(
            labels=np.empty(0, dtype=np.int32),
            representatives=[],
        )

        shifted, next_offset = offset_cluster_result(empty, offset=42)

        assert shifted.labels.dtype == np.int32
        assert shifted.labels.size == 0
        assert shifted.representatives == []
        assert next_offset == 42


class TestChargeBucketsAndSentinels:
    def test_sentinels_and_charge_sort_key(self):
        assert NONE_SENTINEL == "none"
        assert AUTO_SENTINEL == "auto"
        raw_charges: list[ChargeValue] = ["unknown", 3, -1, 1]
        ordered = sorted(raw_charges, key=charge_sort_key)
        assert ordered == [-1, 1, 3, "unknown"]

    def test_parse_charge_bucket_list_canonicalizes_valid_inputs(self):
        parsed = parse_charge_bucket_list(
            [["unknown", 2, 1], {-128, 127}, "other"]
        )
        assert parsed == ((1, 2, "unknown"), (-128, 127), "other")

    def test_parse_optional_charge_buckets_handles_none_and_auto(self):
        assert parse_optional_charge_buckets(None) is None
        assert parse_optional_charge_buckets(AUTO_SENTINEL) is None
        assert parse_optional_charge_buckets([["unknown", 2, 1], "other"]) == (
            (1, 2, "unknown"),
            "other",
        )

    @pytest.mark.parametrize(
        ("raw_value", "match"),
        [
            (None, "non-empty list"),
            (AUTO_SENTINEL, "non-empty list"),
            ("12", "non-empty list"),
            ([], "must not be empty"),
            ([1], "sequence of charges"),
            ([[]], "must not be empty"),
            ([[True]], "not booleans"),
            ([[False]], "not booleans"),
            ([[-129]], "outside int8"),
            ([[128]], "outside int8"),
            ([[1.5]], "Invalid charge bucket entry"),
            ([["abc"]], "Invalid charge bucket entry"),
            ([["other"]], "Invalid charge bucket entry"),
            ([[2, 2]], "Duplicate charge value"),
            ([["unknown", "unknown"]], "Duplicate charge value"),
            ([[1, 2], [2, 3]], "more than one bucket"),
            ([[1, "unknown"], ["unknown"]], "more than one bucket"),
            ([[1], "other", "other"], "appears more than once"),
        ],
    )
    def test_parse_charge_bucket_list_rejects_invalid_inputs(
        self, raw_value, match
    ):
        with pytest.raises(ValueError, match=match):
            parse_charge_bucket_list(raw_value)

    @pytest.mark.parametrize(
        ("raw_value", "match"),
        [
            (NONE_SENTINEL, "non-empty list"),
            ([], "must not be empty"),
            ([[128]], "outside int8"),
        ],
    )
    def test_parse_optional_charge_buckets_rejects_invalid_inputs(
        self, raw_value, match
    ):
        with pytest.raises(ValueError, match=match):
            parse_optional_charge_buckets(raw_value)


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

    def test_handles_float_rounding_adjustments(self):
        # 1.7 / 0.1 == 17.0, 17 * 0.1 > 1.7 -> start_idx decrements.
        n_bins, start, end = mz_bin_bounds(1.7, 2.0, 0.1)
        assert start <= 1.7
        assert end > 2.0
        assert n_bins == 5

        # 4.3 / 0.1 == 42.99999999999999 -> end_idx += 1.
        n_bins2, start2, end2 = mz_bin_bounds(0.0, 4.3, 0.1)
        assert start2 == 0.0
        assert end2 > 4.3
        assert n_bins2 == 44

        n_bins3, start3, end3 = mz_bin_bounds(0.0, 1500.0, 0.05)
        assert start3 == 0.0
        assert end3 > 1500.0
        assert n_bins3 == 30001

    @pytest.mark.parametrize(
        ("min_mz", "max_mz", "bin_size", "match"),
        [
            (0.0, float("inf"), 0.05, "finite"),
            (float("nan"), 100.0, 0.05, "finite"),
            (0.0, 100.0, float("inf"), "finite"),
            (0.0, 100.0, 0.0, "positive"),
            (0.0, 100.0, -0.05, "positive"),
            (200.0, 100.0, 0.05, "must not exceed"),
            (0.0, 1e308, 1e-308, "non-finite"),
            (0.0, 1e308, 1e308, "non-finite"),
            (0.0, 1e20, 1.0, "float64 precision"),
        ],
    )
    def test_rejects_invalid_inputs(self, min_mz, max_mz, bin_size, match):
        with pytest.raises(ValueError, match=match):
            mz_bin_bounds(min_mz, max_mz, bin_size)
