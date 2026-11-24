"""Unit tests for falcon.types records, charge buckets, and m/z bins."""

import math

import numpy as np
import pytest

from falcon.types import (
    AUTO_SENTINEL,
    NONE_SENTINEL,
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


class TestRecordsAndOffsetting:
    @pytest.mark.parametrize(
        "charge", [2, 2.0, np.int8(3), None, float("nan"), float("inf")]
    )
    def test_make_consensus_tuple_normalizes_dtypes(self, charge):
        rep = make_consensus_tuple(
            precursor_mz=500.25,
            precursor_charge=charge,
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensity=np.array([0.6, 0.8], dtype=np.float64),
            retention_time=12.5,
            cluster_id=7,
        )
        assert rep.precursor_mz.dtype == np.float32
        assert rep.mz.dtype == np.float32
        assert rep.intensity.dtype == np.float32
        assert rep.retention_time.dtype == np.float32
        assert rep.cluster_id == np.int32(7)
        if charge is None or not math.isfinite(float(charge)):
            assert math.isnan(rep.precursor_charge)
        else:
            assert rep.precursor_charge == np.int32(int(charge))

    def test_consensus_tuple_from_spectrum(self):
        spec = SpectrumTuple(
            412.5,
            None,
            np.array([150.0, 250.0], dtype=np.float64),
            np.array([0.5, 0.9], dtype=np.float64),
        )
        rep = ConsensusTuple.from_spectrum(
            spec, retention_time=45.0, cluster_id=3
        )
        assert rep.precursor_mz == np.float32(412.5)
        assert math.isnan(rep.precursor_charge)
        assert rep.cluster_id == np.int32(3)

    def test_offset_cluster_result_non_mutating_and_optional_reps(
        self, consensus_factory
    ):
        labels = np.array([0, 2, 1, 2], dtype=np.int32)
        reps = [consensus_factory(cluster_id=i) for i in range(3)]
        original = ClusterResult(labels=labels, representatives=reps)

        shifted, next_offset = offset_cluster_result(original, offset=10)
        assert shifted.labels.tolist() == [10, 12, 11, 12]
        assert [int(r.cluster_id) for r in shifted.representatives] == [
            10,
            11,
            12,
        ]
        assert next_offset == 13
        assert labels.tolist() == [0, 2, 1, 2]
        assert [int(r.cluster_id) for r in reps] == [0, 1, 2]

        no_reps, next_no_reps = offset_cluster_result(
            original, offset=5, include_representatives=False
        )
        assert no_reps.labels.tolist() == [5, 7, 6, 7]
        assert no_reps.representatives == []
        assert next_no_reps == 8

        empty = ClusterResult(np.empty(0, dtype=np.int32), [])
        empty_shifted, preserved = offset_cluster_result(empty, offset=42)
        assert empty_shifted.labels.size == 0
        assert preserved == 42


class TestChargeBucketsAndMzBins:
    def test_charge_bucket_canonicalization_and_optional(self):
        assert sorted(["unknown", 3, -1, 1], key=charge_sort_key) == [
            -1,
            1,
            3,
            "unknown",
        ]
        assert parse_charge_bucket_list(
            [["unknown", 2, 1], {-128, 127}, "other"]
        ) == ((1, 2, "unknown"), (-128, 127), "other")
        assert parse_optional_charge_buckets(None) is None
        assert parse_optional_charge_buckets(AUTO_SENTINEL) is None

    @pytest.mark.parametrize(
        "raw_value",
        [
            None,
            NONE_SENTINEL,
            [],
            [[]],
            [[True]],
            [[-129]],
            [[128]],
            [[1.5]],
            [["abc"]],
            [[2, 2]],
            [[1, 2], [2, 3]],
            [[1], "other", "other"],
        ],
    )
    def test_charge_buckets_reject_invalid(self, raw_value):
        with pytest.raises(ValueError, match=r".+"):
            parse_charge_bucket_list(raw_value)

    def test_mz_bin_bounds_outward_float64_rounding(self):
        assert mz_bin_bounds(100.0, 200.0, 1.0) == (101, 100.0, 201.0)
        assert mz_bin_bounds(250.0, 250.0, 0.5) == (1, 250.0, 250.5)

        # Float rounding cases: 1.7/0.1 -> 17*0.1 > 1.7; 4.3/0.1 -> 42.9999...
        for min_mz, max_mz, step in (
            (1.7, 2.0, 0.1),
            (0.0, 4.3, 0.1),
            (101.03, 1500.03, 0.05),
        ):
            n_bins, start, end = mz_bin_bounds(min_mz, max_mz, step)
            assert start <= min_mz
            assert end > max_mz
            assert n_bins >= 1
            assert end - start == pytest.approx(n_bins * step)

        with pytest.raises(ValueError, match="Invalid m/z bin bounds"):
            mz_bin_bounds(200.0, 100.0, 0.05)
        with pytest.raises(ValueError, match="Invalid m/z bin bounds"):
            mz_bin_bounds(0.0, 100.0, 0.0)
