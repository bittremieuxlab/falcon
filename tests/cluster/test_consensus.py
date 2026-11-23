"""Unit tests for falcon.cluster.consensus."""

import math

import numba as nb
import numpy as np
import pytest

from falcon.cluster import consensus, similarity


def _spec(mz, intensity, precursor_mz=500.0, charge=2, normalize=True):
    mz_arr = np.asarray(mz, dtype=np.float32)
    int_arr = np.asarray(intensity, dtype=np.float32)
    if normalize and int_arr.size > 0:
        int_arr = int_arr / np.linalg.norm(int_arr)
    return similarity.SpectrumTuple(
        precursor_mz=np.float32(precursor_mz),
        precursor_charge=charge,
        mz=mz_arr,
        intensity=int_arr,
    )


# ----------------------------------------------------------------------
# _get_cluster_group_idx
# ----------------------------------------------------------------------


class TestGetClusterGroupIdx:
    def test_noise_singletons_and_contiguous_clusters(self):
        clusters = np.array([-1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
        assert list(consensus._get_cluster_group_idx(clusters)) == [
            (0, 1),
            (1, 2),
            (2, 4),
            (4, 7),
        ]

    def test_non_contiguous_labels_and_all_noise(self):
        non_contig = np.array([0, 0, 5, 5, 10], dtype=np.int32)
        assert list(consensus._get_cluster_group_idx(non_contig)) == [
            (0, 2),
            (2, 4),
            (4, 5),
        ]
        all_noise = np.array([-1, -1, -1], dtype=np.int32)
        assert list(consensus._get_cluster_group_idx(all_noise)) == [
            (0, 1),
            (1, 2),
            (2, 3),
        ]

    def test_empty_and_single_element(self):
        assert (
            list(consensus._get_cluster_group_idx(np.array([], np.int32)))
            == []
        )
        assert list(
            consensus._get_cluster_group_idx(np.array([0], np.int32))
        ) == [(0, 1)]


# ----------------------------------------------------------------------
# _get_cluster_medoids
# ----------------------------------------------------------------------


class TestGetClusterMedoids:
    def test_small_clusters_select_first_member(self, mock_spectra):
        spectra = mock_spectra(2)
        labels = np.array([0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0], dtype=np.float32)
        order_map = np.array([1, 0], dtype=np.int64)
        pdist = np.array([0.5], dtype=np.float32)
        res = consensus._get_cluster_medoids(
            spectra, labels, rts, order_map, pdist
        )
        assert res[4] == [10.0]
        assert res[5] == [0]

    def test_selects_min_row_sum_with_non_identity_order_map_and_noise(self):
        # 5 spectra in canonical order:
        #   noise: spec 4 (label -1)
        #   cluster 0 (size 3): specs [3, 0, 2] (triggers i > j swap)
        #   cluster 1 (size 1): spec 1
        spectra = [
            _spec([100.0 + i], [1.0], precursor_mz=500.0 + i) for i in range(5)
        ]
        # Condensed pdist for m=5 (10 entries):
        # (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3,
        # (1,2)=4, (1,3)=5, (1,4)=6,
        # (2,3)=7, (2,4)=8, (3,4)=9
        # For cluster {3, 0, 2}:
        #   d(0, 2)=0.6, d(0, 3)=0.1, d(2, 3)=0.2
        #   row sums: spec 3 -> 0.1+0.2=0.3 (medoid!),
        #             spec 0 -> 0.1+0.6=0.7, spec 2 -> 0.6+0.2=0.8
        pdist = np.full(10, 0.9, dtype=np.float32)
        pdist[1] = 0.6  # (0, 2)
        pdist[2] = 0.1  # (0, 3)
        pdist[7] = 0.2  # (2, 3)

        labels = np.array([-1, 0, 0, 0, 1], dtype=np.int32)
        order_map = np.array([4, 2, 0, 3, 1], dtype=np.int64)
        rts = np.array([50.0, 20.0, 10.0, 30.0, 40.0], dtype=np.float32)

        precursor_mzs, _ch, _mzs, _ints, out_rts, cluster_ids = (
            consensus._get_cluster_medoids(
                spectra, labels, rts, order_map, pdist
            )
        )
        assert cluster_ids == [-1, 0, 1]
        # Noise -> spec 4 (504, rt 50); cluster 0 -> spec 3 (503, 30);
        # cluster 1 -> spec 1 (mz 501, rt 40).
        np.testing.assert_allclose(precursor_mzs, [504.0, 503.0, 501.0])
        np.testing.assert_allclose(out_rts, [50.0, 30.0, 40.0])


# ----------------------------------------------------------------------
# _spectrum_binning
# ----------------------------------------------------------------------


class TestSpectrumBinning:
    def test_70_percent_occupancy_boundary(self):
        """Bin with 7/10 spectra survives; bin with 6/10 is dropped."""
        spectra = []
        for i in range(10):
            mzs = [100.2]
            ints = [1.0]
            if i < 7:
                mzs.append(120.2)
                ints.append(0.5)
            if i < 6:
                mzs.append(150.2)
                ints.append(0.3)
            spectra.append(_spec(mzs, ints))

        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        assert len(bins_idx) == len(bins_peaks) == len(bins_mz) == 2
        kept_mzs = sorted(float(np.mean(m)) for m in bins_mz)
        np.testing.assert_allclose(kept_mzs, [100.2, 120.2], atol=1e-4)

    def test_multiple_peaks_in_same_bin_count_as_one_spectrum(self):
        """Multiple peaks from 1 of 3 spectra cannot satisfy 70%."""
        spectra = [
            _spec([100.1, 100.2, 100.3, 150.1], [0.5, 0.5, 0.5, 1.0]),
            _spec([150.2], [1.0]),
            _spec([150.3], [1.0]),
        ]
        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        # Bin 100 has 3 peaks from 1/3 spectra (33% < 70%) -> dropped.
        # Bin 150 has 3/3 spectra (100%) -> retained.
        assert len(bins_idx) == 1
        assert len(bins_peaks[0]) == 3
        np.testing.assert_allclose(bins_mz[0], [150.1, 150.2, 150.3])

    def test_out_of_range_peaks_ignored_and_all_filtered_returns_empty(self):
        spectra = [
            _spec([10.0, 100.0, 999.0], [1.0, 1.0, 1.0]),
            _spec([10.0, 120.0, 999.0], [1.0, 1.0, 1.0]),
        ]
        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        # 10/999 are out of range; 100/120 have 50% < 70% occupancy.
        assert len(bins_idx) == 0
        assert len(bins_peaks) == 0
        assert len(bins_mz) == 0


# ----------------------------------------------------------------------
# _sigma_clip, _sigma_clipping, and _outlier_rejection
# ----------------------------------------------------------------------


class TestSigmaClippingAndOutlierRejection:
    def test_sigma_clip_exact_inclusive_and_asymmetric_bounds(self):
        values = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0], dtype=np.float32)
        # median=10, std=5, bounds [5.0, 20.0] (inclusive).
        mask = consensus._sigma_clip(values, 10.0, 5.0, 1.0, 2.0)
        np.testing.assert_array_equal(
            mask, [False, True, True, True, True, False]
        )

    def test_sigma_clipping_iterates_to_remove_masked_secondary_outlier(self):
        # Outlier 1000 inflates std so 20 survives iter 1; after 1000
        # is removed, iter 2 rejects 20 too.
        intensities = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 1000.0], dtype=np.float32
        )
        mzs = np.full_like(intensities, 100.0)
        kept_i, kept_m = consensus._sigma_clipping(intensities, mzs, 1.5, 1.5)
        np.testing.assert_allclose(kept_i, [1.0, 1.0, 1.0, 1.0, 1.0])
        assert len(kept_m) == 5

    def test_sigma_clipping_stops_on_zero_std_and_small_arrays(self):
        identical = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mzs = np.array([100.0, 100.1, 100.2], dtype=np.float32)
        kept_i, kept_m = consensus._sigma_clipping(identical, mzs, 1.5, 1.5)
        np.testing.assert_array_equal(kept_i, identical)
        np.testing.assert_array_equal(kept_m, mzs)

        pair = np.array([0.1, 100.0], dtype=np.float32)
        kept_p, _ = consensus._sigma_clipping(pair, mzs[:2], 0.1, 0.1)
        np.testing.assert_array_equal(kept_p, pair)

    def test_outlier_rejection_handles_small_bins_and_dropped_bins(self):
        bins_indices = np.array([0, 1, 2], dtype=np.int32)
        # Bin 0 (len 2): skips clipping -> exact mean (0.2+0.8)/2 = 0.5.
        # Bin 1 (len 4): clips 10.0 -> mean of [0.4, 0.5, 0.6] = 0.5.
        # Bin 2 (len 4): cutoff 0.05 clips every point -> dropped.
        bins_peaks = nb.typed.List(
            [
                np.array([0.2, 0.8], dtype=np.float32),
                np.array([0.4, 0.5, 0.6, 10.0], dtype=np.float32),
                np.array([1.0, 2.0, 4.0, 5.0], dtype=np.float32),
            ]
        )
        bins_mz = nb.typed.List(
            [
                np.array([100.0, 100.2], dtype=np.float32),
                np.array([200.0, 200.1, 200.2, 200.9], dtype=np.float32),
                np.array([300.0, 300.1, 300.2, 300.3], dtype=np.float32),
            ]
        )
        # Verify bins 0 and 1 with standard 1.5/1.5 cutoffs first.
        out_idx, out_peaks, out_mz = consensus._outlier_rejection(
            bins_indices[:2],
            nb.typed.List([bins_peaks[0], bins_peaks[1]]),
            nb.typed.List([bins_mz[0], bins_mz[1]]),
            1.5,
            1.5,
        )
        np.testing.assert_array_equal(out_idx, [0, 1])
        np.testing.assert_allclose(out_peaks, [0.5, 0.5], atol=1e-5)
        np.testing.assert_allclose(out_mz, [100.1, 200.1], atol=1e-4)

        # Now verify a bin emptied by aggressive cutoffs is omitted.
        out_idx2, out_peaks2, out_mz2 = consensus._outlier_rejection(
            np.array([0, 2], dtype=np.int32),
            nb.typed.List([bins_peaks[0], bins_peaks[2]]),
            nb.typed.List([bins_mz[0], bins_mz[2]]),
            0.05,
            0.05,
        )
        np.testing.assert_array_equal(out_idx2, [0])
        assert len(out_peaks2) == len(out_mz2) == 1


# ----------------------------------------------------------------------
# _construct_average_spectrum, _get_cluster_average, dispatch
# ----------------------------------------------------------------------


class TestAverageAndRepresentativeDispatch:
    def test_construct_average_spectrum_and_tuple_conversion(self):
        bins_indices = np.array([5, 10], dtype=np.int32)
        bins_peaks = np.array([0.5, 0.7], dtype=np.float32)
        bins_mz = np.array([100.0, 101.0], dtype=np.float32)
        prec_mz, charge, mz, inten, rt, cid = (
            consensus._construct_average_spectrum(
                bins_indices,
                bins_peaks,
                bins_mz,
                avg_precursor_mz=200.0,
                charge=2,
                avg_rt=60.0,
                cluster=5,
            )
        )
        assert prec_mz == pytest.approx(200.0)
        assert charge == 2
        assert rt == pytest.approx(60.0)
        assert cid == 5
        np.testing.assert_array_equal(mz, bins_mz)
        np.testing.assert_array_equal(inten, bins_peaks)

    def test_consensus_tuples_from_columns_handles_none_and_nan_charge(self):
        tuples = consensus._consensus_tuples_from_columns(
            precursor_mzs=[500.0, 600.0, 700.0],
            precursor_charges=[2, np.nan, None],
            mzs=[
                np.array([100.0], np.float64),
                np.array([200.0], np.float32),
                np.array([300.0], np.float32),
            ],
            intensities=[
                np.array([1.0], np.float64),
                np.array([0.5], np.float32),
                np.array([0.5], np.float32),
            ],
            retention_times=[10.0, np.nan, 30.0],
            cluster_ids=[0, 1, 2],
        )
        assert tuples[0].precursor_charge == np.int32(2)
        assert tuples[0].mz.dtype == np.float32
        assert tuples[0].intensity.dtype == np.float32
        assert math.isnan(tuples[1].precursor_charge)
        assert math.isnan(tuples[1].retention_time)
        assert math.isnan(tuples[2].precursor_charge)
        assert all(t.mz_split is None for t in tuples)

    def test_average_cluster_and_singleton_with_order_map_and_nan_rt(self):
        # Cluster 0 (specs 2, 0): averaged peaks, mean precursor,
        # NaN RT propagates because one member has NaN RT.
        # Cluster 1 (spec 1): singleton passthrough unchanged.
        s0 = _spec(
            [100.2, 200.2], [0.6, 0.8], precursor_mz=500.4, normalize=False
        )
        s1 = _spec(
            [150.0, 250.0],
            [0.3, 0.9],
            precursor_mz=600.0,
            charge=None,
            normalize=False,
        )
        s2 = _spec(
            [100.0, 200.0], [0.4, 0.6], precursor_mz=500.0, normalize=False
        )

        labels = np.array([0, 0, 1], dtype=np.int32)
        order_map = np.array([2, 0, 1], dtype=np.int64)
        rts = np.array([10.0, np.nan, 42.0], dtype=np.float32)
        params = {
            "min_mz": 50.0,
            "max_mz": 300.0,
            "bin_size": 1.0,
            "outlier_cutoff_lower": 1.5,
            "outlier_cutoff_upper": 1.5,
        }
        reps = consensus._get_representative_spectra(
            [s0, s1, s2], labels, rts, order_map, "average", params
        )
        assert len(reps) == 2

        # Cluster 0: mean precursor 500.2, mean m/z [100.1, 200.1],
        # mean intensity [0.5, 0.7] (not L2-renormalized), NaN RT.
        assert reps[0].cluster_id == 0
        assert reps[0].precursor_mz == pytest.approx(500.2, abs=1e-4)
        assert reps[0].precursor_charge == 2
        assert math.isnan(reps[0].retention_time)
        np.testing.assert_allclose(reps[0].mz, [100.1, 200.1], atol=1e-4)
        np.testing.assert_allclose(reps[0].intensity, [0.5, 0.7], atol=1e-5)

        # Cluster 1: singleton passthrough with None charge -> NaN.
        assert reps[1].cluster_id == 1
        assert math.isnan(reps[1].precursor_charge)
        assert reps[1].retention_time == pytest.approx(42.0)
        np.testing.assert_allclose(reps[1].mz, [150.0, 250.0])

    def test_average_with_no_surviving_bins_and_nan_charge(self):
        s1 = _spec([100.0], [1.0], charge=np.nan)
        s2 = _spec([200.0], [1.0], charge=np.nan)
        reps = consensus._get_representative_spectra(
            [s1, s2],
            np.array([0, 0], dtype=np.int32),
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            "average",
            {
                "min_mz": 50.0,
                "max_mz": 300.0,
                "bin_size": 1.0,
                "outlier_cutoff_lower": 1.5,
                "outlier_cutoff_upper": 1.5,
            },
        )
        assert len(reps) == 1
        assert math.isnan(reps[0].precursor_charge)
        assert len(reps[0].mz) == 0
        assert len(reps[0].intensity) == 0

    def test_invalid_consensus_method_raises(self, mock_spectra):
        with pytest.raises(
            ValueError, match="Unknown consensus spectrum method"
        ):
            consensus._get_representative_spectra(
                mock_spectra(2),
                np.array([0, 0], dtype=np.int32),
                np.array([10.0, 20.0], dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                "nonexistent",
                {},
            )


class TestTypedListToNumpy:
    def test_non_empty_and_empty(self):
        lst = nb.typed.List.empty_list(nb.types.float32)
        empty = consensus.typed_list_to_numpy(lst)
        assert empty.dtype == np.float32
        assert empty.shape == (0,)

        for v in (1.0, 2.5, 3.5):
            lst.append(np.float32(v))
        arr = consensus.typed_list_to_numpy(lst)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.array([1.0, 2.5, 3.5], np.float32)
        )
