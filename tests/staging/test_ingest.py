"""Unit tests for falcon.staging.ingest (globs, routing, writes)."""

import functools
import json
import logging
import threading

import lance
import numpy as np
import pyarrow as pa
import pytest

from falcon import ms_io, preprocessing, staging
from falcon.dataset import (
    SPECTRUM_SCHEMA,
    bucket_dataset_path,
    spectra_dir,
)
from falcon.staging import ingest as staging_ingest
from tests.helpers import LOW_QUALITY_PEAKS, TOY_PEAKS_5

_TOY_PROC = functools.partial(
    preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
)


@pytest.fixture
def staging_work_dir(tmp_path):
    """Create a temporary work dir with a ``spectra`` subfolder."""
    spectra_dir(tmp_path).mkdir()
    return tmp_path


# ----------------------------------------------------------------------
# Public exports, input globs, and charge router
# ----------------------------------------------------------------------


class TestResolveInputFiles:
    def test_public_exports_match_expected_surface(self):
        assert staging.__all__ == [
            "STAGING_CACHE_FILENAME",
            "STAGING_CACHE_SCHEMA_VERSION",
            "ChargeRouter",
            "build_staging_manifest",
            "deserialize_cached_buckets",
            "diff_staging_manifests",
            "load_or_stage_buckets",
            "resolve_input_files",
            "serialize_cached_buckets",
            "stage_spectra",
        ]

    def test_expands_globs_sorted_deterministically_and_preserves_duplicates(
        self, tmp_path
    ):
        (tmp_path / "b.mgf").write_text("", encoding="utf-8")
        (tmp_path / "a.mgf").write_text("", encoding="utf-8")
        resolved = staging.resolve_input_files(
            [str(tmp_path / "*.mgf"), str(tmp_path / "a.mgf")]
        )
        assert resolved == [
            tmp_path / "a.mgf",
            tmp_path / "b.mgf",
            tmp_path / "a.mgf",
        ]

    def test_empty_pattern_list_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            staging.resolve_input_files([])

    def test_unmatched_pattern_raises_file_not_found_per_pattern(
        self, tmp_path
    ):
        existing = tmp_path / "good.mgf"
        existing.write_text("", encoding="utf-8")
        missing_pat = str(tmp_path / "missing_*.mgf")

        with pytest.raises(FileNotFoundError, match="matched no files"):
            staging.resolve_input_files([missing_pat])

        # Fails fast even when an earlier pattern matched valid files.
        with pytest.raises(FileNotFoundError, match="missing_\\*\\.mgf"):
            staging.resolve_input_files([str(existing), missing_pat])

    def test_broken_symlink_raises_file_not_found(self, tmp_path):
        broken = tmp_path / "dangling.mgf"
        broken.symlink_to(tmp_path / "nonexistent_target.mgf")
        with pytest.raises(
            FileNotFoundError, match="not an existing regular file"
        ):
            staging.resolve_input_files([str(broken)])

    def test_rejects_literal_and_glob_matched_directories(self, tmp_path):
        subdir = tmp_path / "nested"
        subdir.mkdir()
        (tmp_path / "a.mgf").write_text("", encoding="utf-8")

        with pytest.raises(IsADirectoryError, match="is a directory"):
            staging.resolve_input_files([str(subdir)])

        with pytest.raises(IsADirectoryError, match="is a directory"):
            staging.resolve_input_files([str(tmp_path / "*")])


class TestChargeRouter:
    def test_auto_mode_routes_each_charge_to_singleton_bucket(self):
        router = staging.ChargeRouter(None)
        assert router.auto is True
        assert router.canonical_buckets is None
        assert router.route(2) == (2,)
        assert router.route(-1) == (-1,)
        assert router.route(None) == ("unknown",)

    def test_numpy_integer_charge_coerced_to_plain_int_for_json(self):
        auto_router = staging.ChargeRouter(None)
        pos_bucket = auto_router.route(np.int8(2))
        neg_bucket = auto_router.route(np.int32(-1))
        assert pos_bucket == (2,)
        assert neg_bucket == (-1,)
        assert isinstance(pos_bucket, tuple)
        assert isinstance(neg_bucket, tuple)
        assert type(pos_bucket[0]) is int
        assert type(neg_bucket[0]) is int
        serialized = staging.serialize_cached_buckets([pos_bucket, neg_bucket])
        assert json.loads(json.dumps(serialized)) == [[2], [-1]]

        explicit_router = staging.ChargeRouter([(2, 3)])
        assert explicit_router.route(np.int8(2)) == (2, 3)

    def test_explicit_buckets_route_single_multi_unknown_and_other(self):
        router = staging.ChargeRouter([(1,), (3, 2), ("unknown",), "other"])
        assert router.auto is False
        assert router.canonical_buckets == (
            (1,),
            (2, 3),
            ("unknown",),
            "other",
        )
        assert router.route(1) == (1,)
        assert router.route(2) == (2, 3)
        assert router.route(3) == (2, 3)
        assert router.route(None) == ("unknown",)
        assert router.route(5) == "other"

    def test_explicit_buckets_without_other_exclude_unmatched(self):
        router = staging.ChargeRouter([(1,), (2,)])
        assert router.route(1) == (1,)
        assert router.route(3) is None
        assert router.route(None) is None

    def test_explicit_other_catches_unknown_when_not_explicitly_listed(self):
        router = staging.ChargeRouter([(2,), "other"])
        assert router.route(2) == (2,)
        assert router.route(None) == "other"
        assert router.route(4) == "other"

    def test_mixed_numeric_and_unknown_bucket(self):
        router = staging.ChargeRouter([("unknown", 3)])
        assert router.canonical_buckets == ((3, "unknown"),)
        assert router.route(3) == (3, "unknown")
        assert router.route(None) == (3, "unknown")

    @pytest.mark.parametrize(
        ("bad_buckets", "match"),
        [
            ([{1, 2}, {2, 3}], "more than one bucket"),
            ([[1, 1]], "Duplicate charge"),
            ([{1}, "other", "other"], "other"),
            ([], "must not be empty"),
        ],
    )
    def test_rejects_invalid_overlapping_or_duplicate_buckets(
        self, bad_buckets, match
    ):
        with pytest.raises(ValueError, match=match):
            staging.ChargeRouter(bad_buckets)


# ----------------------------------------------------------------------
# Columnar Arrow conversion, Lance writing, and batch writer concurrency
# ----------------------------------------------------------------------


class TestLanceWritingAndBatchWriter:
    def test_columnar_arrow_table_matches_pylist_and_spectrum_schema(self):
        rows = [
            {
                "identifier": "s:1",
                "precursor_mz": np.float32(500.0),
                "precursor_charge": 2,
                "mz": np.array([100.0, 200.0], dtype=np.float32),
                "intensity": np.array([0.6, 0.8], dtype=np.float32),
                "retention_time": np.float32(10.0),
            },
            {
                "identifier": "s:2",
                "precursor_mz": 600.25,
                "precursor_charge": None,
                "mz": np.array([], dtype=np.float32),
                "intensity": np.array([], dtype=np.float32),
                "retention_time": None,
            },
            {
                "identifier": "s:3",
                "precursor_mz": 700.5,
                "precursor_charge": -2,
                "mz": [150.0, 250.0, 350.0],
                "intensity": [0.5, 0.5, 0.707],
                "retention_time": 42.5,
            },
        ]
        table = staging_ingest._spectra_to_arrow_table(rows)
        reference = pa.Table.from_pylist(rows, SPECTRUM_SCHEMA)
        assert table.schema.equals(SPECTRUM_SCHEMA, check_metadata=True)
        assert table.equals(reference)

        empty_table = staging_ingest._spectra_to_arrow_table([])
        assert empty_table.schema.equals(SPECTRUM_SCHEMA, check_metadata=True)
        assert empty_table.equals(pa.Table.from_pylist([], SPECTRUM_SCHEMA))

    def test_spectra_to_arrow_table_rejects_mismatched_peak_lengths(
        self, spectrum_row
    ):
        bad = spectrum_row("s:bad", precursor_charge=2)
        bad["intensity"] = bad["intensity"][:-1]
        with pytest.raises(ValueError, match="mismatched mz"):
            staging_ingest._spectra_to_arrow_table([bad])

    def test_compute_list_offsets_raises_overflow_when_exceeding_int32_limit(
        self, monkeypatch
    ):
        monkeypatch.setattr(staging_ingest, "INT32_MAX", 5)
        arrays = [
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        with pytest.raises(OverflowError, match="offset limit"):
            staging_ingest._compute_list_offsets(arrays)

    def test_creates_appends_and_overwrites_at_explicit_work_dir(
        self, tmp_path, spectrum_row
    ):
        explicit_dir = tmp_path / "target"
        spectra_dir(explicit_dir).mkdir(parents=True)

        state = staging_ingest._BucketState()
        count1 = staging_ingest._write_to_dataset(
            [
                spectrum_row("s1", precursor_charge=2),
                spectrum_row("s2", precursor_charge=2),
            ],
            (2,),
            state,
            explicit_dir,
        )
        assert state.initialized is True
        count2 = staging_ingest._write_to_dataset(
            [spectrum_row("s3", precursor_charge=2)],
            (2,),
            state,
            explicit_dir,
        )
        assert count1 == 2
        assert count2 == 1
        target_ds = bucket_dataset_path(explicit_dir, (2,))
        assert target_ds.exists()
        assert lance.dataset(str(target_ds)).count_rows() == 3

        # A fresh BucketState overwrites prior rows cleanly.
        fresh_state = staging_ingest._BucketState()
        count3 = staging_ingest._write_to_dataset(
            [spectrum_row("s4", precursor_charge=2)],
            (2,),
            fresh_state,
            explicit_dir,
        )
        assert count3 == 1
        assert lance.dataset(str(target_ds)).count_rows() == 1

    def test_batch_writer_serializes_concurrent_batches_per_bucket(
        self, staging_work_dir, monkeypatch, spectrum_row
    ):
        monkeypatch.setattr(staging_ingest, "_LANCE_WRITE_BATCH_SIZE", 2)
        first_write_started = threading.Event()
        release_first_write = threading.Event()
        orig_write = staging_ingest._write_to_dataset

        def _gated_write(batch, bucket, state, work_dir):
            if bucket == (2,) and not first_write_started.is_set():
                first_write_started.set()
                assert release_first_write.wait(timeout=5.0)
            return orig_write(batch, bucket, state, work_dir)

        monkeypatch.setattr(staging_ingest, "_write_to_dataset", _gated_write)

        with staging_ingest._LanceBatchWriter(
            staging.ChargeRouter(None), staging_work_dir, max_workers=2
        ) as writer:
            writer.add_spectra(
                [spectrum_row(f"c2_{i}", precursor_charge=2) for i in range(2)]
            )
            assert first_write_started.wait(timeout=5.0)
            # Subsequent batches for the same bucket submit without blocking
            # the main thread; releasing the first write completes all rows.
            writer.add_spectra(
                [
                    spectrum_row(f"c2_{i}", precursor_charge=2)
                    for i in range(2, 7)
                ]
            )
            release_first_write.set()
            writer.add_spectra(
                [spectrum_row(f"c1_{i}", precursor_charge=1) for i in range(4)]
            )

        assert writer.excluded_count == 0
        assert (
            lance.dataset(
                str(bucket_dataset_path(staging_work_dir, (2,)))
            ).count_rows()
            == 7
        )
        assert (
            lance.dataset(
                str(bucket_dataset_path(staging_work_dir, (1,)))
            ).count_rows()
            == 4
        )

    def test_batch_writer_cancels_and_joins_on_mid_ingestion_error(
        self, staging_work_dir, monkeypatch, spectrum_row
    ):
        monkeypatch.setattr(staging_ingest, "_LANCE_WRITE_BATCH_SIZE", 1)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("mid-stream lance error")

        monkeypatch.setattr(staging_ingest, "_write_to_dataset", _boom)
        with (
            pytest.raises(RuntimeError, match="mid-stream lance error"),
            staging_ingest._LanceBatchWriter(
                staging.ChargeRouter(None), staging_work_dir, max_workers=2
            ) as writer,
        ):
            writer.add_spectra(
                [spectrum_row(f"s_{i}", precursor_charge=2) for i in range(3)]
            )

    def test_submit_failure_releases_semaphore_and_leaves_bucket_clean(
        self, staging_work_dir, monkeypatch, spectrum_row
    ):
        monkeypatch.setattr(staging_ingest, "_LANCE_WRITE_BATCH_SIZE", 1)
        writer = staging_ingest._LanceBatchWriter(
            staging.ChargeRouter(None), staging_work_dir, max_workers=1
        )
        max_permits = 2  # max(2, max_workers * 2)

        def _fail_submit(*_args, **_kwargs):
            raise RuntimeError("executor rejected task")

        monkeypatch.setattr(writer._executor, "submit", _fail_submit)
        with (
            pytest.raises(RuntimeError, match="executor rejected task"),
            writer,
        ):
            writer.add_spectrum(spectrum_row("s_0", precursor_charge=2))

        assert not bucket_dataset_path(staging_work_dir, (2,)).exists()
        assert writer.written_counts == {}
        # All semaphore permits were released (none leaked).
        acquired = [
            writer._inflight_sem.acquire(blocking=False)
            for _ in range(max_permits)
        ]
        assert acquired == [True] * max_permits
        assert writer._inflight_sem.acquire(blocking=False) is False
        for _ in range(max_permits):
            writer._inflight_sem.release()

    def test_writer_workers_decoupled_from_file_count(self):
        auto_router = staging.ChargeRouter(None)
        assert staging_ingest._writer_workers(auto_router, 1) == 1
        assert staging_ingest._writer_workers(auto_router, 4) == 4
        assert (
            staging_ingest._writer_workers(auto_router, 64)
            == staging_ingest._MAX_AUTO_WRITER_WORKERS
        )

        explicit_router = staging.ChargeRouter([(1,), (2, 3)])
        assert staging_ingest._writer_workers(explicit_router, 64) == 2
        assert staging_ingest._writer_workers(explicit_router, 1) == 1


# ----------------------------------------------------------------------
# Spectrum reading, single-file streaming, and multi-file staging
# ----------------------------------------------------------------------


class TestReadAndStageSpectra:
    def test_read_spectra_counts_accepted_and_rejected(
        self, tmp_path, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            tmp_path / "mix.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1),
                mgf_block(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=2),
            ],
        )
        accepted, rejected = staging_ingest._read_spectra(mgf, _TOY_PROC)
        assert len(accepted) == 1
        assert accepted[0]["identifier"] == "mix:scan:1"
        assert rejected == 1

    def test_single_file_streams_incrementally_without_joblib(
        self, staging_work_dir, mgf_block, mgf_writer, monkeypatch, caplog
    ):
        mgf = mgf_writer(
            staging_work_dir / "stream.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0 + i, charge=2, scan=i + 1)
                for i in range(5)
            ]
            + [mgf_block(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=99)],
        )

        def _forbid_parallel(*_args, **_kwargs):
            raise AssertionError(
                "Single-file staging should stream without loky pool"
            )

        monkeypatch.setattr(
            staging_ingest, "loky_unordered_pool", _forbid_parallel
        )
        monkeypatch.setattr(staging_ingest, "_LANCE_WRITE_BATCH_SIZE", 2)

        submitted_at_yield = []
        yielded_count = 0
        orig_get_spectra = ms_io.get_spectra
        orig_submit = staging_ingest._LanceBatchWriter._submit

        def _counting_get_spectra(path):
            nonlocal yielded_count
            for spec in orig_get_spectra(path):
                yielded_count += 1
                yield spec

        def _recording_submit(self, bucket, batch):
            submitted_at_yield.append((yielded_count, len(batch)))
            return orig_submit(self, bucket, batch)

        monkeypatch.setattr(ms_io, "get_spectra", _counting_get_spectra)
        monkeypatch.setattr(
            staging_ingest._LanceBatchWriter, "_submit", _recording_submit
        )

        with caplog.at_level(logging.INFO, logger="falcon"):
            valid = staging.stage_spectra(
                [mgf], staging_work_dir, _TOY_PROC, None
            )

        assert valid == [(2,)]
        ds_path = bucket_dataset_path(staging_work_dir, (2,))
        assert lance.dataset(str(ds_path)).count_rows() == 5
        # Full batches dispatch mid-stream before the file finishes.
        assert submitted_at_yield == [(2, 2), (4, 2), (6, 1)]
        assert any(
            "Staged 5 spectra from 1 peak file(s)" in r.message
            for r in caplog.records
        )
        assert any(
            "Skipped 1 low-quality spectra" in r.message
            for r in caplog.records
        )

    def test_multi_file_with_single_worker_streams_without_loky(
        self, staging_work_dir, mgf_block, mgf_writer, monkeypatch
    ):
        mgf_a = mgf_writer(
            staging_work_dir / "s1.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
        )
        mgf_b = mgf_writer(
            staging_work_dir / "s2.mgf",
            [mgf_block(TOY_PEAKS_5, 501.0, charge=3, scan=1)],
        )

        def _forbid_parallel(*_args, **_kwargs):
            raise AssertionError("1-worker staging must stream directly")

        monkeypatch.setattr(staging_ingest, "effective_n_workers", lambda: 1)
        monkeypatch.setattr(
            staging_ingest, "loky_unordered_pool", _forbid_parallel
        )
        valid = staging.stage_spectra(
            [mgf_a, mgf_b], staging_work_dir, _TOY_PROC, None
        )
        assert valid == [(2,), (3,)]

    def test_multi_file_stages_in_parallel_across_buckets(
        self, staging_work_dir, mgf_block, mgf_writer, caplog
    ):
        mgf_a = mgf_writer(
            staging_work_dir / "part_a.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1),
                mgf_block(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=2),
            ],
        )
        mgf_b = mgf_writer(
            staging_work_dir / "part_b.mgf",
            [
                mgf_block(TOY_PEAKS_5, 501.0, charge=2, scan=1),
                mgf_block(TOY_PEAKS_5, 600.0, charge=3, scan=2),
            ],
        )

        with caplog.at_level(logging.INFO, logger="falcon"):
            valid = staging.stage_spectra(
                [mgf_a, mgf_b], staging_work_dir, _TOY_PROC, None
            )

        assert valid == [(2,), (3,)]
        assert (
            lance.dataset(
                str(bucket_dataset_path(staging_work_dir, (2,)))
            ).count_rows()
            == 2
        )
        assert (
            lance.dataset(
                str(bucket_dataset_path(staging_work_dir, (3,)))
            ).count_rows()
            == 1
        )
        assert any(
            "Staged 3 spectra from 2 peak file(s)" in r.message
            for r in caplog.records
        )
        assert any(
            "Skipped 1 low-quality spectra" in r.message
            for r in caplog.records
        )

    def test_stage_spectra_auto_mode_sorts_numeric_charges_then_unknown(
        self, tmp_path, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            tmp_path / "auto.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=3, scan=1),
                mgf_block(TOY_PEAKS_5, 500.0, charge=-1, scan=2),
                mgf_block(TOY_PEAKS_5, 500.0, charge=None, scan=3),
                mgf_block(TOY_PEAKS_5, 500.0, charge=1, scan=4),
            ],
        )
        work_dir = tmp_path / "work"

        valid = staging.stage_spectra([mgf], work_dir, _TOY_PROC, None)
        assert valid == [(-1,), (1,), (3,), ("unknown",)]
        for bucket in valid:
            ds_path = bucket_dataset_path(work_dir, bucket)
            assert ds_path.is_dir()
            assert lance.dataset(str(ds_path)).count_rows() == 1

    def test_repeated_direct_stage_spectra_overwrites_without_duplicates(
        self, staging_work_dir, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            staging_work_dir / "repeat.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1),
                mgf_block(TOY_PEAKS_5, 501.0, charge=2, scan=2),
            ],
        )
        assert staging.stage_spectra(
            [mgf], staging_work_dir, _TOY_PROC, None
        ) == [(2,)]
        assert staging.stage_spectra(
            [mgf], staging_work_dir, _TOY_PROC, None
        ) == [(2,)]
        ds_path = bucket_dataset_path(staging_work_dir, (2,))
        assert lance.dataset(str(ds_path)).count_rows() == 2

    def test_stage_spectra_explicit_buckets_order_skip_and_excluded_log(
        self, staging_work_dir, mgf_block, mgf_writer, caplog
    ):
        mgf = mgf_writer(
            staging_work_dir / "sample.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1),
                mgf_block(TOY_PEAKS_5, 500.0, charge=4, scan=2),
                mgf_block(TOY_PEAKS_5, 500.0, charge=1, scan=3),
            ],
        )

        # Explicit order ((2, 3) before (1,)) is preserved;
        # empty (5,) is skipped.
        valid = staging.stage_spectra(
            [mgf],
            staging_work_dir,
            _TOY_PROC,
            [(2, 3), (5,), (1,), "other"],
        )
        assert valid == [(2, 3), (1,), "other"]

        with caplog.at_level(logging.INFO, logger="falcon"):
            valid_drop = staging.stage_spectra(
                [mgf], staging_work_dir, _TOY_PROC, [(2, 3), (1,)]
            )
        assert valid_drop == [(2, 3), (1,)]
        assert any("Excluded 1 spectra" in r.message for r in caplog.records)

        with pytest.raises(ValueError, match="more than one bucket"):
            staging.stage_spectra(
                [mgf], staging_work_dir, _TOY_PROC, [{1, 2}, {2, 3}]
            )

        with pytest.raises(ValueError, match="other"):
            staging.stage_spectra(
                [mgf], staging_work_dir, _TOY_PROC, [{1}, "other", "other"]
            )

    def test_stage_spectra_empty_input_list_raises_file_not_found(
        self, staging_work_dir
    ):
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            staging.stage_spectra([], staging_work_dir, _TOY_PROC, None)

    def test_stage_spectra_propagates_writer_thread_error(
        self, staging_work_dir, mgf_block, mgf_writer, monkeypatch
    ):
        mgf = mgf_writer(
            staging_work_dir / "err.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("lance disk full")

        monkeypatch.setattr(staging_ingest, "_write_to_dataset", _boom)
        with pytest.raises(RuntimeError, match="lance disk full"):
            staging.stage_spectra([mgf], staging_work_dir, _TOY_PROC, None)

    def test_stage_spectra_propagates_reader_error(self, staging_work_dir):
        missing = staging_work_dir / "missing.mgf"
        with pytest.raises(FileNotFoundError, match="Non-existing peak file"):
            staging.stage_spectra([missing], staging_work_dir, _TOY_PROC, None)

    def test_stage_spectra_relative_multi_file_after_chdir(
        self, tmp_path, mgf_block, mgf_writer, monkeypatch
    ):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        work_a = tmp_path / "work_a"
        work_b = tmp_path / "work_b"
        for d in (dir_a, dir_b, work_a, work_b):
            d.mkdir()
        for d, prefix in ((dir_a, "a"), (dir_b, "b")):
            mgf_writer(
                d / f"{prefix}1.mgf",
                [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
            )
            mgf_writer(
                d / f"{prefix}2.mgf",
                [mgf_block(TOY_PEAKS_5, 501.0, charge=2, scan=2)],
            )

        monkeypatch.setattr(staging_ingest, "effective_n_workers", lambda: 2)
        monkeypatch.chdir(dir_a)
        assert staging.stage_spectra(
            ["a1.mgf", "a2.mgf"], work_a, _TOY_PROC, None
        ) == [(2,)]

        # Warm loky workers still have dir_a as cwd; parent resolution
        # keeps relative paths in dir_b valid.
        monkeypatch.chdir(dir_b)
        assert staging.stage_spectra(
            ["b1.mgf", "b2.mgf"], work_b, _TOY_PROC, None
        ) == [(2,)]
        ds = lance.dataset(str(bucket_dataset_path(work_b, (2,))))
        assert ds.count_rows() == 2
