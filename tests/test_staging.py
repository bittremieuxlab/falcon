"""Unit tests for falcon.staging (globs, Lance staging, cache)."""

import functools
import json
import os
import queue
import threading
from types import SimpleNamespace

import lance
import pytest
from conftest import LOW_QUALITY_PEAKS, TOY_PEAKS_5

from falcon import preprocessing, staging
from falcon.dataset import (
    bucket_dataset_name,
    bucket_dataset_path,
    spectra_dir,
)


def _bump_file(path, content):
    """Rewrite ``path`` and advance its mtime so cache stats change."""
    path.write_text(content, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _write_cache_manifest(work_dir, payload):
    """Write ``staging_cache.json`` under ``spectra_dir(work_dir)``."""
    cache_path = spectra_dir(work_dir) / staging.STAGING_CACHE_FILENAME
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return cache_path


def _forbid_stage_spectra(monkeypatch):
    """Fail the test if ``staging.stage_spectra`` is invoked."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("stage_spectra should not be called")

    monkeypatch.setattr(staging, "stage_spectra", _fail)


def _row(spectrum_row, identifier, charge):
    """Build a staged spectrum row for routing and writing tests."""
    return spectrum_row(
        identifier, 500.0, [100.0, 200.0, 300.0], [0.5, 0.6, 0.7], charge
    )


def _run_write_worker(spectra, router, work_dir, stop_event=None):
    """Run ``_write_spectra_lance`` and return staged row counts."""
    q = queue.Queue()
    for spec in spectra:
        q.put(spec)
    q.put(None)

    locks = staging._PerChargeLockRegistry()
    staging._write_spectra_lance(q, locks, router, work_dir, stop_event)

    s_dir = spectra_dir(work_dir)
    counts = {}
    if s_dir.exists():
        for p in s_dir.iterdir():
            if p.suffix == ".lance":
                counts[p.name] = lance.dataset(str(p)).count_rows()
    return counts


@pytest.fixture
def staging_work_dir(tmp_path):
    """Create a temporary work dir with a ``spectra`` subfolder."""
    spectra_dir(tmp_path).mkdir()
    return tmp_path


@pytest.fixture
def cache_env(tmp_path, config_factory):
    """Provision a temporary cache dir, dummy input, and helpers."""
    input_file = tmp_path / "in.mgf"
    input_file.write_text("v1", encoding="utf-8")
    s_dir = spectra_dir(tmp_path)
    s_dir.mkdir()

    def make_cfg(output_name="out", io=None, **sections):
        io_opts = {"work_dir": str(tmp_path), **(io or {})}
        return config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / output_name,
            io=io_opts,
            **sections,
        )

    def seed_cache(cfg=None, buckets=((2,),), create_dirs=False):
        active_cfg = cfg or make_cfg()
        manifest = staging.build_staging_manifest(
            active_cfg, [input_file], buckets=list(buckets)
        )
        if create_dirs:
            for bucket in buckets:
                (s_dir / bucket_dataset_name(bucket)).mkdir(exist_ok=True)
        _write_cache_manifest(tmp_path, manifest)
        return active_cfg, manifest

    return SimpleNamespace(
        work_dir=tmp_path,
        input_file=input_file,
        s_dir=s_dir,
        make_cfg=make_cfg,
        seed_cache=seed_cache,
    )


# ----------------------------------------------------------------------
# Input globs, charge router, and lock registry
# ----------------------------------------------------------------------


class TestResolveInputFiles:
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

    def test_unmatched_glob_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            staging.resolve_input_files([str(tmp_path / "missing_*.mgf")])


class TestChargeRouterAndLocks:
    def test_auto_mode_routes_each_charge_to_singleton_bucket(self):
        router = staging.ChargeRouter(None)
        assert router.auto is True
        assert router.canonical_buckets is None
        assert router.route(2) == (2,)
        assert router.route(-1) == (-1,)
        assert router.route(None) == ("unknown",)

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

    def test_lock_registry_is_thread_safe(self):
        reg = staging._PerChargeLockRegistry()
        assert reg.get((1,)) is reg.get((1,))
        assert reg.get((1,)) is not reg.get((2,))

        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(reg.get((2,)))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(lock) for lock in results}) == 1


# ----------------------------------------------------------------------
# Staging manifest build, diff, and bucket deserialization
# ----------------------------------------------------------------------


class TestStagingManifest:
    def test_manifest_and_diff_detect_only_staging_changes(
        self, tmp_path, config_factory
    ):
        f1 = tmp_path / "a.mgf"
        f2 = tmp_path / "b.mgf"
        f1.write_text("data1", encoding="utf-8")
        f2.write_text("data2", encoding="utf-8")

        base_cfg = config_factory(
            input_filenames=[str(f1)],
            output_filename=tmp_path / "out1",
        )
        base_manifest = staging.build_staging_manifest(
            base_cfg, [f1], buckets=[(2,), ("unknown",), "other"]
        )
        assert (
            base_manifest["schema_version"]
            == staging.STAGING_CACHE_SCHEMA_VERSION
        )
        assert base_manifest["buckets"] == [[2], ["unknown"], "other"]
        assert staging.deserialize_cached_buckets(
            base_manifest["buckets"]
        ) == [(2,), ("unknown",), "other"]

        # Clustering/consensus/output sweeps must NOT invalidate cache.
        sweep_cfg = config_factory(
            input_filenames=[str(f1)],
            output_filename=tmp_path / "out2",
            io={"export_representatives": True},
            clustering={
                "distance_threshold": 0.25,
                "linkage": "average",
                "rt_tol": 10.0,
                "min_matched_peaks": 6,
                "batch_size": 512,
            },
            consensus={"method": "average"},
        )
        assert (
            staging.diff_staging_manifests(
                base_manifest, staging.build_staging_manifest(sweep_cfg, [f1])
            )
            == []
        )

        # Preprocessing, fragment_tol, charge buckets, and inputs
        # invalidate the staging cache.
        prep_cfg = config_factory(
            input_filenames=[str(f1)],
            output_filename=tmp_path / "out1",
            preprocessing={"min_mz": 50.0},
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(prep_cfg, [f1])
        ) == ["preprocessing"]

        frag_cfg = config_factory(
            input_filenames=[str(f1)],
            output_filename=tmp_path / "out1",
            clustering={"fragment_tol": 0.02},
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(frag_cfg, [f1])
        ) == ["fragment_tol"]

        bucket_cfg = config_factory(
            input_filenames=[str(f1)],
            output_filename=tmp_path / "out1",
            clustering={"precursor_charge_buckets": [[2]]},
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(bucket_cfg, [f1])
        ) == ["precursor_charge_buckets"]

        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(base_cfg, [f1, f2])
        ) == ["inputs"]

        _bump_file(f1, "modified_longer_content")
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(base_cfg, [f1])
        ) == ["inputs"]

        assert staging.diff_staging_manifests(["legacy"], base_manifest) == [
            "schema_version"
        ]
        bad_ver = {**base_manifest, "schema_version": 999}
        assert staging.diff_staging_manifests(bad_ver, base_manifest) == [
            "schema_version"
        ]

    def test_deserialize_cached_buckets_handles_empty_and_rejects_corrupted(
        self,
    ):
        assert staging.deserialize_cached_buckets([]) == []
        with pytest.raises(ValueError, match="buckets"):
            staging.deserialize_cached_buckets("not-a-list")
        with pytest.raises(ValueError, match="bucket"):
            staging.deserialize_cached_buckets([123])
        with pytest.raises(ValueError, match="bucket"):
            staging.deserialize_cached_buckets([[True]])

    def test_write_json_atomic_cleans_up_temp_file_on_error(self, tmp_path):
        target = tmp_path / staging.STAGING_CACHE_FILENAME
        with pytest.raises(TypeError):
            staging._write_json_atomic(target, {"bad": object()})
        assert not target.exists()
        assert list(tmp_path.glob(".staging_cache_*.tmp")) == []


# ----------------------------------------------------------------------
# Cache reuse, mismatch, legacy protection, and overwrite
# ----------------------------------------------------------------------


class TestLoadOrStageBuckets:
    def test_creates_spectra_dir_automatically_when_missing(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1", encoding="utf-8")
        fresh_work_dir = tmp_path / "fresh_work"
        assert not spectra_dir(fresh_work_dir).exists()

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(fresh_work_dir)},
        )
        monkeypatch.setattr(
            staging,
            "stage_spectra",
            lambda _inputs, _wd, _proc, _buckets: [(2,)],
        )

        buckets = staging.load_or_stage_buckets(
            cfg, fresh_work_dir, lambda _s: None
        )
        assert buckets == [(2,)]
        assert spectra_dir(fresh_work_dir).is_dir()
        assert (
            spectra_dir(fresh_work_dir) / staging.STAGING_CACHE_FILENAME
        ).is_file()

    def test_reuses_matching_cache_across_clustering_and_consensus_changes(
        self, cache_env, monkeypatch
    ):
        orig_cfg = cache_env.make_cfg(
            output_name="out1",
            clustering={"precursor_charge_buckets": [[1], [2]]},
        )
        cache_env.seed_cache(
            cfg=orig_cfg, buckets=[(1,), (2,)], create_dirs=True
        )

        sweep_cfg = cache_env.make_cfg(
            output_name="out2",
            io={"export_representatives": True},
            clustering={
                "precursor_charge_buckets": [[1], [2]],
                "distance_threshold": 0.25,
                "linkage": "average",
                "rt_tol": 15.0,
            },
            consensus={"method": "average"},
        )
        _forbid_stage_spectra(monkeypatch)
        assert staging.load_or_stage_buckets(
            sweep_cfg, cache_env.work_dir, lambda _s: None
        ) == [(1,), (2,)]

    @pytest.mark.parametrize(
        ("override_sections", "expected_section"),
        [
            ({"preprocessing": {"min_mz": 50.0}}, "preprocessing"),
            ({"clustering": {"fragment_tol": 0.02}}, "fragment_tol"),
            (
                {"clustering": {"precursor_charge_buckets": [[1]]}},
                "precursor_charge_buckets",
            ),
        ],
    )
    def test_mismatched_staging_config_raises_with_section_name(
        self, cache_env, override_sections, expected_section
    ):
        cache_env.seed_cache()
        changed_cfg = cache_env.make_cfg(**override_sections)
        with pytest.raises(ValueError, match=expected_section):
            staging.load_or_stage_buckets(
                changed_cfg, cache_env.work_dir, lambda _s: None
            )

    def test_modified_input_file_invalidates_staging_cache(self, cache_env):
        cfg, _ = cache_env.seed_cache()
        _bump_file(cache_env.input_file, "v2_modified")
        with pytest.raises(ValueError, match="inputs"):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    @pytest.mark.parametrize(
        ("corrupt_bytes", "expected_match"),
        [
            (b"{not valid json", "Corrupted staging cache manifest"),
            (b"\xff\xfe", "Corrupted staging cache manifest"),
            (
                json.dumps({"schema_version": 999, "buckets": []}).encode(),
                "schema_version",
            ),
        ],
    )
    def test_corrupted_staging_cache_json_raises_value_error(
        self, cache_env, corrupt_bytes, expected_match
    ):
        (cache_env.s_dir / staging.STAGING_CACHE_FILENAME).write_bytes(
            corrupt_bytes
        )
        cfg = cache_env.make_cfg()
        with pytest.raises(ValueError, match=expected_match):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    def test_invalid_buckets_in_matching_manifest_raises_actionable_error(
        self, cache_env
    ):
        cfg = cache_env.make_cfg()
        manifest = staging.build_staging_manifest(cfg, [cache_env.input_file])
        manifest["buckets"] = "bad"
        _write_cache_manifest(cache_env.work_dir, manifest)
        with pytest.raises(ValueError, match="Invalid staging cache buckets"):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    @pytest.mark.parametrize(
        "stale_name", ["charges.joblib", "spectra_charge_2.lance"]
    )
    def test_unverified_legacy_or_orphaned_cache_raises_without_overwrite(
        self, cache_env, stale_name
    ):
        stale_path = cache_env.s_dir / stale_name
        if stale_name.endswith(".lance"):
            stale_path.mkdir()
        else:
            stale_path.write_bytes(b"legacy")

        cfg = cache_env.make_cfg()
        with pytest.raises(ValueError, match="overwrite"):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    def test_overwrite_clears_cached_directories_and_writes_json_manifest(
        self, cache_env, monkeypatch
    ):
        stale_dir = cache_env.s_dir / "spectra_charge_99.lance"
        stale_dir.mkdir()
        (stale_dir / "data").write_text("stale", encoding="utf-8")
        (cache_env.s_dir / "charges.joblib").write_text(
            "stale", encoding="utf-8"
        )

        cfg = cache_env.make_cfg(io={"overwrite": True})
        monkeypatch.setattr(
            staging,
            "stage_spectra",
            lambda _inputs, _wd, _proc, _buckets: [(2,)],
        )

        buckets = staging.load_or_stage_buckets(
            cfg, cache_env.work_dir, lambda _s: None
        )
        assert buckets == [(2,)]
        assert not stale_dir.exists()
        assert not (cache_env.s_dir / "charges.joblib").exists()
        saved = json.loads(
            (cache_env.s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert saved["buckets"] == [[2]]
        assert saved["precursor_charge_buckets"] is None

    def test_input_modified_during_staging_triggers_cache_mismatch(
        self, cache_env, monkeypatch
    ):
        pre_stat = os.stat(cache_env.input_file)
        cfg = cache_env.make_cfg()

        def _stage_while_input_changes(_inputs, _wd, _proc, _buckets):
            _bump_file(cache_env.input_file, "v2_modified_during_staging")
            return [(2,), ("unknown",), "other"]

        monkeypatch.setattr(
            staging, "stage_spectra", _stage_while_input_changes
        )

        buckets = staging.load_or_stage_buckets(
            cfg, cache_env.work_dir, lambda _s: None
        )
        assert buckets == [(2,), ("unknown",), "other"]

        saved = json.loads(
            (cache_env.s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert saved["inputs"] == [
            {
                "path": str(cache_env.input_file.resolve()),
                "size": pre_stat.st_size,
                "mtime_ns": pre_stat.st_mtime_ns,
            }
        ]
        assert saved["buckets"] == [[2], ["unknown"], "other"]

        with pytest.raises(
            ValueError,
            match=r"mismatched: inputs.*--overwrite.*--work-dir",
        ):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    def test_fresh_staging_reuses_manifest_without_restatting_inputs(
        self, cache_env, monkeypatch
    ):
        cfg = cache_env.make_cfg(
            clustering={"precursor_charge_buckets": [[1, 2], "other"]}
        )

        build_calls = 0
        orig_build = staging.build_staging_manifest

        def _spy_build(*args, **kwargs):
            nonlocal build_calls
            build_calls += 1
            return orig_build(*args, **kwargs)

        def _fake_stage(_inputs, _wd, _proc, _buckets):
            for name in (
                "spectra_charge_1_2.lance",
                "spectra_charge_unknown.lance",
                "spectra_charge_other.lance",
            ):
                (cache_env.s_dir / name).mkdir()
            return [(2, 1), ("unknown",), "other"]

        monkeypatch.setattr(staging, "build_staging_manifest", _spy_build)
        monkeypatch.setattr(staging, "stage_spectra", _fake_stage)

        staging.load_or_stage_buckets(cfg, cache_env.work_dir, lambda _s: None)
        assert build_calls == 1

        saved = json.loads(
            (cache_env.s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        expected = staging.build_staging_manifest(
            cfg,
            [cache_env.input_file],
            buckets=[(2, 1), ("unknown",), "other"],
        )
        assert saved == expected

        _forbid_stage_spectra(monkeypatch)
        reused = staging.load_or_stage_buckets(
            cfg, cache_env.work_dir, lambda _s: None
        )
        assert reused == [(1, 2), ("unknown",), "other"]

    @pytest.mark.parametrize(
        ("cached_buckets", "disk_entries"),
        [
            ([(1,), (2,)], [("spectra_charge_1.lance", True)]),
            (
                [(2,)],
                [
                    ("spectra_charge_2.lance", True),
                    ("spectra_charge_3.lance", True),
                ],
            ),
            ([], [("spectra_charge_2.lance", True)]),
            ([(2,)], [("spectra_charge_2.lance", False)]),
        ],
    )
    def test_cached_buckets_mismatching_staged_datasets_raises(
        self, cache_env, cached_buckets, disk_entries
    ):
        for name, is_dir in disk_entries:
            path = cache_env.s_dir / name
            if is_dir:
                path.mkdir()
            else:
                path.write_text("not a dataset", encoding="utf-8")

        cfg, _ = cache_env.seed_cache(buckets=cached_buckets)
        with pytest.raises(
            ValueError,
            match=r"rerun with --overwrite or a different --work-dir",
        ):
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )

    def test_empty_cached_buckets_reused_when_no_staged_datasets_exist(
        self, cache_env, monkeypatch
    ):
        cfg, _ = cache_env.seed_cache(buckets=[])
        _forbid_stage_spectra(monkeypatch)
        assert (
            staging.load_or_stage_buckets(
                cfg, cache_env.work_dir, lambda _s: None
            )
            == []
        )


# ----------------------------------------------------------------------
# Lance writing, charge routing, and parallel staging
# ----------------------------------------------------------------------


class TestLanceWritingAndRouting:
    def test_creates_and_appends_at_explicit_work_dir(
        self, tmp_path, spectrum_row
    ):
        explicit_dir = tmp_path / "target"
        spectra_dir(explicit_dir).mkdir(parents=True)

        lock = threading.Lock()
        count1 = staging._write_to_dataset(
            [_row(spectrum_row, "s1", 2), _row(spectrum_row, "s2", 2)],
            (2,),
            lock,
            explicit_dir,
        )
        count2 = staging._write_to_dataset(
            [_row(spectrum_row, "s3", 2)],
            (2,),
            lock,
            explicit_dir,
        )
        assert count1 == 2
        assert count2 == 1
        target_ds = bucket_dataset_path(explicit_dir, (2,))
        assert target_ds.exists()
        assert lance.dataset(str(target_ds)).count_rows() == 3

    def test_explicit_single_multi_unknown_and_other_routing(
        self, staging_work_dir, spectrum_row
    ):
        router = staging.ChargeRouter([(1,), (2, 3), ("unknown",), "other"])
        spectra = [
            _row(spectrum_row, "a", 1),
            _row(spectrum_row, "b", 2),
            _row(spectrum_row, "c", 3),
            _row(spectrum_row, "d", None),
            _row(spectrum_row, "e", 5),
        ]
        counts = _run_write_worker(spectra, router, staging_work_dir)
        assert counts == {
            "spectra_charge_1.lance": 1,
            "spectra_charge_2_3.lance": 2,
            "spectra_charge_unknown.lance": 1,
            "spectra_charge_other.lance": 1,
        }

    def test_unmapped_charge_dropped_without_other(
        self, staging_work_dir, spectrum_row
    ):
        counts = _run_write_worker(
            [_row(spectrum_row, "x", 3), _row(spectrum_row, "u", None)],
            staging.ChargeRouter([(1,)]),
            staging_work_dir,
        )
        assert counts == {}

    def test_auto_bucket_isolates_all_charges_including_negative_and_unknown(
        self, staging_work_dir, spectrum_row
    ):
        spectra = [
            _row(spectrum_row, "a", 1),
            _row(spectrum_row, "b", 2),
            _row(spectrum_row, "c", 1),
            _row(spectrum_row, "d", -2),
            _row(spectrum_row, "u", None),
        ]
        counts = _run_write_worker(
            spectra, staging.ChargeRouter(None), staging_work_dir
        )
        assert counts == {
            "spectra_charge_1.lance": 2,
            "spectra_charge_2.lance": 1,
            "spectra_charge_-2.lance": 1,
            "spectra_charge_unknown.lance": 1,
        }

    def test_batch_flush_threshold_writes_all_rows_across_flushes(
        self, staging_work_dir, monkeypatch, spectrum_row
    ):
        monkeypatch.setattr(staging, "_LANCE_WRITE_BATCH_SIZE", 3)
        spectra = [
            *(_row(spectrum_row, f"c2_{i}", 2) for i in range(6)),
            *(_row(spectrum_row, f"c1_{i}", 1) for i in range(4)),
        ]
        counts = _run_write_worker(
            spectra, staging.ChargeRouter(None), staging_work_dir
        )
        assert counts == {
            "spectra_charge_2.lance": 6,
            "spectra_charge_1.lance": 4,
        }

    def test_write_worker_exits_early_when_stop_event_set(
        self, staging_work_dir, spectrum_row
    ):
        stop = threading.Event()
        stop.set()
        counts = _run_write_worker(
            [_row(spectrum_row, "a", 2)],
            staging.ChargeRouter(None),
            staging_work_dir,
            stop_event=stop,
        )
        assert counts == {}

    def test_enqueue_spectrum_aborts_when_stop_event_set_on_full_queue(
        self, spectrum_row
    ):
        q = queue.Queue(maxsize=1)
        q.put(_row(spectrum_row, "first", 2))
        stop = threading.Event()
        timer = threading.Timer(0.05, stop.set)
        timer.start()
        try:
            assert (
                staging._enqueue_spectrum(
                    q, _row(spectrum_row, "second", 2), stop
                )
                is False
            )
        finally:
            timer.cancel()


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
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        accepted, rejected = staging._read_spectra(mgf, proc)
        assert len(accepted) == 1
        assert accepted[0]["identifier"] == "mix:scan:1"
        assert rejected == 1

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
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        valid = staging.stage_spectra([mgf], work_dir, proc, None)
        assert valid == [(-1,), (1,), (3,), ("unknown",)]
        for bucket in valid:
            ds_path = bucket_dataset_path(work_dir, bucket)
            assert ds_path.is_dir()
            assert lance.dataset(str(ds_path)).count_rows() == 1

    def test_stage_spectra_explicit_buckets_preserves_order_and_skips_empty(
        self, staging_work_dir, mgf_block, mgf_writer
    ):
        mgf = mgf_writer(
            staging_work_dir / "sample.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1),
                mgf_block(TOY_PEAKS_5, 500.0, charge=4, scan=2),
                mgf_block(TOY_PEAKS_5, 500.0, charge=1, scan=3),
            ],
        )
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        # Explicit order ((2, 3) before (1,)) is preserved;
        # empty (5,) is skipped.
        valid = staging.stage_spectra(
            [mgf],
            staging_work_dir,
            proc,
            [(2, 3), (5,), (1,), "other"],
        )
        assert valid == [(2, 3), (1,), "other"]

        with pytest.raises(ValueError, match="more than one bucket"):
            staging.stage_spectra(
                [mgf], staging_work_dir, proc, [{1, 2}, {2, 3}]
            )

        with pytest.raises(ValueError, match="other"):
            staging.stage_spectra(
                [mgf], staging_work_dir, proc, [{1}, "other", "other"]
            )

    def test_stage_spectra_empty_input_list_raises_file_not_found(
        self, staging_work_dir
    ):
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            staging.stage_spectra([], staging_work_dir, proc, None)

    def test_stage_spectra_propagates_writer_thread_error(
        self, staging_work_dir, mgf_block, mgf_writer, monkeypatch
    ):
        mgf = mgf_writer(
            staging_work_dir / "err.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
        )
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("lance disk full")

        monkeypatch.setattr(staging, "_write_to_dataset", _boom)
        with pytest.raises(RuntimeError, match="lance disk full"):
            staging.stage_spectra([mgf], staging_work_dir, proc, None)

    def test_stage_spectra_propagates_reader_error(self, staging_work_dir):
        proc = functools.partial(
            preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        missing = staging_work_dir / "missing.mgf"
        with pytest.raises(FileNotFoundError, match="Non-existing peak file"):
            staging.stage_spectra([missing], staging_work_dir, proc, None)
