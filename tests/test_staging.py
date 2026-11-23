"""Unit tests for falcon.staging (globs, Lance staging, cache manifests)."""

import functools
import json
import os
import queue
import threading

import lance
import numpy as np
import pytest

from falcon import staging
from falcon.cluster import spectrum
from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    FalconConfig,
    IOConfig,
    PreprocessingConfig,
)


def _make_spec(identifier, charge):
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    intensity = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    intensity /= np.linalg.norm(intensity)
    return {
        "identifier": identifier,
        "precursor_mz": np.float32(500.0),
        "precursor_charge": charge,
        "mz": mz,
        "intensity": intensity,
        "retention_time": np.float32(0.0),
    }


def _run_write_worker(
    spectra,
    charge_to_bucket,
    catch_other_charges,
    work_dir,
    auto_bucket=False,
):
    q = queue.Queue()
    for spec in spectra:
        q.put(spec)
    q.put(None)

    locks = staging._PerChargeLockRegistry()
    staging._write_spectra_lance(
        q,
        locks,
        charge_to_bucket,
        catch_other_charges,
        auto_bucket,
        work_dir,
    )

    s_dir = staging.spectra_dir(work_dir)
    counts = {}
    if s_dir.exists():
        for p in s_dir.iterdir():
            if p.suffix == ".lance":
                counts[p.name] = lance.dataset(str(p)).count_rows()
    return counts


@pytest.fixture
def staging_work_dir(tmp_path):
    """Create a temporary working directory with a ``spectra`` folder."""
    staging.spectra_dir(tmp_path).mkdir()
    return tmp_path


# ----------------------------------------------------------------------
# Input globs, bucket naming/formatting, and lock registry
# ----------------------------------------------------------------------


class TestResolveInputFiles:
    def test_expands_globs_sorted_deterministically_and_preserves_duplicates(
        self, tmp_path
    ):
        (tmp_path / "b.mgf").write_text("")
        (tmp_path / "a.mgf").write_text("")
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


class TestBucketNamingAndLocks:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ((1,), "_1"),
            ((1, 2), "_1_2"),
            ((-2, -1), "_-2_-1"),
            ((3, "unknown"), "_3_unknown"),
            ("other", "_other"),
            ("unknown", "_unknown"),
            ({2, 1}, "_1_2"),
            ({"unknown", 3}, "_3_unknown"),
        ],
    )
    def test_bucket_key_to_str(self, key, expected):
        assert staging.bucket_key_to_str(key) == expected

    def test_bucket_key_to_str_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported bucket key type"):
            staging.bucket_key_to_str(42)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("bucket", "expected_label"),
        [
            ((2,), "2"),
            ((1, 2), "1, 2"),
            (("unknown",), "unknown"),
            ("other", "other"),
        ],
    )
    def test_format_bucket_label(self, bucket, expected_label):
        assert staging.format_bucket_label(bucket) == expected_label

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
    def test_manifest_and_diff_detect_only_staging_changes(self, tmp_path):
        f1 = tmp_path / "a.mgf"
        f2 = tmp_path / "b.mgf"
        f1.write_text("data1", encoding="utf-8")
        f2.write_text("data2", encoding="utf-8")

        base_cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(str(f1),),
                output_filename=tmp_path / "out1",
            )
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
        sweep_cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(str(f1),),
                output_filename=tmp_path / "out2",
                export_representatives=True,
            ),
            clustering=ClusteringConfig(
                distance_threshold=0.25,
                linkage="average",
                rt_tol=10.0,
                min_matched_peaks=6,
                batch_size=512,
            ),
            consensus=AverageConsensusConfig(),
        )
        assert (
            staging.diff_staging_manifests(
                base_manifest, staging.build_staging_manifest(sweep_cfg, [f1])
            )
            == []
        )

        # Preprocessing, fragment_tol, charge buckets, and inputs invalidate.
        prep_cfg = FalconConfig(
            io=base_cfg.io,
            preprocessing=PreprocessingConfig(min_mz=50.0),
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(prep_cfg, [f1])
        ) == ["preprocessing"]

        frag_cfg = FalconConfig(
            io=base_cfg.io,
            clustering=ClusteringConfig(fragment_tol=0.02),
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(frag_cfg, [f1])
        ) == ["fragment_tol"]

        bucket_cfg = FalconConfig(
            io=base_cfg.io,
            clustering=ClusteringConfig(precursor_charge_buckets=((2,),)),
        )
        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(bucket_cfg, [f1])
        ) == ["precursor_charge_buckets"]

        assert staging.diff_staging_manifests(
            base_manifest, staging.build_staging_manifest(base_cfg, [f1, f2])
        ) == ["inputs"]

        f1.write_text("modified_longer_content", encoding="utf-8")
        st = os.stat(f1)
        os.utime(f1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
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

    def test_deserialize_cached_buckets_rejects_corrupted_entries(self):
        with pytest.raises(ValueError, match="buckets"):
            staging.deserialize_cached_buckets("not-a-list")
        with pytest.raises(ValueError, match="bucket"):
            staging.deserialize_cached_buckets([123])
        with pytest.raises(ValueError, match="bucket"):
            staging.deserialize_cached_buckets([[True]])


# ----------------------------------------------------------------------
# Cache reuse, mismatch, legacy protection, and overwrite
# ----------------------------------------------------------------------


class TestLoadOrStageBuckets:
    def test_reuses_matching_cache_across_clustering_and_consensus_changes(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("dummy")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()

        orig_cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out1",
            io={"work_dir": str(tmp_path)},
            clustering={"precursor_charge_buckets": [[1], [2]]},
        )
        manifest = staging.build_staging_manifest(
            orig_cfg, [input_file], buckets=[(1,), (2,)]
        )
        (s_dir / "spectra_charge_1.lance").mkdir()
        (s_dir / "spectra_charge_2.lance").mkdir()
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        sweep_cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out2",
            io={"work_dir": str(tmp_path), "export_representatives": True},
            clustering={
                "precursor_charge_buckets": [[1], [2]],
                "distance_threshold": 0.25,
                "linkage": "average",
                "rt_tol": 15.0,
            },
            consensus={"method": "average"},
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError("stage_spectra should not be called")

        monkeypatch.setattr(staging, "stage_spectra", _fail)
        assert staging.load_or_stage_buckets(
            sweep_cfg, tmp_path, lambda s: None
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
        self, tmp_path, override_sections, expected_section, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("dummy")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()

        orig_cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = staging.build_staging_manifest(
            orig_cfg, [input_file], buckets=[(2,)]
        )
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        changed_cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
            **override_sections,
        )
        with pytest.raises(ValueError, match=expected_section):
            staging.load_or_stage_buckets(
                changed_cfg, tmp_path, lambda s: None
            )

    def test_modified_input_file_invalidates_staging_cache(
        self, tmp_path, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = staging.build_staging_manifest(
            cfg, [input_file], buckets=[(2,)]
        )
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        input_file.write_text("v2_modified")
        st = os.stat(input_file)
        os.utime(input_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        with pytest.raises(ValueError, match="inputs"):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

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
        self, tmp_path, corrupt_bytes, expected_match, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()
        (s_dir / staging.STAGING_CACHE_FILENAME).write_bytes(corrupt_bytes)
        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        with pytest.raises(ValueError, match=expected_match):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

    def test_invalid_buckets_in_matching_manifest_raises_actionable_error(
        self, tmp_path, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()
        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = staging.build_staging_manifest(cfg, [input_file])
        manifest["buckets"] = "bad"
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Invalid staging cache buckets"):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

    @pytest.mark.parametrize(
        "stale_name", ["charges.joblib", "spectra_charge_2.lance"]
    )
    def test_unverified_legacy_or_orphaned_cache_raises_without_overwrite(
        self, tmp_path, stale_name, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()
        stale_path = s_dir / stale_name
        if stale_name.endswith(".lance"):
            stale_path.mkdir()
        else:
            stale_path.write_bytes(b"legacy")

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        with pytest.raises(ValueError, match="overwrite"):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

    def test_overwrite_clears_cached_directories_and_writes_json_manifest(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        stale_dir = s_dir / "spectra_charge_99.lance"
        stale_dir.mkdir(parents=True)
        (stale_dir / "data").write_text("stale")
        (s_dir / "charges.joblib").write_text("stale")

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path), "overwrite": True},
        )
        monkeypatch.setattr(
            staging,
            "stage_spectra",
            lambda _inputs, _wd, _proc, _buckets: [(2,)],
        )

        buckets = staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)
        assert buckets == [(2,)]
        assert not stale_dir.exists()
        assert not (s_dir / "charges.joblib").exists()
        saved = json.loads(
            (s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert saved["buckets"] == [[2]]
        assert saved["precursor_charge_buckets"] is None

    def test_input_modified_during_staging_triggers_cache_mismatch(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        pre_stat = os.stat(input_file)
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )

        def _stage_while_input_changes(_inputs, _wd, _proc, _buckets):
            input_file.write_text("v2_modified_during_staging")
            st = os.stat(input_file)
            os.utime(
                input_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000)
            )
            return [(2,), ("unknown",), "other"]

        monkeypatch.setattr(
            staging, "stage_spectra", _stage_while_input_changes
        )

        buckets = staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)
        assert buckets == [(2,), ("unknown",), "other"]

        saved = json.loads(
            (s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert saved["inputs"] == [
            {
                "path": str(input_file.resolve()),
                "size": pre_stat.st_size,
                "mtime_ns": pre_stat.st_mtime_ns,
            }
        ]
        assert saved["buckets"] == [[2], ["unknown"], "other"]

        with pytest.raises(
            ValueError,
            match=r"mismatched: inputs.*--overwrite.*--work-dir",
        ):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

    def test_fresh_staging_reuses_manifest_without_restatting_inputs(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
            clustering={"precursor_charge_buckets": [[1, 2], "other"]},
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
                (s_dir / name).mkdir()
            return [(2, 1), ("unknown",), "other"]

        monkeypatch.setattr(staging, "build_staging_manifest", _spy_build)
        monkeypatch.setattr(staging, "stage_spectra", _fake_stage)

        staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)
        assert build_calls == 1

        saved = json.loads(
            (s_dir / staging.STAGING_CACHE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        expected = staging.build_staging_manifest(
            cfg, [input_file], buckets=[(2, 1), ("unknown",), "other"]
        )
        assert saved == expected

        def _fail(*_args, **_kwargs):
            raise AssertionError("stage_spectra should not be called")

        monkeypatch.setattr(staging, "stage_spectra", _fail)
        reused = staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)
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
        self, tmp_path, cached_buckets, disk_entries, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()
        for name, is_dir in disk_entries:
            path = s_dir / name
            if is_dir:
                path.mkdir()
            else:
                path.write_text("not a dataset")

        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = staging.build_staging_manifest(
            cfg, [input_file], buckets=cached_buckets
        )
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(
            ValueError,
            match=r"rerun with --overwrite or a different --work-dir",
        ):
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None)

    def test_empty_cached_buckets_reused_when_no_staged_datasets_exist(
        self, tmp_path, monkeypatch, config_factory
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        s_dir = staging.spectra_dir(tmp_path)
        s_dir.mkdir()
        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = staging.build_staging_manifest(
            cfg, [input_file], buckets=[]
        )
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError("stage_spectra should not be called")

        monkeypatch.setattr(staging, "stage_spectra", _fail)
        assert (
            staging.load_or_stage_buckets(cfg, tmp_path, lambda s: None) == []
        )


# ----------------------------------------------------------------------
# Lance writing, charge routing, and parallel staging
# ----------------------------------------------------------------------


class TestLanceWritingAndRouting:
    def test_creates_and_appends_at_explicit_work_dir(self, tmp_path):
        explicit_dir = tmp_path / "target"
        staging.spectra_dir(explicit_dir).mkdir(parents=True)

        lock = threading.Lock()
        count1 = staging._write_to_dataset(
            [_make_spec("s1", 2), _make_spec("s2", 2)],
            (2,),
            lock,
            explicit_dir,
        )
        count2 = staging._write_to_dataset(
            [_make_spec("s3", 2)],
            (2,),
            lock,
            explicit_dir,
        )
        assert count1 == 2
        assert count2 == 1
        target_ds = staging.bucket_dataset_path(explicit_dir, (2,))
        assert target_ds.exists()
        assert lance.dataset(str(target_ds)).count_rows() == 3

    def test_explicit_single_multi_unknown_and_other_routing(
        self, staging_work_dir
    ):
        charge_to_bucket = {
            1: (1,),
            2: (2, 3),
            3: (2, 3),
            "unknown": ("unknown",),
        }
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", 3),
            _make_spec("d", None),
            _make_spec("e", 5),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=True,
            work_dir=staging_work_dir,
        )
        assert counts == {
            "spectra_charge_1.lance": 1,
            "spectra_charge_2_3.lance": 2,
            "spectra_charge_unknown.lance": 1,
            "spectra_charge_other.lance": 1,
        }

    def test_unmapped_charge_dropped_without_other(self, staging_work_dir):
        counts = _run_write_worker(
            [_make_spec("x", 3), _make_spec("u", None)],
            charge_to_bucket={1: (1,)},
            catch_other_charges=False,
            work_dir=staging_work_dir,
        )
        assert counts == {}

    def test_auto_bucket_isolates_all_charges_including_negative_and_unknown(
        self, staging_work_dir
    ):
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", 1),
            _make_spec("d", -2),
            _make_spec("u", None),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_1.lance": 2,
            "spectra_charge_2.lance": 1,
            "spectra_charge_-2.lance": 1,
            "spectra_charge_unknown.lance": 1,
        }

    def test_batch_flush_threshold_writes_all_rows_across_flushes(
        self, staging_work_dir, monkeypatch
    ):
        monkeypatch.setattr(staging, "_LANCE_WRITE_BATCH_SIZE", 3)
        spectra = [
            *(_make_spec(f"c2_{i}", 2) for i in range(6)),
            *(_make_spec(f"c1_{i}", 1) for i in range(4)),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_2.lance": 6,
            "spectra_charge_1.lance": 4,
        }

    def test_discover_auto_buckets_sorts_numeric_then_unknown(
        self, staging_work_dir
    ):
        _run_write_worker(
            [
                _make_spec("a", 3),
                _make_spec("b", -1),
                _make_spec("c", None),
                _make_spec("d", 1),
            ],
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            auto_bucket=True,
        )
        s_dir = staging.spectra_dir(staging_work_dir)
        (s_dir / staging.STAGING_CACHE_FILENAME).write_text("{}")
        (s_dir / "notes.txt").write_text("ignore")

        assert staging._discover_auto_buckets(s_dir) == [
            (-1,),
            (1,),
            (3,),
            ("unknown",),
        ]


class TestReadAndStageSpectra:
    def test_read_spectra_counts_accepted_and_rejected(
        self, tmp_path, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        good_peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        lowq_peaks = [(150.0, 100.0), (160.0, 200.0)]
        mgf = write_mgf(
            tmp_path / "mix.mgf",
            [
                format_mgf_spectrum(good_peaks, 500.0, charge=2, scan=1),
                format_mgf_spectrum(lowq_peaks, 500.0, charge=2, scan=2),
            ],
        )
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        accepted, rejected = staging._read_spectra(mgf, proc)
        assert len(accepted) == 1
        assert accepted[0]["identifier"] == "mix:scan:1"
        assert rejected == 1

    def test_stage_spectra_explicit_buckets_skips_empty_and_rejects_overlap(
        self, staging_work_dir, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        mgf = write_mgf(
            staging_work_dir / "sample.mgf",
            [
                format_mgf_spectrum(peaks, 500.0, charge=2, scan=1),
                format_mgf_spectrum(peaks, 500.0, charge=4, scan=2),
            ],
        )
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        valid = staging.stage_spectra(
            [mgf], staging_work_dir, proc, [(1,), (2, 3), "other"]
        )
        assert valid == [(2, 3), "other"]

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
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            staging.stage_spectra([], staging_work_dir, proc, None)

    def test_stage_spectra_propagates_writer_thread_error(
        self, staging_work_dir, mgf_writer, monkeypatch
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        mgf = write_mgf(
            staging_work_dir / "err.mgf",
            [format_mgf_spectrum(peaks, 500.0, charge=2, scan=1)],
        )
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        def _boom(*_args, **_kwargs):
            raise RuntimeError("lance disk full")

        monkeypatch.setattr(staging, "_write_to_dataset", _boom)
        with pytest.raises(RuntimeError, match="lance disk full"):
            staging.stage_spectra([mgf], staging_work_dir, proc, None)
