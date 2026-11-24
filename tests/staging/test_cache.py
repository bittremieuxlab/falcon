"""Unit tests for falcon.staging.cache (manifests and cache reuse)."""

import json
import os
from types import SimpleNamespace

import pytest

from falcon import staging
from falcon.dataset import bucket_dataset_name, spectra_dir
from falcon.staging import cache as staging_cache


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
    """Fail if ``stage_spectra`` is invoked by cache loading."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("stage_spectra should not be called")

    monkeypatch.setattr(staging_cache, "stage_spectra", _fail)


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

    def test_diff_rejects_bool_schema_version_and_missing_keys(
        self, tmp_path, config_factory
    ):
        f1 = tmp_path / "a.mgf"
        f1.write_text("data", encoding="utf-8")
        cfg = config_factory(
            input_filenames=[str(f1)], output_filename=tmp_path / "out"
        )
        expected = staging.build_staging_manifest(cfg, [f1])
        assert expected["precursor_charge_buckets"] is None

        bool_ver = {**expected, "schema_version": True}
        assert staging.diff_staging_manifests(bool_ver, expected) == [
            "schema_version"
        ]

        missing_key = dict(expected)
        del missing_key["precursor_charge_buckets"]
        assert staging.diff_staging_manifests(missing_key, expected) == [
            "precursor_charge_buckets"
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

    @pytest.mark.parametrize(
        ("payload", "exc_type"),
        [
            ({"bad": object()}, TypeError),
            ({"bad": float("nan")}, ValueError),
        ],
    )
    def test_write_json_atomic_cleans_up_temp_file_on_error(
        self, tmp_path, payload, exc_type
    ):
        target = tmp_path / staging.STAGING_CACHE_FILENAME
        with pytest.raises(exc_type):
            staging_cache._write_json_atomic(target, payload)
        assert not target.exists()
        assert list(tmp_path.glob(".staging_cache_*.tmp")) == []

    def test_write_json_atomic_closes_fd_and_unlinks_if_fdopen_fails(
        self, tmp_path, monkeypatch
    ):
        closed_fds = []
        real_close = os.close

        def _spy_close(fd):
            closed_fds.append(fd)
            real_close(fd)

        def _boom_fdopen(*_args, **_kwargs):
            raise OSError("fdopen failed")

        monkeypatch.setattr(os, "close", _spy_close)
        monkeypatch.setattr(os, "fdopen", _boom_fdopen)

        target = tmp_path / staging.STAGING_CACHE_FILENAME
        with pytest.raises(OSError, match="fdopen failed"):
            staging_cache._write_json_atomic(target, {"ok": 1})
        assert len(closed_fds) == 1
        assert not target.exists()
        assert list(tmp_path.glob(".staging_cache_*.tmp")) == []

    def test_clear_directory_unlinks_symlink_to_dir_preserving_target(
        self, tmp_path
    ):
        external = tmp_path / "external_dir"
        external.mkdir()
        keep_file = external / "keep.txt"
        keep_file.write_text("important", encoding="utf-8")

        target = tmp_path / "clear_me"
        target.mkdir()
        (target / "sub").mkdir()
        (target / "file.txt").write_text("bye", encoding="utf-8")
        link = target / "linked_dir"
        link.symlink_to(external)

        staging_cache._clear_directory(target)
        assert list(target.iterdir()) == []
        assert not link.is_symlink()
        assert keep_file.read_text(encoding="utf-8") == "important"


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
            staging_cache,
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

    @pytest.mark.parametrize("kind", ["dir", "broken_symlink"])
    def test_directory_or_broken_symlink_at_cache_path_fails_fast(
        self, cache_env, monkeypatch, kind
    ):
        cache_path = cache_env.s_dir / staging.STAGING_CACHE_FILENAME
        if kind == "dir":
            cache_path.mkdir()
        else:
            cache_path.symlink_to(cache_env.work_dir / "missing_target")

        _forbid_stage_spectra(monkeypatch)
        cfg = cache_env.make_cfg()
        with pytest.raises(ValueError, match="Corrupted staging cache"):
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
            staging_cache,
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
            staging_cache, "stage_spectra", _stage_while_input_changes
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

    def test_fresh_staging_persists_canonical_manifest_reusable_on_next_run(
        self, cache_env, monkeypatch
    ):
        cfg = cache_env.make_cfg(
            clustering={"precursor_charge_buckets": [[1, 2], "other"]}
        )

        def _fake_stage(_inputs, _wd, _proc, _buckets):
            for name in (
                "spectra_charge_1_2.lance",
                "spectra_charge_unknown.lance",
                "spectra_charge_other.lance",
            ):
                (cache_env.s_dir / name).mkdir()
            return [(2, 1), ("unknown",), "other"]

        monkeypatch.setattr(staging_cache, "stage_spectra", _fake_stage)

        staging.load_or_stage_buckets(cfg, cache_env.work_dir, lambda _s: None)

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
