"""Unit tests for falcon.falcon orchestration helpers."""

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest

from falcon import falcon as falcon_mod
from falcon.cluster.consensus import ConsensusTuple
from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    MedoidConsensusConfig,
    build_staging_manifest,
    load_config,
)


def make_falcon_config(
    input_filenames=("in.mgf",),
    output_filename="out",
    **sections,
):
    """Build a validated ``FalconConfig`` with optional section overrides."""
    io_section = {
        "input_filenames": list(input_filenames),
        "output_filename": str(output_filename),
        **sections.pop("io", {}),
    }
    return load_config(overrides={"io": io_section, **sections})


# ----------------------------------------------------------------------
# Working directory setup and output validation
# ----------------------------------------------------------------------


class TestSetupWorkDir:
    def test_creates_temp_dir_without_mutating_config(self):
        cfg = make_falcon_config()
        work_dir, rm_work_dir = falcon_mod._setup_work_dir(cfg.io.work_dir)
        try:
            assert rm_work_dir is True
            assert cfg.io.work_dir is None
            assert work_dir.is_dir()
            assert (work_dir / "spectra").is_dir()
        finally:
            (work_dir / "spectra").rmdir()
            work_dir.rmdir()

    def test_explicit_work_dir_new_and_existing_warns(self, tmp_path, caplog):
        wd = tmp_path / "work"
        work_dir, rm_work_dir = falcon_mod._setup_work_dir(wd)
        assert work_dir == wd
        assert rm_work_dir is False
        assert (wd / "spectra").is_dir()

        with caplog.at_level(logging.WARNING, logger="falcon"):
            work_dir2, rm2 = falcon_mod._setup_work_dir(wd)
        assert work_dir2 == wd
        assert rm2 is False
        assert any("already exists" in r.message for r in caplog.records)


class TestValidateOutputs:
    def test_allows_run_when_outputs_do_not_exist(self, tmp_path):
        cfg = make_falcon_config(output_filename=tmp_path / "results")
        assert falcon_mod._validate_outputs(cfg.io) is False

    @pytest.mark.parametrize("existing_ext", ["csv", "mgf"])
    def test_aborts_without_overwrite_preserving_existing_file(
        self, tmp_path, existing_ext
    ):
        prefix = tmp_path / "run_v1.2"
        existing = tmp_path / f"run_v1.2.{existing_ext}"
        existing.write_text("keep")
        cfg = make_falcon_config(output_filename=prefix)

        assert falcon_mod._validate_outputs(cfg.io) is True
        assert existing.read_text() == "keep"

    def test_removes_both_csv_and_mgf_when_overwrite_enabled(self, tmp_path):
        prefix = tmp_path / "run_v1.2"
        csv_file = tmp_path / "run_v1.2.csv"
        mgf_file = tmp_path / "run_v1.2.mgf"
        csv_file.write_text("old csv")
        mgf_file.write_text("old mgf")
        cfg = make_falcon_config(
            output_filename=prefix, io={"overwrite": True}
        )

        assert falcon_mod._validate_outputs(cfg.io) is False
        assert not csv_file.exists()
        assert not mgf_file.exists()


class TestWarnAggressiveOutlierCutoffs:
    def test_warns_only_for_average_when_both_cutoffs_below_one(self, caplog):
        cons = AverageConsensusConfig(
            outlier_cutoff_lower=0.5, outlier_cutoff_upper=0.8
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            falcon_mod._warn_aggressive_outlier_cutoffs(cons)
        assert any("less than 1" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "cons",
        [
            MedoidConsensusConfig(),
            AverageConsensusConfig(
                outlier_cutoff_lower=0.5, outlier_cutoff_upper=1.0
            ),
            AverageConsensusConfig(
                outlier_cutoff_lower=1.0, outlier_cutoff_upper=0.5
            ),
        ],
    )
    def test_no_warning_outside_aggressive_average_regime(self, caplog, cons):
        with caplog.at_level(logging.WARNING, logger="falcon"):
            falcon_mod._warn_aggressive_outlier_cutoffs(cons)
        assert caplog.records == []


class TestBuildProcessSpectrum:
    def test_maps_scaling_off_to_none_and_applies_get_dim_bounds(self):
        cfg_off = make_falcon_config(
            preprocessing={"scaling": "off", "min_mz": 101.0}
        )
        proc = falcon_mod._build_process_spectrum(cfg_off)
        assert proc.keywords["scaling"] is None
        assert proc.keywords["mz_min"] <= 101.0

        cfg_root = make_falcon_config(preprocessing={"scaling": "root"})
        proc_root = falcon_mod._build_process_spectrum(cfg_root)
        assert proc_root.keywords["scaling"] == "root"


# ----------------------------------------------------------------------
# Staging cache reuse, mismatch, legacy protection, and overwrite
# ----------------------------------------------------------------------


class TestLoadOrPrepareBuckets:
    def test_reuses_matching_cache_across_clustering_and_consensus_changes(
        self, tmp_path, monkeypatch
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("dummy")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()

        orig_cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out1",
            io={"work_dir": str(tmp_path)},
            clustering={"precursor_charge_buckets": [[1], [2]]},
        )
        manifest = build_staging_manifest(
            orig_cfg, [input_file], buckets=[(1,), (2,)]
        )
        (spectra_dir / "spectra_charge_1.lance").mkdir()
        (spectra_dir / "spectra_charge_2.lance").mkdir()
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        # Sweep clustering/consensus/output settings; cache must be reused.
        sweep_cfg = make_falcon_config(
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
            raise AssertionError("_prepare_spectra should not be called")

        monkeypatch.setattr(falcon_mod, "_prepare_spectra", _fail)
        assert falcon_mod._load_or_prepare_buckets(
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
        self, tmp_path, override_sections, expected_section
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("dummy")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()

        orig_cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = build_staging_manifest(
            orig_cfg, [input_file], buckets=[(2,)]
        )
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        changed_cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
            **override_sections,
        )
        with pytest.raises(ValueError, match=expected_section):
            falcon_mod._load_or_prepare_buckets(
                changed_cfg, tmp_path, lambda s: None
            )

    def test_modified_input_file_invalidates_staging_cache(self, tmp_path):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()

        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = build_staging_manifest(cfg, [input_file], buckets=[(2,)])
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        input_file.write_text("v2_modified")
        st = os.stat(input_file)
        os.utime(input_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        with pytest.raises(ValueError, match="inputs"):
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

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
        self, tmp_path, corrupt_bytes, expected_match
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        (spectra_dir / "staging_cache.json").write_bytes(corrupt_bytes)
        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        with pytest.raises(ValueError, match=expected_match):
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

    def test_invalid_buckets_in_matching_manifest_raises_actionable_error(
        self, tmp_path
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = build_staging_manifest(cfg, [input_file])
        manifest["buckets"] = "bad"
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Invalid staging cache buckets"):
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

    @pytest.mark.parametrize(
        "stale_name", ["charges.joblib", "spectra_charge_2.lance"]
    )
    def test_unverified_legacy_or_orphaned_cache_raises_without_overwrite(
        self, tmp_path, stale_name
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        stale_path = spectra_dir / stale_name
        if stale_name.endswith(".lance"):
            stale_path.mkdir()
        else:
            stale_path.write_bytes(b"legacy")

        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        with pytest.raises(ValueError, match="overwrite"):
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

    def test_overwrite_clears_cached_directories_and_writes_json_manifest(
        self, tmp_path, monkeypatch
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        stale_dir = spectra_dir / "spectra_charge_99.lance"
        stale_dir.mkdir(parents=True)
        (stale_dir / "data").write_text("stale")
        (spectra_dir / "charges.joblib").write_text("stale")

        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path), "overwrite": True},
        )
        monkeypatch.setattr(
            falcon_mod,
            "_prepare_spectra",
            lambda _inputs, _wd, _proc, _buckets: [(2,)],
        )

        buckets = falcon_mod._load_or_prepare_buckets(
            cfg, tmp_path, lambda s: None
        )
        assert buckets == [(2,)]
        assert not stale_dir.exists()
        assert not (spectra_dir / "charges.joblib").exists()
        saved = json.loads(
            (spectra_dir / "staging_cache.json").read_text(encoding="utf-8")
        )
        assert saved["buckets"] == [[2]]
        assert saved["precursor_charge_buckets"] is None

    def test_input_modified_during_staging_triggers_cache_mismatch(
        self, tmp_path, monkeypatch
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        pre_stat = os.stat(input_file)
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()

        cfg = make_falcon_config(
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
            falcon_mod, "_prepare_spectra", _stage_while_input_changes
        )

        buckets = falcon_mod._load_or_prepare_buckets(
            cfg, tmp_path, lambda s: None
        )
        assert buckets == [(2,), ("unknown",), "other"]

        saved = json.loads(
            (spectra_dir / "staging_cache.json").read_text(encoding="utf-8")
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
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

    def test_fresh_staging_reuses_manifest_without_restatting_inputs(
        self, tmp_path, monkeypatch
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()

        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
            clustering={"precursor_charge_buckets": [[1, 2], "other"]},
        )

        build_calls = 0
        orig_build = falcon_mod.build_staging_manifest

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
                (spectra_dir / name).mkdir()
            return [(2, 1), ("unknown",), "other"]

        monkeypatch.setattr(falcon_mod, "build_staging_manifest", _spy_build)
        monkeypatch.setattr(falcon_mod, "_prepare_spectra", _fake_stage)

        falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)
        assert build_calls == 1

        saved = json.loads(
            (spectra_dir / "staging_cache.json").read_text(encoding="utf-8")
        )
        expected = build_staging_manifest(
            cfg, [input_file], buckets=[(2, 1), ("unknown",), "other"]
        )
        assert saved == expected

        def _fail(*_args, **_kwargs):
            raise AssertionError("_prepare_spectra should not be called")

        monkeypatch.setattr(falcon_mod, "_prepare_spectra", _fail)
        reused = falcon_mod._load_or_prepare_buckets(
            cfg, tmp_path, lambda s: None
        )
        assert reused == [(1, 2), ("unknown",), "other"]

    @pytest.mark.parametrize(
        ("cached_buckets", "disk_entries"),
        [
            # Missing staged dataset for one of the cached buckets.
            ([(1,), (2,)], [("spectra_charge_1.lance", True)]),
            # Extra staged dataset not listed in the manifest.
            (
                [(2,)],
                [
                    ("spectra_charge_2.lance", True),
                    ("spectra_charge_3.lance", True),
                ],
            ),
            # Empty manifest cannot reuse when staged datasets exist.
            ([], [("spectra_charge_2.lance", True)]),
            # Regular file instead of a Lance dataset directory.
            ([(2,)], [("spectra_charge_2.lance", False)]),
        ],
    )
    def test_cached_buckets_mismatching_staged_datasets_raises(
        self, tmp_path, cached_buckets, disk_entries
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        for name, is_dir in disk_entries:
            path = spectra_dir / name
            if is_dir:
                path.mkdir()
            else:
                path.write_text("not a dataset")

        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = build_staging_manifest(
            cfg, [input_file], buckets=cached_buckets
        )
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with pytest.raises(
            ValueError,
            match=r"rerun with --overwrite or a different --work-dir",
        ):
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)

    def test_empty_cached_buckets_reused_when_no_staged_datasets_exist(
        self, tmp_path, monkeypatch
    ):
        input_file = tmp_path / "in.mgf"
        input_file.write_text("v1")
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        cfg = make_falcon_config(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path)},
        )
        manifest = build_staging_manifest(cfg, [input_file], buckets=[])
        (spectra_dir / "staging_cache.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError("_prepare_spectra should not be called")

        monkeypatch.setattr(falcon_mod, "_prepare_spectra", _fail)
        assert (
            falcon_mod._load_or_prepare_buckets(cfg, tmp_path, lambda s: None)
            == []
        )


# ----------------------------------------------------------------------
# _run_clustering, _export_results, and programmatic run()
# ----------------------------------------------------------------------


class TestRunClusteringAndExportResults:
    def test_run_clustering_passes_average_params_and_skips_empty_buckets(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        monkeypatch,
    ):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        lance_dataset([], name="spectra/spectra_charge_1.lance")
        lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            name="spectra/spectra_charge_2.lance",
        )
        cfg = make_falcon_config(
            io={"work_dir": str(tmp_path)},
            clustering={"fragment_tol": 0.05},
            consensus={
                "method": "average",
                "outlier_cutoff_lower": 2.0,
                "outlier_cutoff_upper": 2.5,
            },
        )

        captured = {}

        def _fake_multi(datasets, *args):
            captured["datasets"] = datasets
            captured["consensus_params"] = args[-1]
            return [(np.array([0], dtype=np.int32), [])]

        monkeypatch.setattr(
            falcon_mod.cluster, "generate_clusters_multi", _fake_multi
        )

        datasets, results = falcon_mod._run_clustering(
            cfg, tmp_path, [(1,), (2,)]
        )
        assert len(datasets) == 1
        assert datasets[0].count_rows() == 1
        assert len(results) == 1
        assert captured["consensus_params"] == {
            "min_mz": 101.0,
            "max_mz": 1500.0,
            "bin_size": pytest.approx(0.1),
            "outlier_cutoff_lower": 2.0,
            "outlier_cutoff_upper": 2.5,
        }

    def test_export_results_offsets_buckets_and_natural_sorts_identifiers(
        self, tmp_path, lance_dataset, spectrum_row
    ):
        ds1 = lance_dataset(
            [
                spectrum_row(
                    "run:scan:10", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
                spectrum_row(
                    "run:scan:2", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
            ],
            charge=2,
        )
        ds2 = lance_dataset(
            [
                spectrum_row(
                    "run:scan:1", 600.0, [150.0, 250.0], [1.0, 1.0], 3, 20.0
                )
            ],
            charge=3,
        )
        rep1 = ConsensusTuple(
            np.float32(500.0),
            np.int32(2),
            np.array([100.0, 200.0], np.float32),
            np.array([0.7, 0.7], np.float32),
            np.float32(10.0),
            np.int32(0),
            np.int32(0),
        )
        rep2 = ConsensusTuple(
            np.float32(600.0),
            np.int32(3),
            np.array([150.0, 250.0], np.float32),
            np.array([0.7, 0.7], np.float32),
            np.float32(20.0),
            np.int32(0),
            np.int32(0),
        )
        out_prefix = tmp_path / "run_v1.2"
        cfg = make_falcon_config(
            output_filename=out_prefix, io={"export_representatives": True}
        )

        falcon_mod._export_results(
            cfg,
            tmp_path,
            [ds1, ds2],
            [
                (np.array([0, 0], dtype=np.int32), [rep1]),
                (np.array([0], dtype=np.int32), [rep2]),
            ],
        )

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["identifier"]) == [
            "run:scan:1",
            "run:scan:2",
            "run:scan:10",
        ]
        by_id = df.set_index("identifier")["cluster"].to_dict()
        assert by_id["run:scan:2"] == by_id["run:scan:10"] == 0
        assert by_id["run:scan:1"] == 1
        mgf_text = (tmp_path / "run_v1.2.mgf").read_text()
        assert "TITLE=falcon:cluster:0" in mgf_text
        assert "TITLE=falcon:cluster:1" in mgf_text

    def test_export_results_empty_inputs_writes_empty_csv_and_mgf(
        self, tmp_path
    ):
        out_prefix = tmp_path / "empty_out"
        cfg = make_falcon_config(
            output_filename=out_prefix, io={"export_representatives": True}
        )
        falcon_mod._export_results(cfg, tmp_path, [], [])
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert len(df) == 0
        assert list(df.columns) == [
            "identifier",
            "precursor_charge",
            "precursor_mz",
            "retention_time",
            "cluster",
        ]
        assert (tmp_path / "empty_out.mgf").is_file()

    def test_export_results_propagates_writer_thread_exceptions(
        self, tmp_path, lance_dataset, spectrum_row
    ):
        ds = lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            charge=2,
        )
        bad_prefix = tmp_path / "nonexistent_dir" / "out"
        cfg = make_falcon_config(output_filename=bad_prefix)

        with pytest.raises(FileNotFoundError):
            falcon_mod._export_results(
                cfg, tmp_path, [ds], [(np.array([0], dtype=np.int32), [])]
            )

    def test_programmatic_run_does_not_shutdown_logging(
        self, tmp_path, mgf_writer, monkeypatch
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
            tmp_path / "prog.mgf",
            [format_mgf_spectrum(peaks, 500.0, charge=2, scan=1, rt=10.0)],
        )
        shutdown_called = False

        def _spy_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        monkeypatch.setattr(logging, "shutdown", _spy_shutdown)
        cfg = make_falcon_config(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "prog_out",
            clustering=ClusteringConfig(distance_threshold=0.1).model_dump(),
        )
        assert falcon_mod.run(cfg) == 0
        assert shutdown_called is False
        assert (tmp_path / "prog_out.csv").is_file()

    def test_run_cleans_up_temp_work_dir_if_logging_raises(
        self, tmp_path, monkeypatch
    ):
        created: list[str] = []
        orig_mkdtemp = falcon_mod.tempfile.mkdtemp

        def _track(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created.append(d)
            return d

        def _boom(*_args, **_kwargs):
            raise RuntimeError("log handler boom")

        monkeypatch.setattr(falcon_mod.tempfile, "mkdtemp", _track)
        monkeypatch.setattr(falcon_mod, "_log_config", _boom)

        cfg = make_falcon_config(output_filename=tmp_path / "out")
        with pytest.raises(RuntimeError, match="log handler boom"):
            falcon_mod.run(cfg)
        assert len(created) == 1
        assert not os.path.exists(created[0])
