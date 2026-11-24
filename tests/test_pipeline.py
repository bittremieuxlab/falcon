"""Unit tests for falcon.pipeline orchestration."""

import logging

import numpy as np
import pytest

from falcon import pipeline
from falcon.types import ClusterResult


class TestWorkDirLifecycleAndWarnings:
    def test_creates_and_cleans_temp_dir_without_mutating_config(
        self, tmp_path, monkeypatch, config_factory, track_mkdtemp
    ):
        observed = []

        def _spy_stage(_cfg, work_dir, _proc):
            observed.append((work_dir, (work_dir / "spectra").is_dir()))
            return []

        monkeypatch.setattr(pipeline, "load_or_stage_buckets", _spy_stage)
        cfg = config_factory(output_filename=tmp_path / "temp_out")

        assert pipeline.run(cfg) == 0
        assert cfg.io.work_dir is None
        assert len(track_mkdtemp) == 1
        assert observed == [(track_mkdtemp[0], True)]
        assert not track_mkdtemp[0].exists()

    def test_explicit_work_dir_new_and_existing_warns(
        self, tmp_path, monkeypatch, caplog, config_factory
    ):
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        wd = tmp_path / "work"

        # First run creates work_dir/spectra without warning.
        cfg_new = config_factory(
            output_filename=tmp_path / "out1",
            io={"work_dir": str(wd)},
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg_new) == 0
        assert (wd / "spectra").is_dir()
        assert not any("already exists" in r.message for r in caplog.records)

        # Second run with existing work_dir warns about cache reuse.
        caplog.clear()
        cfg_reuse = config_factory(
            output_filename=tmp_path / "out2",
            io={"work_dir": str(wd), "overwrite": False},
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg_reuse) == 0
        assert any(
            "already exists" in r.message and "reused" in r.message
            for r in caplog.records
        )

        # Third run with overwrite=True warns about overwriting cache.
        caplog.clear()
        cfg_overwrite = config_factory(
            output_filename=tmp_path / "out2",
            io={"work_dir": str(wd), "overwrite": True},
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg_overwrite) == 0
        assert any(
            "already exists" in r.message and "overwritten" in r.message
            for r in caplog.records
        )

    def test_explicit_relative_work_dir_is_absolutized(
        self, tmp_path, monkeypatch, config_factory
    ):
        observed = []

        def _spy_stage(_cfg, work_dir, _proc):
            observed.append(work_dir)
            return []

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(pipeline, "load_or_stage_buckets", _spy_stage)
        cfg = config_factory(
            output_filename=tmp_path / "rel_out",
            io={"work_dir": "rel_work"},
        )
        assert pipeline.run(cfg) == 0
        assert observed == [tmp_path / "rel_work"]
        assert observed[0].is_absolute()
        assert (tmp_path / "rel_work" / "spectra").is_dir()

    def test_cleans_up_temp_dir_if_spectra_subdir_creation_fails(
        self, tmp_path, monkeypatch, config_factory, track_mkdtemp
    ):
        class _FailDir:
            def mkdir(self, **_kwargs):
                raise OSError("disk full")

        monkeypatch.setattr(pipeline, "spectra_dir", lambda _p: _FailDir())
        cfg = config_factory(output_filename=tmp_path / "fail_out")
        with pytest.raises(OSError, match="disk full"):
            pipeline.run(cfg)
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

    def test_warns_only_for_average_when_both_outlier_cutoffs_below_one(
        self, tmp_path, monkeypatch, caplog, config_factory
    ):
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg = config_factory(
            output_filename=tmp_path / "aggressive",
            consensus={
                "method": "average",
                "outlier_cutoff_lower": 0.5,
                "outlier_cutoff_upper": 0.8,
            },
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg) == 0
        assert any("less than 1" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "consensus_section",
        [
            {"method": "medoid"},
            {
                "method": "average",
                "outlier_cutoff_lower": 0.5,
                "outlier_cutoff_upper": 1.0,
            },
            {
                "method": "average",
                "outlier_cutoff_lower": 1.0,
                "outlier_cutoff_upper": 0.5,
            },
        ],
    )
    def test_no_outlier_warning_outside_aggressive_average_regime(
        self,
        tmp_path,
        monkeypatch,
        caplog,
        config_factory,
        consensus_section,
    ):
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg = config_factory(
            output_filename=tmp_path / "no_warn",
            consensus=consensus_section,
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg) == 0
        assert not any("less than 1" in r.message for r in caplog.records)


class TestRunClusteringAndPipeline:
    @pytest.mark.parametrize("export_reps", [False, True])
    def test_run_clustering_passes_typed_configs_and_skips_empty_buckets(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        monkeypatch,
        export_reps,
        config_factory,
    ):
        (tmp_path / "spectra").mkdir()
        lance_dataset([], name="spectra/spectra_charge_1.lance")
        lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            name="spectra/spectra_charge_2.lance",
        )
        cfg = config_factory(
            io={
                "work_dir": str(tmp_path),
                "export_representatives": export_reps,
            },
            clustering={"fragment_tol": 0.05},
            consensus={
                "method": "average",
                "outlier_cutoff_lower": 2.0,
                "outlier_cutoff_upper": 2.5,
            },
        )

        captured = {}

        def _fake_multi(datasets, clustering, consensus, **kwargs):
            captured["datasets"] = datasets
            captured["clustering"] = clustering
            captured["consensus"] = consensus
            captured["kwargs"] = kwargs
            return [ClusterResult(np.array([0], dtype=np.int32), [])]

        monkeypatch.setattr(pipeline, "generate_clusters_multi", _fake_multi)

        datasets, results = pipeline._run_clustering(
            cfg, tmp_path, [(1,), (2,)]
        )
        assert len(datasets) == 1
        assert datasets[0].count_rows() == 1
        assert len(results) == 1
        assert captured["clustering"] == cfg.clustering
        assert captured["consensus"] == cfg.consensus
        assert captured["kwargs"] == {
            "min_mz": 101.0,
            "max_mz": 1500.0,
            "bucket_labels": ["2"],
            "compute_representatives": export_reps,
            "temp_dir": tmp_path,
        }

    def test_run_clustering_short_circuits_when_no_non_empty_buckets(
        self, tmp_path, lance_dataset, monkeypatch, config_factory
    ):
        (tmp_path / "spectra").mkdir()
        lance_dataset([], name="spectra/spectra_charge_1.lance")
        cfg = config_factory(io={"work_dir": str(tmp_path)})

        def _forbid_multi(*_args, **_kwargs):
            raise AssertionError("generate_clusters_multi must not run")

        monkeypatch.setattr(pipeline, "generate_clusters_multi", _forbid_multi)
        assert pipeline._run_clustering(cfg, tmp_path, []) == ([], [])
        assert pipeline._run_clustering(cfg, tmp_path, [(1,)]) == ([], [])

    def test_programmatic_run_does_not_shutdown_logging(
        self,
        tmp_path,
        monkeypatch,
        config_factory,
        track_mkdtemp,
    ):
        shutdown_called = False

        def _spy_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        monkeypatch.setattr(logging, "shutdown", _spy_shutdown)
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg = config_factory(output_filename=tmp_path / "prog_out")
        assert pipeline.run(cfg) == 0
        assert shutdown_called is False
        assert (tmp_path / "prog_out.csv").is_file()
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

    def test_run_returns_1_when_output_exists_without_overwrite(
        self, tmp_path, config_factory, track_mkdtemp
    ):
        existing_csv = tmp_path / "existing.csv"
        existing_csv.write_text("keep", encoding="utf-8")
        explicit_wd = tmp_path / "never_created_work"
        cfg = config_factory(output_filename=tmp_path / "existing")

        assert pipeline.run(cfg) == 1
        assert existing_csv.read_text(encoding="utf-8") == "keep"
        assert track_mkdtemp == []

        cfg_explicit = config_factory(
            output_filename=tmp_path / "existing",
            io={"work_dir": str(explicit_wd)},
        )
        assert pipeline.run(cfg_explicit) == 1
        assert not explicit_wd.exists()

    def test_run_preserves_explicit_work_dir(
        self, tmp_path, monkeypatch, config_factory
    ):
        explicit_wd = tmp_path / "work"
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg = config_factory(
            output_filename=tmp_path / "explicit_out",
            io={"work_dir": str(explicit_wd)},
        )
        assert pipeline.run(cfg) == 0
        assert explicit_wd.is_dir()
        assert (tmp_path / "explicit_out.csv").is_file()

    def test_run_fails_before_creating_work_dir_on_missing_output_parent(
        self, tmp_path, config_factory, track_mkdtemp
    ):
        cfg = config_factory(
            output_filename=tmp_path / "nonexistent_dir" / "out"
        )
        with pytest.raises(FileNotFoundError, match="Output directory"):
            pipeline.run(cfg)
        assert track_mkdtemp == []

    def test_run_overwrite_preserves_existing_outputs_on_staging_failure(
        self, tmp_path, config_factory, track_mkdtemp
    ):
        existing_csv = tmp_path / "existing.csv"
        existing_mgf = tmp_path / "existing.mgf"
        existing_csv.write_text("precious csv", encoding="utf-8")
        existing_mgf.write_text("precious mgf", encoding="utf-8")
        cfg = config_factory(
            input_filenames=[str(tmp_path / "typo_*.mgf")],
            output_filename=tmp_path / "existing",
            io={"overwrite": True},
        )
        with pytest.raises(FileNotFoundError):
            pipeline.run(cfg)
        assert existing_csv.read_text(encoding="utf-8") == "precious csv"
        assert existing_mgf.read_text(encoding="utf-8") == "precious mgf"
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

    def test_run_overwrite_removes_stale_mgf_when_export_reps_disabled(
        self, tmp_path, monkeypatch, config_factory
    ):
        existing_csv = tmp_path / "run.csv"
        stale_mgf = tmp_path / "run.mgf"
        existing_csv.write_text("old csv", encoding="utf-8")
        stale_mgf.write_text("stale mgf", encoding="utf-8")
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg = config_factory(
            output_filename=tmp_path / "run",
            io={"overwrite": True, "export_representatives": False},
        )
        assert pipeline.run(cfg) == 0
        assert existing_csv.is_file()
        assert not stale_mgf.exists()
