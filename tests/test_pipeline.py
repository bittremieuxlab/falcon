"""Unit tests for falcon.pipeline orchestration."""

import logging
from pathlib import Path

import numpy as np
import pytest

from falcon import pipeline
from falcon.config import (
    AverageConsensusConfig,
    MedoidConsensusConfig,
)
from falcon.types import ClusterResult


class TestSetupWorkDir:
    def test_creates_temp_dir_without_mutating_config(self, config_factory):
        cfg = config_factory()
        work_dir, rm_work_dir = pipeline._setup_work_dir(cfg.io.work_dir)
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
        work_dir, rm_work_dir = pipeline._setup_work_dir(wd)
        assert work_dir == wd
        assert rm_work_dir is False
        assert (wd / "spectra").is_dir()

        with caplog.at_level(logging.WARNING, logger="falcon"):
            work_dir2, rm2 = pipeline._setup_work_dir(wd)
        assert work_dir2 == wd
        assert rm2 is False
        assert any(
            "already exists" in r.message and "reused" in r.message
            for r in caplog.records
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="falcon"):
            work_dir3, rm3 = pipeline._setup_work_dir(wd, overwrite=True)
        assert work_dir3 == wd
        assert rm3 is False
        assert any(
            "already exists" in r.message and "overwritten" in r.message
            for r in caplog.records
        )

    def test_explicit_relative_work_dir_is_absolutized(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        work_dir, rm_work_dir = pipeline._setup_work_dir(Path("rel_work"))
        assert work_dir == tmp_path / "rel_work"
        assert work_dir.is_absolute()
        assert rm_work_dir is False
        assert (tmp_path / "rel_work" / "spectra").is_dir()

    def test_cleans_up_temp_dir_if_spectra_subdir_creation_fails(
        self, monkeypatch, track_mkdtemp
    ):
        class _FailDir:
            def mkdir(self, **_kwargs):
                raise OSError("disk full")

        monkeypatch.setattr(pipeline, "spectra_dir", lambda _p: _FailDir())
        with pytest.raises(OSError, match="disk full"):
            pipeline._setup_work_dir(None)
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()


class TestWarnAggressiveOutlierCutoffs:
    def test_warns_only_for_average_when_both_cutoffs_below_one(self, caplog):
        cons = AverageConsensusConfig(
            outlier_cutoff_lower=0.5, outlier_cutoff_upper=0.8
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            pipeline._warn_aggressive_outlier_cutoffs(cons)
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
            pipeline._warn_aggressive_outlier_cutoffs(cons)
        assert caplog.records == []


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
