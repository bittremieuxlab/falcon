"""Unit tests for falcon.pipeline orchestration."""

import logging

import numpy as np
import pytest
from conftest import TOY_PEAKS_5

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
        assert any("already exists" in r.message for r in caplog.records)


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
        }

    def test_programmatic_run_does_not_shutdown_logging(
        self,
        tmp_path,
        mgf_block,
        mgf_writer,
        monkeypatch,
        config_factory,
        track_mkdtemp,
    ):
        mgf = mgf_writer(
            tmp_path / "prog.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1, rt=10.0)],
        )
        shutdown_called = False

        def _spy_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        monkeypatch.setattr(logging, "shutdown", _spy_shutdown)
        cfg = config_factory(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "prog_out",
        )
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
        cfg = config_factory(output_filename=tmp_path / "existing")

        assert pipeline.run(cfg) == 1
        assert existing_csv.read_text(encoding="utf-8") == "keep"
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

    def test_run_preserves_explicit_work_dir(
        self, tmp_path, mgf_block, mgf_writer, config_factory
    ):
        mgf = mgf_writer(
            tmp_path / "explicit.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1, rt=10.0)],
        )
        explicit_wd = tmp_path / "work"
        cfg = config_factory(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "explicit_out",
            io={"work_dir": str(explicit_wd)},
        )
        assert pipeline.run(cfg) == 0
        assert explicit_wd.is_dir()
        assert (tmp_path / "explicit_out.csv").is_file()

    def test_run_cleans_up_temp_work_dir_if_logging_raises(
        self, tmp_path, monkeypatch, config_factory, track_mkdtemp
    ):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("log handler boom")

        monkeypatch.setattr(pipeline, "_log_config", _boom)

        cfg = config_factory(output_filename=tmp_path / "out")
        with pytest.raises(RuntimeError, match="log handler boom"):
            pipeline.run(cfg)
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()
