"""Unit tests for falcon.pipeline orchestration and CSV/MGF export."""

import logging

import numpy as np
import pandas as pd
import pytest

from falcon import pipeline
from falcon.types import ClusterResult


class TestPrepareOutputsAndExport:
    def test_prepare_outputs_validation_and_overwrite_flag(
        self, tmp_path, config_factory
    ):
        inp = tmp_path / "sample.mgf"
        inp.write_text("MGF", encoding="utf-8")
        prefix = tmp_path / "run_v1.2"
        cfg = config_factory(
            input_filenames=[str(inp)], output_filename=prefix
        )
        assert pipeline.prepare_outputs(cfg.io) is True

        # Existing CSV or MGF blocks without overwrite.
        csv_file = tmp_path / "run_v1.2.csv"
        csv_file.write_text("keep", encoding="utf-8")
        assert pipeline.prepare_outputs(cfg.io) is False
        assert csv_file.read_text(encoding="utf-8") == "keep"

        # With overwrite=True, prepare_outputs allows run non-destructively.
        cfg_ow = config_factory(
            input_filenames=[str(inp)],
            output_filename=prefix,
            io={"overwrite": True},
        )
        assert pipeline.prepare_outputs(cfg_ow.io) is True
        assert csv_file.read_text(encoding="utf-8") == "keep"

        # Colliding with input, work_dir, missing parent, or directory raises.
        with pytest.raises(ValueError, match="aliases input file"):
            pipeline.prepare_outputs(
                config_factory(
                    input_filenames=[str(inp)],
                    output_filename=tmp_path / "sample",
                ).io
            )
        with pytest.raises(ValueError, match="aliases working directory"):
            pipeline.prepare_outputs(
                config_factory(
                    output_filename=tmp_path / "run",
                    io={"work_dir": str(tmp_path / "run.csv")},
                ).io
            )
        with pytest.raises(FileNotFoundError, match="Output directory"):
            pipeline.prepare_outputs(
                config_factory(
                    output_filename=tmp_path / "no_such_dir" / "out"
                ).io
            )
        (tmp_path / "dir_out.csv").mkdir()
        with pytest.raises(IsADirectoryError, match="directory"):
            pipeline.prepare_outputs(
                config_factory(output_filename=tmp_path / "dir_out").io
            )

    def test_export_results_offsets_natural_sorts_and_formats_nullable_int8(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        consensus_factory,
        config_factory,
    ):
        ds1 = lance_dataset(
            [
                spectrum_row("run:scan:10", 500.0, retention_time=10.0),
                spectrum_row("run:scan:2", 500.0, retention_time=10.0),
            ],
            charge=2,
        )
        ds2 = lance_dataset(
            [
                spectrum_row(
                    "run:scan:1",
                    600.0,
                    precursor_charge=None,
                    retention_time=np.nan,
                )
            ],
            charge="unknown",
        )
        labels1 = np.array([0, 0], dtype=np.int32)
        labels2 = np.array([0], dtype=np.int32)
        reps1 = [consensus_factory(precursor_mz=500.0, charge=2, cluster_id=0)]
        reps2 = [
            consensus_factory(precursor_mz=600.0, charge=None, cluster_id=0)
        ]

        out_prefix = tmp_path / "run_v1.2"
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": True}
        )
        pipeline.export_results(
            cfg,
            tmp_path,
            [ds1, ds2],
            [ClusterResult(labels1, reps1), ClusterResult(labels2, reps2)],
        )

        # Caller arrays must not be mutated.
        np.testing.assert_array_equal(labels1, [0, 0])
        np.testing.assert_array_equal(labels2, [0])

        raw_lines = [
            line
            for line in (tmp_path / "run_v1.2.csv")
            .read_text(encoding="utf-8")
            .splitlines()
            if not line.startswith("#")
        ]
        rows_by_id = {
            parts[0]: parts
            for parts in (line.split(",") for line in raw_lines[1:])
        }
        # Known charge formats as integer "2" (never "2.0"), unknown as "".
        assert rows_by_id["run:scan:2"][1] == "2"
        assert rows_by_id["run:scan:1"][1] == ""

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["identifier"]) == [
            "run:scan:1",
            "run:scan:2",
            "run:scan:10",
        ]
        mgf_text = (tmp_path / "run_v1.2.mgf").read_text(encoding="utf-8")
        assert "TITLE=falcon:cluster:0" in mgf_text
        assert "TITLE=falcon:cluster:1" in mgf_text

    def test_overwrite_clears_stale_mgf_when_representatives_disabled(
        self, tmp_path, lance_dataset, spectrum_row, config_factory
    ):
        ds = lance_dataset([spectrum_row("s:1", 500.0)], charge=2)
        prefix = tmp_path / "stale"
        (tmp_path / "stale.csv").write_text("old", encoding="utf-8")
        stale_mgf = tmp_path / "stale.mgf"
        stale_mgf.write_text("old mgf", encoding="utf-8")
        cfg = config_factory(
            output_filename=prefix,
            io={"overwrite": True, "export_representatives": False},
        )
        pipeline.export_results(
            cfg, tmp_path, [ds], [ClusterResult(np.array([0], np.int32), [])]
        )
        assert not stale_mgf.exists()
        assert (tmp_path / "stale.csv").is_file()


class TestPipelineRunLifecycle:
    def test_temp_work_dir_cleanup_and_explicit_work_dir_preservation(
        self, tmp_path, monkeypatch, config_factory, track_mkdtemp
    ):
        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        cfg_tmp = config_factory(output_filename=tmp_path / "tmp_out")
        assert pipeline.run(cfg_tmp) == 0
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

        wd = tmp_path / "explicit_work"
        cfg_exp = config_factory(
            output_filename=tmp_path / "exp_out", io={"work_dir": str(wd)}
        )
        assert pipeline.run(cfg_exp) == 0
        assert (wd / "spectra").is_dir()

    def test_run_returns_1_on_existing_output_and_preserves_on_staging_error(
        self, tmp_path, config_factory, track_mkdtemp
    ):
        existing = tmp_path / "existing.csv"
        existing.write_text("precious", encoding="utf-8")
        cfg = config_factory(output_filename=tmp_path / "existing")
        assert pipeline.run(cfg) == 1
        assert track_mkdtemp == []

        # Overwrite=True still preserves existing CSV if staging fails.
        cfg_fail = config_factory(
            input_filenames=[str(tmp_path / "missing_*.mgf")],
            output_filename=tmp_path / "existing",
            io={"overwrite": True},
        )
        with pytest.raises(FileNotFoundError):
            pipeline.run(cfg_fail)
        assert existing.read_text(encoding="utf-8") == "precious"
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

    def test_warns_when_both_average_outlier_cutoffs_below_one(
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
