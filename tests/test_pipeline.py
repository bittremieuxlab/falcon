"""Unit tests for falcon.pipeline orchestration and CSV/MGF export."""

import logging

import numpy as np
import pandas as pd
import pytest

from falcon import pipeline
from falcon.cluster import ClusterResult
from tests.helpers import (
    make_consensus_record,
    make_falcon_config,
    make_spectrum_row,
)


class TestPrepareOutputsAndExport:
    def test_prepare_outputs_validation_and_overwrite(self, tmp_path):
        prefix = tmp_path / "run_v1.2"
        cfg = make_falcon_config(output_filename=prefix)
        assert pipeline.prepare_outputs(cfg.io) is True

        csv_file = tmp_path / "run_v1.2.csv"
        csv_file.write_text("keep", encoding="utf-8")
        assert pipeline.prepare_outputs(cfg.io) is False
        assert csv_file.read_text(encoding="utf-8") == "keep"

        cfg_ow = make_falcon_config(
            output_filename=prefix, io={"overwrite": True}
        )
        assert pipeline.prepare_outputs(cfg_ow.io) is True
        assert csv_file.read_text(encoding="utf-8") == "keep"

        with pytest.raises(FileNotFoundError, match="Output directory"):
            pipeline.prepare_outputs(
                make_falcon_config(
                    output_filename=tmp_path / "no_such_dir" / "out"
                ).io
            )
        (tmp_path / "dir_out.csv").mkdir()
        with pytest.raises(IsADirectoryError, match="directory"):
            pipeline.prepare_outputs(
                make_falcon_config(output_filename=tmp_path / "dir_out").io
            )

    def test_export_results_offsets_natural_sorts_and_formats_nullable_int8(
        self, tmp_path, lance_dataset
    ):
        ds1 = lance_dataset(
            [
                make_spectrum_row("run:scan:10", 500.0, retention_time=10.0),
                make_spectrum_row("run:scan:2", 500.0, retention_time=10.0),
            ],
            charge=2,
        )
        ds2 = lance_dataset(
            [
                make_spectrum_row(
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
        reps1 = [
            make_consensus_record(precursor_mz=500.0, charge=2, cluster_id=0)
        ]
        reps2 = [
            make_consensus_record(
                precursor_mz=600.0, charge=None, cluster_id=0
            )
        ]

        out_prefix = tmp_path / "run_v1.2"
        cfg = make_falcon_config(
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
        self, tmp_path, lance_dataset
    ):
        ds = lance_dataset([make_spectrum_row("s:1", 500.0)], charge=2)
        prefix = tmp_path / "stale"
        (tmp_path / "stale.csv").write_text("old", encoding="utf-8")
        stale_mgf = tmp_path / "stale.mgf"
        stale_mgf.write_text("old mgf", encoding="utf-8")
        cfg = make_falcon_config(
            output_filename=prefix,
            io={"overwrite": True, "export_representatives": False},
        )
        pipeline.export_results(
            cfg, tmp_path, [ds], [ClusterResult(np.array([0], np.int32), [])]
        )
        assert not stale_mgf.exists()
        assert (tmp_path / "stale.csv").is_file()


class TestPipelineRunLifecycle:
    def test_work_dir_cleanup_existing_abort_and_aggressive_cutoff_warning(
        self, tmp_path, monkeypatch, caplog, track_mkdtemp
    ):
        existing = tmp_path / "existing.csv"
        existing.write_text("precious", encoding="utf-8")
        assert (
            pipeline.run(
                make_falcon_config(output_filename=tmp_path / "existing")
            )
            == 1
        )
        assert track_mkdtemp == []

        # Overwrite=True preserves existing CSV if staging fails.
        cfg_fail = make_falcon_config(
            input_filenames=[str(tmp_path / "missing_*.mgf")],
            output_filename=tmp_path / "existing",
            io={"overwrite": True},
        )
        with pytest.raises(FileNotFoundError):
            pipeline.run(cfg_fail)
        assert existing.read_text(encoding="utf-8") == "precious"
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

        monkeypatch.setattr(
            pipeline, "load_or_stage_buckets", lambda *_a, **_kw: []
        )
        wd = tmp_path / "explicit_work"
        cfg_warn = make_falcon_config(
            output_filename=tmp_path / "aggressive",
            io={"work_dir": str(wd)},
            consensus={
                "method": "average",
                "outlier_cutoff_lower": 0.5,
                "outlier_cutoff_upper": 0.8,
            },
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert pipeline.run(cfg_warn) == 0
        assert wd.is_dir()
        assert any("less than 1" in r.message for r in caplog.records)
