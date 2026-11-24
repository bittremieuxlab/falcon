"""Unit tests for falcon.pipeline orchestration and CSV/MGF export."""

import logging

import numpy as np
import pandas as pd
import pytest

from falcon import pipeline
from falcon.cluster import ClusterResult
from tests.helpers import (
    format_mgf_spectrum,
    make_consensus_record,
    make_falcon_config,
    make_spectrum_row,
    write_mgf,
)


class TestPrepareOutputsAndExport:
    def test_prepare_outputs_validation_and_overwrite(self, tmp_path):
        prefix = tmp_path / "run_v1.2"
        cfg = make_falcon_config(output_filename=prefix)
        assert pipeline.prepare_outputs(cfg.io) is True

        csv_file = tmp_path / "run_v1.2.csv"
        mgf_file = tmp_path / "run_v1.2.mgf"
        csv_file.write_text("keep csv", encoding="utf-8")
        mgf_file.write_text("keep mgf", encoding="utf-8")
        assert pipeline.prepare_outputs(cfg.io) is False
        assert csv_file.read_text(encoding="utf-8") == "keep csv"
        assert mgf_file.read_text(encoding="utf-8") == "keep mgf"

        cfg_ow = make_falcon_config(
            output_filename=prefix,
            io={"overwrite": True, "export_representatives": False},
        )
        assert pipeline.prepare_outputs(cfg_ow.io) is True
        assert not csv_file.exists()
        assert not mgf_file.exists()

        broken_csv = tmp_path / "run_v1.2.csv"
        broken_mgf = tmp_path / "run_v1.2.mgf"
        broken_csv.symlink_to(tmp_path / "missing_csv")
        broken_mgf.symlink_to(tmp_path / "missing_mgf")
        assert pipeline.prepare_outputs(cfg.io) is False
        assert broken_csv.is_symlink()
        assert broken_mgf.is_symlink()
        assert pipeline.prepare_outputs(cfg_ow.io) is True
        assert not broken_csv.is_symlink()
        assert not broken_mgf.is_symlink()

        with pytest.raises(FileNotFoundError, match="Output directory"):
            pipeline.prepare_outputs(
                make_falcon_config(
                    output_filename=tmp_path / "no_such_dir" / "out"
                ).io
            )

        sibling_csv = tmp_path / "dir_out.csv"
        sibling_csv.write_text("intact", encoding="utf-8")
        (tmp_path / "dir_out.mgf").mkdir()
        with pytest.raises(IsADirectoryError, match="directory"):
            pipeline.prepare_outputs(
                make_falcon_config(
                    output_filename=tmp_path / "dir_out",
                    io={"overwrite": True},
                ).io
            )
        assert sibling_csv.read_text(encoding="utf-8") == "intact"

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


class TestPipelineRunLifecycle:
    def test_overwrite_removes_old_mgf_before_input_glob_resolution(
        self, tmp_path
    ):
        peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_mgf(
            data_dir / "raw.mgf",
            [format_mgf_spectrum(peaks, 500.0, charge=2, scan=1, rt=10.0)],
        )
        stale_csv = data_dir / "result.csv"
        stale_csv.write_text("old csv", encoding="utf-8")
        stale_mgf = write_mgf(
            data_dir / "result.mgf",
            [format_mgf_spectrum(peaks, 600.0, charge=2, scan=1, rt=20.0)],
        )

        cfg = make_falcon_config(
            input_filenames=[str(data_dir / "*.mgf")],
            output_filename=data_dir / "result",
            io={"overwrite": True, "export_representatives": False},
        )
        assert pipeline.run(cfg) == 0
        assert not stale_mgf.exists()
        df = pd.read_csv(stale_csv, comment="#")
        assert list(df["identifier"]) == ["raw:scan:1"]

        # When only the stale output MGF matched the glob, removing it
        # upfront makes glob resolution fail instead of clustering it.
        stale_only = tmp_path / "stale_only"
        stale_only.mkdir()
        lonely_mgf = write_mgf(
            stale_only / "result.mgf",
            [format_mgf_spectrum(peaks, 600.0, charge=2, scan=1, rt=20.0)],
        )
        cfg_empty = make_falcon_config(
            input_filenames=[str(stale_only / "*.mgf")],
            output_filename=stale_only / "result",
            io={"overwrite": True},
        )
        with pytest.raises(FileNotFoundError, match="matched no files"):
            pipeline.run(cfg_empty)
        assert not lonely_mgf.exists()

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
        assert existing.read_text(encoding="utf-8") == "precious"
        assert track_mkdtemp == []

        # Overwrite=True unlinks existing output before glob resolution
        # and still cleans up temporary work_dir when staging fails.
        cfg_fail = make_falcon_config(
            input_filenames=[str(tmp_path / "missing_*.mgf")],
            output_filename=tmp_path / "existing",
            io={"overwrite": True},
        )
        with pytest.raises(FileNotFoundError):
            pipeline.run(cfg_fail)
        assert not existing.exists()
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
