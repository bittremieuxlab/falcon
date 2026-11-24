"""Unit tests for falcon.export (output prep and CSV/MGF export)."""

import concurrent.futures

import numpy as np
import pandas as pd
import pytest

from falcon import export
from falcon.types import ClusterResult


class TestPrepareOutputs:
    def test_allows_run_when_outputs_do_not_exist(
        self, tmp_path, config_factory
    ):
        cfg = config_factory(output_filename=tmp_path / "results")
        assert export.prepare_outputs(cfg.io) is True

    @pytest.mark.parametrize("existing_ext", ["csv", "mgf"])
    def test_blocks_without_overwrite_preserving_existing_file(
        self, tmp_path, existing_ext, config_factory
    ):
        prefix = tmp_path / "run_v1.2"
        existing = tmp_path / f"run_v1.2.{existing_ext}"
        existing.write_text("keep", encoding="utf-8")
        cfg = config_factory(output_filename=prefix)

        assert export.prepare_outputs(cfg.io) is False
        assert existing.read_text(encoding="utf-8") == "keep"

    def test_removes_both_csv_and_mgf_when_overwrite_enabled(
        self, tmp_path, config_factory
    ):
        prefix = tmp_path / "run_v1.2"
        csv_file = tmp_path / "run_v1.2.csv"
        mgf_file = tmp_path / "run_v1.2.mgf"
        csv_file.write_text("old csv", encoding="utf-8")
        mgf_file.write_text("old mgf", encoding="utf-8")
        cfg = config_factory(output_filename=prefix, io={"overwrite": True})

        assert export.prepare_outputs(cfg.io) is True
        assert not csv_file.exists()
        assert not mgf_file.exists()

    @pytest.mark.parametrize("alias_ext", ["mgf", "csv"])
    def test_rejects_output_aliasing_input_before_unlinking(
        self, tmp_path, alias_ext, config_factory
    ):
        input_file = tmp_path / f"sample.{alias_ext}"
        input_file.write_text("precious input", encoding="utf-8")
        cfg = config_factory(
            input_filenames=[str(input_file)],
            output_filename=tmp_path / "sample",
            io={"overwrite": True},
        )
        with pytest.raises(ValueError, match="aliases input file"):
            export.prepare_outputs(cfg.io)
        assert input_file.read_text(encoding="utf-8") == "precious input"

    def test_rejects_missing_output_parent_directory(
        self, tmp_path, config_factory
    ):
        cfg = config_factory(
            output_filename=tmp_path / "missing_dir" / "results"
        )
        with pytest.raises(FileNotFoundError, match="Output directory"):
            export.prepare_outputs(cfg.io)

    @pytest.mark.parametrize("dir_ext", ["csv", "mgf"])
    @pytest.mark.parametrize("overwrite", [False, True])
    def test_rejects_directory_at_output_path(
        self, tmp_path, dir_ext, overwrite, config_factory
    ):
        (tmp_path / f"run.{dir_ext}").mkdir()
        cfg = config_factory(
            output_filename=tmp_path / "run",
            io={"overwrite": overwrite},
        )
        with pytest.raises(IsADirectoryError, match="is a directory"):
            export.prepare_outputs(cfg.io)

    def test_handles_broken_symlink_at_output_path(
        self, tmp_path, config_factory
    ):
        broken = tmp_path / "run.csv"
        broken.symlink_to(tmp_path / "nonexistent_target")
        cfg_no_ow = config_factory(output_filename=tmp_path / "run")
        assert export.prepare_outputs(cfg_no_ow.io) is False
        assert broken.is_symlink()

        cfg_ow = config_factory(
            output_filename=tmp_path / "run", io={"overwrite": True}
        )
        assert export.prepare_outputs(cfg_ow.io) is True
        assert not broken.is_symlink()

    def test_handles_symlink_loop_at_output_path_without_crashing(
        self, tmp_path, config_factory
    ):
        loop_a = tmp_path / "loop.csv"
        loop_b = tmp_path / "loop_target.csv"
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)

        cfg = config_factory(output_filename=tmp_path / "loop")
        assert export.prepare_outputs(cfg.io) is False

    def test_directory_at_mgf_preserves_existing_csv_under_overwrite(
        self, tmp_path, config_factory
    ):
        csv_file = tmp_path / "run.csv"
        csv_file.write_text("keep csv", encoding="utf-8")
        (tmp_path / "run.mgf").mkdir()
        cfg = config_factory(
            output_filename=tmp_path / "run", io={"overwrite": True}
        )
        with pytest.raises(IsADirectoryError, match="is a directory"):
            export.prepare_outputs(cfg.io)
        assert csv_file.read_text(encoding="utf-8") == "keep csv"

    def test_unlink_false_preserves_existing_files_under_overwrite(
        self, tmp_path, config_factory
    ):
        csv_file = tmp_path / "run.csv"
        mgf_file = tmp_path / "run.mgf"
        csv_file.write_text("old csv", encoding="utf-8")
        mgf_file.write_text("old mgf", encoding="utf-8")
        cfg = config_factory(
            output_filename=tmp_path / "run", io={"overwrite": True}
        )
        assert export.prepare_outputs(cfg.io, unlink=False) is True
        assert csv_file.read_text(encoding="utf-8") == "old csv"
        assert mgf_file.read_text(encoding="utf-8") == "old mgf"

    def test_nul_byte_output_path_does_not_crash_aliasing_check(
        self, config_factory
    ):
        cfg = config_factory(output_filename="bad\x00out")
        export._check_input_output_aliasing(cfg.io)


class TestWriteClusterCsv:
    def test_writes_metadata_comments_and_overwrites_cleanly(
        self, tmp_path, config_factory
    ):
        cfg = config_factory(output_filename=tmp_path / "run_v1.2")
        df = pd.DataFrame(
            {
                "identifier": ["f:scan:1", "f:scan:2"],
                "precursor_charge": [2, 2],
                "precursor_mz": [500.0, 500.05],
                "retention_time": [10.0, 11.0],
                "cluster": [0, 0],
            }
        )
        # Write twice to prove "w" mode does not append rows.
        export.write_cluster_csv(df, cfg, tmp_path)
        export.write_cluster_csv(df, cfg, tmp_path)

        csv_path = cfg.io.csv_path
        assert csv_path == tmp_path / "run_v1.2.csv"
        raw_lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert raw_lines[0].startswith("# falcon version ")
        assert any(
            line == "# clustering.linkage = complete" for line in raw_lines
        )
        assert any(line == f"# io.work_dir = {tmp_path}" for line in raw_lines)
        parsed = pd.read_csv(csv_path, comment="#")
        assert list(parsed.columns) == [
            "identifier",
            "precursor_charge",
            "precursor_mz",
            "retention_time",
            "cluster",
        ]
        assert list(parsed["identifier"]) == ["f:scan:1", "f:scan:2"]
        assert list(parsed["cluster"]) == [0, 0]

    def test_escapes_embedded_newlines_in_metadata_comments(
        self, tmp_path, config_factory, monkeypatch
    ):
        cfg = config_factory(output_filename=tmp_path / "safe")
        df = pd.DataFrame(
            {
                "identifier": ["f:scan:1"],
                "precursor_charge": [2],
                "precursor_mz": [500.0],
                "retention_time": [10.0],
                "cluster": [0],
            }
        )
        monkeypatch.setattr(
            export,
            "format_metadata_lines",
            lambda *_a, **_kw: [
                "falcon version 0.1",
                "io.work_dir = /tmp/line1\ninjected,row\rline2",
            ],
        )
        export.write_cluster_csv(df, cfg, tmp_path)

        raw_lines = cfg.io.csv_path.read_text(encoding="utf-8").splitlines()
        comment_lines = [line for line in raw_lines if line.startswith("#")]
        assert len(comment_lines) == 3
        assert "\\n" in comment_lines[1]
        assert "\\r" in comment_lines[1]
        parsed = pd.read_csv(cfg.io.csv_path, comment="#")
        assert list(parsed["identifier"]) == ["f:scan:1"]


class TestExportResults:
    def test_offsets_buckets_without_mutating_inputs_and_natural_sorts(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        consensus_factory,
        config_factory,
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
        rep1 = consensus_factory(
            precursor_mz=500.0,
            charge=2,
            mz=(100.0, 200.0),
            intensity=(0.7, 0.7),
            rt=10.0,
            cluster_id=0,
        )
        rep2 = consensus_factory(
            precursor_mz=600.0,
            charge=3,
            mz=(150.0, 250.0),
            intensity=(0.7, 0.7),
            rt=20.0,
            cluster_id=0,
        )
        labels1 = np.array([0, 0], dtype=np.int32)
        labels2 = np.array([0], dtype=np.int32)
        out_prefix = tmp_path / "run_v1.2"
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": True}
        )

        export.export_results(
            cfg,
            tmp_path,
            [ds1, ds2],
            [ClusterResult(labels1, [rep1]), ClusterResult(labels2, [rep2])],
        )

        # Caller label arrays must remain unmutated.
        np.testing.assert_array_equal(labels1, [0, 0])
        np.testing.assert_array_equal(labels2, [0])

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["identifier"]) == [
            "run:scan:1",
            "run:scan:2",
            "run:scan:10",
        ]
        by_id = df.set_index("identifier")["cluster"].to_dict()
        assert by_id["run:scan:2"] == by_id["run:scan:10"] == 0
        assert by_id["run:scan:1"] == 1
        mgf_text = (tmp_path / "run_v1.2.mgf").read_text(encoding="utf-8")
        assert "TITLE=falcon:cluster:0" in mgf_text
        assert "TITLE=falcon:cluster:1" in mgf_text

    def test_csv_only_when_export_representatives_disabled(
        self, tmp_path, lance_dataset, spectrum_row, config_factory
    ):
        ds = lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            charge=2,
        )
        out_prefix = tmp_path / "csv_only"
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": False}
        )
        export.export_results(
            cfg,
            tmp_path,
            [ds],
            [ClusterResult(np.array([0], dtype=np.int32), [])],
        )
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["identifier"]) == ["s:1"]
        assert list(df["cluster"]) == [0]
        assert not (tmp_path / "csv_only.mgf").exists()

    @pytest.mark.parametrize("with_row", [False, True])
    def test_skips_thread_pool_when_clusters_or_representatives_empty(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        config_factory,
        monkeypatch,
        with_row,
    ):
        def _forbid_pool(*_args, **_kwargs):
            raise AssertionError("ThreadPoolExecutor should not be spawned")

        monkeypatch.setattr(
            concurrent.futures, "ThreadPoolExecutor", _forbid_pool
        )
        rows = (
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)]
            if with_row
            else []
        )
        ds = lance_dataset(rows, charge=2)
        labels = np.array([0] if with_row else [], dtype=np.int32)
        out_prefix = tmp_path / ("no_reps" if with_row else "empty_out")
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": True}
        )
        export.export_results(cfg, tmp_path, [ds], [ClusterResult(labels, [])])
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df.columns) == [
            "identifier",
            "precursor_charge",
            "precursor_mz",
            "retention_time",
            "cluster",
        ]
        assert list(df["identifier"]) == (["s:1"] if with_row else [])
        assert out_prefix.with_suffix(".mgf").is_file()

    def test_formats_mixed_known_and_unknown_charges_as_integers_in_csv(
        self, tmp_path, lance_dataset, spectrum_row, config_factory
    ):
        ds_known = lance_dataset(
            [
                spectrum_row(
                    "s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
                spectrum_row(
                    "s:2", 501.0, [100.0, 200.0], [1.0, 1.0], -2, np.nan
                ),
            ],
            charge=2,
        )
        ds_unknown = lance_dataset(
            [
                spectrum_row(
                    "s:3", 600.0, [100.0, 200.0], [1.0, 1.0], None, 20.0
                ),
            ],
            charge="unknown",
        )
        out_prefix = tmp_path / "mixed_charges"
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": False}
        )
        export.export_results(
            cfg,
            tmp_path,
            [ds_known, ds_unknown],
            [
                ClusterResult(np.array([0, 1], dtype=np.int32), []),
                ClusterResult(np.array([0], dtype=np.int32), []),
            ],
        )

        data_lines = [
            line
            for line in (tmp_path / "mixed_charges.csv")
            .read_text(encoding="utf-8")
            .splitlines()
            if not line.startswith("#")
        ]
        assert data_lines[0] == (
            "identifier,precursor_charge,precursor_mz,retention_time,cluster"
        )
        rows_by_id = {
            parts[0]: parts
            for parts in (line.split(",") for line in data_lines[1:])
        }
        assert rows_by_id["s:1"][1] == "2"
        assert rows_by_id["s:2"][1] == "-2"
        assert rows_by_id["s:2"][3] == ""
        assert rows_by_id["s:3"][1] == ""

    @pytest.mark.parametrize("export_reps", [False, True])
    def test_propagates_csv_writer_exceptions(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        export_reps,
        config_factory,
    ):
        ds = lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            charge=2,
        )
        bad_prefix = tmp_path / "nonexistent_dir" / "out"
        cfg = config_factory(
            output_filename=bad_prefix,
            io={"export_representatives": export_reps},
        )
        with pytest.raises(FileNotFoundError):
            export.export_results(
                cfg,
                tmp_path,
                [ds],
                [ClusterResult(np.array([0], dtype=np.int32), [])],
            )

    def test_propagates_mgf_writer_exception_via_thread_pool(
        self,
        tmp_path,
        lance_dataset,
        spectrum_row,
        consensus_factory,
        config_factory,
        monkeypatch,
    ):
        ds = lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            charge=2,
        )
        rep = consensus_factory(cluster_id=0)
        cfg = config_factory(
            output_filename=tmp_path / "out",
            io={"export_representatives": True},
        )

        def _boom(*_args, **_kwargs):
            raise OSError("mgf disk full")

        monkeypatch.setattr(export.ms_io, "write_spectra", _boom)
        with pytest.raises(OSError, match="mgf disk full"):
            export.export_results(
                cfg,
                tmp_path,
                [ds],
                [ClusterResult(np.array([0], dtype=np.int32), [rep])],
            )

    @pytest.mark.parametrize("bad_labels", [[], [0]])
    def test_rejects_label_count_mismatch_with_dataset_rows(
        self, tmp_path, lance_dataset, spectrum_row, config_factory, bad_labels
    ):
        ds = lance_dataset(
            [
                spectrum_row(
                    "s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0
                ),
                spectrum_row(
                    "s:2", 501.0, [100.0, 200.0], [1.0, 1.0], 2, 11.0
                ),
            ],
            charge=2,
        )
        cfg = config_factory(output_filename=tmp_path / "mismatch")
        with pytest.raises(ValueError, match="does not match"):
            export.export_results(
                cfg,
                tmp_path,
                [ds],
                [ClusterResult(np.array(bad_labels, dtype=np.int32), [])],
            )
