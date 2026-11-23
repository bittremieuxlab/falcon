"""Unit tests for falcon.export (output validation and CSV/MGF export)."""

import numpy as np
import pandas as pd
import pytest

from falcon import export
from falcon.config import FalconConfig, IOConfig


class TestValidateOutputs:
    def test_allows_run_when_outputs_do_not_exist(
        self, tmp_path, config_factory
    ):
        cfg = config_factory(output_filename=tmp_path / "results")
        assert export.validate_outputs(cfg.io) is False

    @pytest.mark.parametrize("existing_ext", ["csv", "mgf"])
    def test_aborts_without_overwrite_preserving_existing_file(
        self, tmp_path, existing_ext, config_factory
    ):
        prefix = tmp_path / "run_v1.2"
        existing = tmp_path / f"run_v1.2.{existing_ext}"
        existing.write_text("keep")
        cfg = config_factory(output_filename=prefix)

        assert export.validate_outputs(cfg.io) is True
        assert existing.read_text() == "keep"

    def test_removes_both_csv_and_mgf_when_overwrite_enabled(
        self, tmp_path, config_factory
    ):
        prefix = tmp_path / "run_v1.2"
        csv_file = tmp_path / "run_v1.2.csv"
        mgf_file = tmp_path / "run_v1.2.mgf"
        csv_file.write_text("old csv")
        mgf_file.write_text("old mgf")
        cfg = config_factory(output_filename=prefix, io={"overwrite": True})

        assert export.validate_outputs(cfg.io) is False
        assert not csv_file.exists()
        assert not mgf_file.exists()


class TestWriteClusterCsv:
    def test_writes_metadata_comments_and_overwrites_cleanly(self, tmp_path):
        cfg = FalconConfig(
            io=IOConfig(
                input_filenames=("in.mgf",),
                output_filename=tmp_path / "run_v1.2",
            )
        )
        df = pd.DataFrame(
            {
                "identifier": ["f:scan:1", "f:scan:2"],
                "precursor_charge": [2, 2],
                "precursor_mz": [500.0, 500.05],
                "retention_time": [10.0, 11.0],
                "cluster": [0, 0],
            }
        )
        # Write twice to prove "w" mode does not append duplicate tables.
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
            charge=np.int32(2),
            mz=(100.0, 200.0),
            intensity=(0.7, 0.7),
            rt=10.0,
            cluster_id=0,
        )
        rep2 = consensus_factory(
            precursor_mz=600.0,
            charge=np.int32(3),
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
            [(labels1, [rep1]), (labels2, [rep2])],
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

    def test_empty_inputs_and_empty_bucket_writes_empty_csv_and_mgf(
        self, tmp_path, lance_dataset, config_factory
    ):
        empty_ds = lance_dataset([], charge=2)
        out_prefix = tmp_path / "empty_out"
        cfg = config_factory(
            output_filename=out_prefix, io={"export_representatives": True}
        )
        export.export_results(
            cfg, tmp_path, [empty_ds], [(np.empty(0, dtype=np.int32), [])]
        )
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

    @pytest.mark.parametrize("export_reps", [False, True])
    def test_propagates_writer_exceptions(
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
                cfg, tmp_path, [ds], [(np.array([0], dtype=np.int32), [])]
            )
