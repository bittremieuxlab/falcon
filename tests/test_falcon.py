"""Unit tests for falcon.falcon orchestration helpers."""

import logging
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from falcon import falcon as falcon_mod
from falcon.cluster.consensus import ConsensusTuple


def _parse_global(reset_global_config, args):
    reset_global_config.parse(args)
    return reset_global_config


# ----------------------------------------------------------------------
# Working directory setup and output validation
# ----------------------------------------------------------------------


class TestSetupWorkDir:
    def test_creates_temp_dir_when_work_dir_omitted(self, reset_global_config):
        cfg = _parse_global(reset_global_config, ["in.mgf", "out"])
        rm_work_dir = falcon_mod._setup_work_dir()
        try:
            assert rm_work_dir is True
            assert os.path.isdir(cfg.work_dir)
            assert os.path.isdir(os.path.join(cfg.work_dir, "spectra"))
        finally:
            os.rmdir(os.path.join(cfg.work_dir, "spectra"))
            os.rmdir(cfg.work_dir)

    def test_explicit_work_dir_new_and_existing_warns(
        self, tmp_path, reset_global_config, caplog
    ):
        wd = tmp_path / "work"
        _parse_global(
            reset_global_config, ["in.mgf", "out", "--work_dir", str(wd)]
        )
        assert falcon_mod._setup_work_dir() is False
        assert (wd / "spectra").is_dir()

        # Re-running against an existing work_dir logs a warning.
        with caplog.at_level(logging.WARNING, logger="falcon"):
            assert falcon_mod._setup_work_dir() is False
        assert any("already exists" in r.message for r in caplog.records)


class TestValidateOutputs:
    def test_allows_run_when_outputs_do_not_exist(
        self, tmp_path, reset_global_config
    ):
        _parse_global(
            reset_global_config, ["in.mgf", str(tmp_path / "results")]
        )
        assert falcon_mod._validate_outputs() is False

    @pytest.mark.parametrize("existing_ext", ["csv", "mgf"])
    def test_aborts_without_overwrite_preserving_existing_file(
        self, tmp_path, reset_global_config, existing_ext
    ):
        prefix = tmp_path / "results"
        existing = tmp_path / f"results.{existing_ext}"
        existing.write_text("keep")
        _parse_global(reset_global_config, ["in.mgf", str(prefix)])

        assert falcon_mod._validate_outputs() is True
        assert existing.read_text() == "keep"

    def test_removes_both_csv_and_mgf_when_overwrite_enabled(
        self, tmp_path, reset_global_config
    ):
        prefix = tmp_path / "results"
        csv_file = tmp_path / "results.csv"
        mgf_file = tmp_path / "results.mgf"
        csv_file.write_text("old csv")
        mgf_file.write_text("old mgf")
        _parse_global(
            reset_global_config, ["in.mgf", str(prefix), "--overwrite"]
        )

        assert falcon_mod._validate_outputs() is False
        assert not csv_file.exists()
        assert not mgf_file.exists()


class TestWarnAggressiveOutlierCutoffs:
    def test_warns_only_for_average_when_both_cutoffs_below_one(
        self, reset_global_config, caplog
    ):
        _parse_global(
            reset_global_config,
            [
                "in.mgf",
                "out",
                "--consensus_method",
                "average",
                "--outlier_cutoff_lower",
                "0.5",
                "--outlier_cutoff_upper",
                "0.8",
            ],
        )
        with caplog.at_level(logging.WARNING, logger="falcon"):
            falcon_mod._warn_aggressive_outlier_cutoffs()
        assert any("less than 1" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "extra_args",
        [
            [
                "--consensus_method",
                "medoid",
                "--outlier_cutoff_lower",
                "0.5",
                "--outlier_cutoff_upper",
                "0.5",
            ],
            [
                "--consensus_method",
                "average",
                "--outlier_cutoff_lower",
                "0.5",
                "--outlier_cutoff_upper",
                "1.0",
            ],
        ],
    )
    def test_no_warning_outside_aggressive_average_regime(
        self, reset_global_config, caplog, extra_args
    ):
        _parse_global(reset_global_config, ["in.mgf", "out", *extra_args])
        with caplog.at_level(logging.WARNING, logger="falcon"):
            falcon_mod._warn_aggressive_outlier_cutoffs()
        assert caplog.records == []


class TestBuildProcessSpectrum:
    def test_maps_scaling_off_to_none_and_applies_get_dim_bounds(
        self, reset_global_config
    ):
        _parse_global(
            reset_global_config,
            ["in.mgf", "out", "--scaling", "off", "--min_mz", "101"],
        )
        proc = falcon_mod._build_process_spectrum()
        assert proc.keywords["scaling"] is None
        assert proc.keywords["mz_min"] <= 101.0

        _parse_global(
            reset_global_config, ["in.mgf", "out", "--scaling", "root"]
        )
        proc_root = falcon_mod._build_process_spectrum()
        assert proc_root.keywords["scaling"] == "root"


# ----------------------------------------------------------------------
# Bucket cache reuse, mismatch, and overwrite clearing
# ----------------------------------------------------------------------


class TestLoadOrPrepareBuckets:
    def test_reuses_matching_cache_without_repreparing(
        self, tmp_path, reset_global_config, monkeypatch
    ):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        joblib.dump(
            {"config": [{1}, {2}], "buckets": [{1}, {2}]},
            spectra_dir / "charges.joblib",
        )
        _parse_global(
            reset_global_config,
            [
                "in.mgf",
                "out",
                "--work_dir",
                str(tmp_path),
                "--precursor_charge_buckets",
                "[1]",
                "[2]",
            ],
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError("_prepare_spectra should not be called")

        monkeypatch.setattr(falcon_mod, "_prepare_spectra", _fail)
        assert falcon_mod._load_or_prepare_buckets(lambda s: None) == [
            {1},
            {2},
        ]

    @pytest.mark.parametrize(
        "cached_payload",
        [
            # Mismatched config vs None.
            {"config": [{1}], "buckets": [{1}]},
            # Legacy list cache without config key.
            [(1,), (2,)],
        ],
    )
    def test_mismatched_or_legacy_cache_raises_value_error(
        self, tmp_path, reset_global_config, cached_payload
    ):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        joblib.dump(cached_payload, spectra_dir / "charges.joblib")
        _parse_global(
            reset_global_config,
            ["in.mgf", "out", "--work_dir", str(tmp_path)],
        )
        with pytest.raises(ValueError, match="Cached charge buckets"):
            falcon_mod._load_or_prepare_buckets(lambda s: None)

    def test_overwrite_clears_cached_directories_and_files(
        self, tmp_path, reset_global_config, monkeypatch
    ):
        spectra_dir = tmp_path / "spectra"
        stale_dir = spectra_dir / "spectra_charge_99.lance"
        stale_dir.mkdir(parents=True)
        (stale_dir / "data").write_text("stale")
        (spectra_dir / "charges.joblib").write_text("stale")

        _parse_global(
            reset_global_config,
            ["in.mgf", "out", "--work_dir", str(tmp_path), "--overwrite"],
        )
        monkeypatch.setattr(
            falcon_mod, "_prepare_spectra", lambda _proc, _buckets: [(2,)]
        )

        buckets = falcon_mod._load_or_prepare_buckets(lambda s: None)
        assert buckets == [(2,)]
        assert not stale_dir.exists()
        saved = joblib.load(spectra_dir / "charges.joblib")
        assert saved == {"config": None, "buckets": [(2,)]}


# ----------------------------------------------------------------------
# _run_clustering and _export_results
# ----------------------------------------------------------------------


class TestRunClusteringAndExportResults:
    def test_run_clustering_passes_average_params_and_skips_empty_buckets(
        self,
        tmp_path,
        reset_global_config,
        lance_dataset,
        spectrum_row,
        monkeypatch,
    ):
        spectra_dir = tmp_path / "spectra"
        spectra_dir.mkdir()
        # Create an empty bucket (charge 1) and a populated one (2).
        lance_dataset([], name="spectra/spectra_charge_1.lance")
        lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            name="spectra/spectra_charge_2.lance",
        )
        _parse_global(
            reset_global_config,
            [
                "in.mgf",
                "out",
                "--work_dir",
                str(tmp_path),
                "--consensus_method",
                "average",
                "--fragment_tol",
                "0.05",
                "--outlier_cutoff_lower",
                "2.0",
                "--outlier_cutoff_upper",
                "2.5",
            ],
        )

        captured = {}

        def _fake_multi(datasets, *args):
            captured["datasets"] = datasets
            captured["consensus_params"] = args[-1]
            return [(np.array([0], dtype=np.int32), [])]

        monkeypatch.setattr(
            falcon_mod.cluster, "generate_clusters_multi", _fake_multi
        )

        datasets, results = falcon_mod._run_clustering([(1,), (2,)])
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
        self, tmp_path, reset_global_config, lance_dataset, spectrum_row
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
        out_prefix = tmp_path / "out"
        _parse_global(
            reset_global_config,
            ["in.mgf", str(out_prefix), "--export_representatives"],
        )

        falcon_mod._export_results(
            [ds1, ds2],
            [
                (np.array([0, 0], dtype=np.int32), [rep1]),
                (np.array([0], dtype=np.int32), [rep2]),
            ],
        )

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        # Natural sort puts scan:1, scan:2, scan:10 (not lex order).
        assert list(df["identifier"]) == [
            "run:scan:1",
            "run:scan:2",
            "run:scan:10",
        ]
        by_id = df.set_index("identifier")["cluster"].to_dict()
        assert by_id["run:scan:2"] == by_id["run:scan:10"] == 0
        assert by_id["run:scan:1"] == 1
        mgf_text = (tmp_path / "out.mgf").read_text()
        assert "TITLE=falcon:cluster:0" in mgf_text
        assert "TITLE=falcon:cluster:1" in mgf_text

    def test_export_results_empty_inputs_writes_empty_csv_and_mgf(
        self, tmp_path, reset_global_config
    ):
        out_prefix = tmp_path / "empty_out"
        _parse_global(
            reset_global_config,
            ["in.mgf", str(out_prefix), "--export_representatives"],
        )
        falcon_mod._export_results([], [])
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
        self, tmp_path, reset_global_config, lance_dataset, spectrum_row
    ):
        ds = lance_dataset(
            [spectrum_row("s:1", 500.0, [100.0, 200.0], [1.0, 1.0], 2, 10.0)],
            charge=2,
        )
        bad_prefix = tmp_path / "nonexistent_dir" / "out"
        _parse_global(reset_global_config, ["in.mgf", str(bad_prefix)])

        with pytest.raises(FileNotFoundError):
            falcon_mod._export_results(
                [ds], [(np.array([0], dtype=np.int32), [])]
            )
