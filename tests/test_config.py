"""Unit tests for falcon.config models, TOML loading, and serialization."""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon.config import (
    ClusteringConfig,
    ConfigError,
    ConsensusConfig,
    FalconConfig,
    FalconSettings,
    IOConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    build_json_schema,
    charge_sort_key,
    format_metadata_lines,
    format_toml,
    load_config,
    load_settings,
    load_toml_file,
    merge_config_dicts,
    mz_bin_bounds,
    parse_charge_buckets,
    write_saved_config,
)
from tests.helpers import make_falcon_config


class TestModelsAndValidation:
    def test_consensus_validation_and_sentinels(self):
        avg = FalconSettings.model_validate(
            {
                "consensus": {
                    "method": "average",
                    "outlier_cutoff_lower": 2.0,
                }
            }
        )
        assert avg.consensus.method == "average"
        assert avg.consensus.outlier_cutoff_lower == 2.0

        # Mixing medoid (explicit or default) with average-only cutoffs fails.
        for bad_cons in (
            {"method": "medoid", "outlier_cutoff_lower": 2.0},
            {"outlier_cutoff_upper": 2.0},
        ):
            with pytest.raises(ValidationError):
                FalconSettings.model_validate({"consensus": bad_cons})

        # TOML string sentinels normalize to None.
        disabled = FalconSettings.model_validate(
            {
                "io": {"work_dir": "none"},
                "preprocessing": {
                    "remove_precursor_tol": "none",
                    "max_peaks_used": "none",
                },
                "clustering": {
                    "rt_tol": "none",
                    "precursor_charge_buckets": "auto",
                },
            }
        )
        assert disabled.io.work_dir is None
        assert disabled.preprocessing.remove_precursor_tol is None
        assert disabled.preprocessing.max_peaks_used is None
        assert disabled.clustering.rt_tol is None
        assert disabled.clustering.precursor_charge_buckets is None

    def test_cross_field_and_io_invariants(self):
        io_cfg = IOConfig(
            input_filenames=("a.mgf",), output_filename=Path("dir/run.v1")
        )
        assert io_cfg.csv_path == Path("dir/run.v1.csv")
        assert io_cfg.mgf_path == Path("dir/run.v1.mgf")

        with pytest.raises(ValidationError, match="min_mz"):
            PreprocessingConfig(min_mz=500.0, max_mz=500.0)
        with pytest.raises(ValidationError, match="min_mz_range"):
            PreprocessingConfig(min_mz=100.0, max_mz=200.0, min_mz_range=150.0)
        with pytest.raises(ValidationError, match="max_peaks_used"):
            PreprocessingConfig(min_peaks=10, max_peaks_used=5)
        with pytest.raises(ValidationError, match="min_matched_peaks"):
            FalconSettings(
                preprocessing=PreprocessingConfig(max_peaks_used=5),
                clustering=ClusteringConfig(min_matched_peaks=6),
            )
        with pytest.raises(ValidationError):
            IOConfig.model_validate(
                {"input_filenames": [], "output_filename": "out"}
            )
        with pytest.raises(ValidationError):
            IOConfig.model_validate(
                {"input_filenames": ["a.mgf"], "output_filename": "dir/"}
            )

    def test_charge_buckets_and_mz_bin_bounds(self):
        assert sorted(["unknown", 3, -1, 1], key=charge_sort_key) == [
            -1,
            1,
            3,
            "unknown",
        ]
        assert parse_charge_buckets(
            [["unknown", 2, 1], {-128, 127}, "other"]
        ) == ((1, 2, "unknown"), (-128, 127), "other")

        for raw_bad in (
            [],
            [[]],
            [[True]],
            [[-129]],
            [[128]],
            [[2, 2]],
            [[1, 2], [2, 3]],
            [[1], "other", "other"],
        ):
            with pytest.raises(ValueError, match=r".+"):
                parse_charge_buckets(raw_bad)

        # Float rounding cases: 1.7/0.1 -> 17*0.1 > 1.7; 4.3/0.1 -> 42.9999...
        for min_mz, max_mz, step in (
            (1.7, 2.0, 0.1),
            (0.0, 4.3, 0.1),
            (101.03, 1500.03, 0.05),
        ):
            n_bins, start, end = mz_bin_bounds(min_mz, max_mz, step)
            assert start <= min_mz
            assert end > max_mz
            assert n_bins >= 1
            assert end - start == pytest.approx(n_bins * step)


class TestTomlLoadingAndSerialization:
    def test_merge_and_extends_diamond_with_cycle_detection(self, tmp_path):
        base_dict = {
            "clustering": {"fragment_tol": 0.05, "linkage": "complete"},
            "consensus": {"method": "average", "outlier_cutoff_lower": 2.0},
        }
        merged = merge_config_dicts(
            base_dict,
            {
                "clustering": {"linkage": "average"},
                "consensus": {"method": "medoid"},
            },
        )
        assert merged["consensus"] == {"method": "medoid"}
        assert base_dict["consensus"]["outlier_cutoff_lower"] == 2.0

        base = tmp_path / "base.toml"
        base.write_text(
            "[preprocessing]\nmin_peaks = 3\nmin_mz = 50.0\n", encoding="utf-8"
        )
        left = tmp_path / "left.toml"
        left.write_text(
            'extends = "base.toml"\n[preprocessing]\nmax_mz = 900.0\n',
            encoding="utf-8",
        )
        right = tmp_path / "right.toml"
        right.write_text(
            'extends = "base.toml"\n[clustering]\nmin_matched_peaks = 6\n',
            encoding="utf-8",
        )
        child = tmp_path / "child.toml"
        child.write_text(
            'extends = ["left.toml", "right.toml"]\n'
            "[clustering]\ndistance_threshold = 0.2\n",
            encoding="utf-8",
        )
        loaded = load_settings(config_files=[child])
        assert loaded.preprocessing.min_mz == 50.0
        assert loaded.preprocessing.max_mz == 900.0
        assert loaded.clustering.min_matched_peaks == 6
        assert loaded.clustering.distance_threshold == 0.2

        cyc_a, cyc_b = tmp_path / "cyc_a.toml", tmp_path / "cyc_b.toml"
        cyc_a.write_text('extends = "cyc_b.toml"\n', encoding="utf-8")
        cyc_b.write_text('extends = "cyc_a.toml"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="Cycle detected"):
            load_toml_file(cyc_a)

        with pytest.raises(ConfigError, match="not found"):
            load_toml_file(tmp_path / "missing.toml")
        with pytest.raises(ConfigError, match="directory"):
            load_toml_file(tmp_path)
        with pytest.raises(ConfigError, match="sequence of paths"):
            load_settings(config_files=base)  # type: ignore[arg-type]

    def test_metabolomics_example_config_loads(self):
        repo_cfg = (
            Path(__file__).resolve().parents[1] / "configs/metabolomics.toml"
        )
        settings = load_settings(config_files=[repo_cfg])
        assert settings.preprocessing.min_mz == 0.0
        assert settings.preprocessing.max_mz == 1000.0
        assert settings.clustering.min_matched_peaks == 6

    def test_toml_roundtrip_metadata_and_json_schema(self, tmp_path):
        default_settings = FalconSettings()
        default_toml = format_toml(default_settings)
        assert '# rt_tol = "none"' in default_toml
        assert "outlier_cutoff_lower" not in default_toml
        assert (
            load_settings(overrides=tomllib.loads(default_toml))
            == default_settings
        )

        custom = FalconConfig(
            io=IOConfig(
                input_filenames=(str(tmp_path / 'sample_"🧪".mgf'),),
                output_filename=tmp_path / "run_v1.2",
                work_dir=tmp_path / "work",
                overwrite=True,
                export_representatives=True,
            ),
            preprocessing=PreprocessingConfig(
                remove_precursor_tol=None, max_peaks_used=None, scaling="root"
            ),
            clustering=ClusteringConfig(
                precursor_tol=PrecursorTolerance(mass=0.05, mode="Da"),
                rt_tol=15.5,
                precursor_charge_buckets=((1,), (2, 3), ("unknown",), "other"),
            ),
            consensus=ConsensusConfig(
                method="average",
                outlier_cutoff_lower=2.0,
                outlier_cutoff_upper=3.0,
            ),
        )
        dest = tmp_path / "saved.toml"
        write_saved_config(custom, dest)
        assert load_config(config_files=[dest]) == custom
        with pytest.raises(ConfigError, match="directory"):
            write_saved_config(custom, tmp_path)

        cfg = make_falcon_config(
            consensus={"method": "average", "outlier_cutoff_lower": 2.0}
        )
        lines = format_metadata_lines(cfg, work_dir=tmp_path)
        assert lines[0].startswith("falcon version ")
        assert "clustering.precursor_tol = 20.0 ppm" in lines
        assert "consensus.outlier_cutoff_lower = 2.0" in lines

        schema = build_json_schema()
        serialized = json.dumps(schema)
        assert "extends" in schema["properties"]
        assert '"const": "none"' in serialized
        assert '"const": "auto"' in serialized
