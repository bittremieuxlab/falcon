"""Unit tests for falcon.config models, TOML loading, and serialization."""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    ConfigError,
    FalconConfig,
    FalconSettings,
    IOConfig,
    MedoidConsensusConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    build_json_schema,
    format_metadata_lines,
    format_toml,
    load_config,
    load_settings,
    load_toml_file,
    merge_config_dicts,
    write_saved_config,
)


class TestModelsAndValidation:
    def test_defaults_immutability_and_strict_types(self):
        settings = FalconSettings()
        assert settings.io.input_filenames is None
        assert settings.preprocessing.min_peaks == 5
        assert settings.clustering.precursor_tol == PrecursorTolerance(
            mass=20.0, mode="ppm"
        )
        assert isinstance(settings.consensus, MedoidConsensusConfig)

        with pytest.raises(ValidationError):
            settings.preprocessing.min_peaks = 10  # type: ignore[misc]
        with pytest.raises(ValidationError):
            FalconSettings.model_validate({"unknown_section": {}})
        with pytest.raises(ValidationError):
            PreprocessingConfig.model_validate({"min_peaks": "5"})
        with pytest.raises(ValidationError):
            PreprocessingConfig.model_validate({"min_mz": float("nan")})

        # Int values are accepted for float fields (valid TOML integers).
        cfg = PreprocessingConfig.model_validate(
            {"min_mz": 100, "max_mz": 500}
        )
        assert cfg.min_mz == 100.0
        assert cfg.max_mz == 500.0

    def test_consensus_discriminated_union_and_sentinels(self):
        avg = FalconSettings.model_validate(
            {
                "consensus": {
                    "method": "average",
                    "outlier_cutoff_lower": 2.0,
                }
            }
        )
        assert isinstance(avg.consensus, AverageConsensusConfig)
        assert avg.consensus.outlier_cutoff_lower == 2.0

        # Mixing medoid with average-only cutoffs fails validation.
        with pytest.raises(ValidationError):
            FalconSettings.model_validate(
                {
                    "consensus": {
                        "method": "medoid",
                        "outlier_cutoff_lower": 2.0,
                    }
                }
            )

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


class TestTomlLoadingAndMerging:
    def test_merge_config_dicts_non_mutating_and_consensus_switch(self):
        base = {
            "clustering": {"fragment_tol": 0.05, "linkage": "complete"},
            "consensus": {"method": "average", "outlier_cutoff_lower": 2.0},
        }
        override = {
            "clustering": {"linkage": "average"},
            "consensus": {"method": "medoid"},
        }
        merged = merge_config_dicts(base, override)
        assert merged["clustering"] == {
            "fragment_tol": 0.05,
            "linkage": "average",
        }
        assert merged["consensus"] == {"method": "medoid"}
        assert base["consensus"]["outlier_cutoff_lower"] == 2.0

    def test_extends_diamond_inheritance_and_cycle_detection(self, tmp_path):
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
        assert loaded.preprocessing.min_peaks == 3
        assert loaded.preprocessing.max_mz == 900.0
        assert loaded.clustering.min_matched_peaks == 6
        assert loaded.clustering.distance_threshold == 0.2

        # Cycles fail with ConfigError.
        cyc_a = tmp_path / "cyc_a.toml"
        cyc_b = tmp_path / "cyc_b.toml"
        cyc_a.write_text('extends = "cyc_b.toml"\n', encoding="utf-8")
        cyc_b.write_text('extends = "cyc_a.toml"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="Cycle detected"):
            load_toml_file(cyc_a)

    def test_layer_precedence_and_invalid_files(self, tmp_path):
        c1 = tmp_path / "c1.toml"
        c2 = tmp_path / "c2.toml"
        c1.write_text(
            "[clustering]\ndistance_threshold = 0.2\nfragment_tol = 0.02\n",
            encoding="utf-8",
        )
        c2.write_text(
            "[clustering]\ndistance_threshold = 0.3\n", encoding="utf-8"
        )
        cfg = load_config(
            config_files=[c1, c2],
            overrides={
                "io": {
                    "input_filenames": ["a.mgf"],
                    "output_filename": "out",
                },
                "clustering": {"distance_threshold": 0.4},
            },
        )
        assert cfg.clustering.fragment_tol == 0.02
        assert cfg.clustering.distance_threshold == 0.4

        with pytest.raises(ConfigError, match="not found"):
            load_toml_file(tmp_path / "missing.toml")
        with pytest.raises(ConfigError, match="directory"):
            load_toml_file(tmp_path)
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[clustering\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_toml_file(bad_toml)
        with pytest.raises(ConfigError, match="sequence of paths"):
            load_settings(config_files=c1)  # type: ignore[arg-type]

    def test_metabolomics_example_config_loads(self):
        repo_cfg = (
            Path(__file__).resolve().parents[1] / "configs/metabolomics.toml"
        )
        settings = load_settings(config_files=[repo_cfg])
        assert settings.preprocessing.min_mz == 0.0
        assert settings.preprocessing.max_mz == 1000.0
        assert settings.clustering.min_matched_peaks == 6


class TestSerializationAndSchema:
    def test_format_toml_round_trip_and_escaping(self, tmp_path):
        default_settings = FalconSettings()
        default_toml = format_toml(default_settings)
        assert '# rt_tol = "none"' in default_toml
        assert '# precursor_charge_buckets = "auto"' in default_toml
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
            consensus=AverageConsensusConfig(
                outlier_cutoff_lower=2.0, outlier_cutoff_upper=3.0
            ),
        )
        dest = tmp_path / "saved.toml"
        write_saved_config(custom, dest)
        assert load_config(config_files=[dest]) == custom

    def test_write_saved_config_rejects_dir_and_io_collisions(
        self, tmp_path, config_factory
    ):
        inp = tmp_path / "in.mgf"
        inp.write_text("MGF", encoding="utf-8")
        cfg = config_factory(
            input_filenames=(str(inp),),
            output_filename=tmp_path / "out",
            io={"work_dir": str(tmp_path / "wd")},
        )
        with pytest.raises(ConfigError, match="directory"):
            write_saved_config(cfg, tmp_path)
        for bad in (inp, cfg.io.csv_path, cfg.io.mgf_path, tmp_path / "wd"):
            with pytest.raises(ConfigError, match="aliases"):
                write_saved_config(cfg, bad)

    def test_metadata_lines_and_json_schema(self, tmp_path, config_factory):
        cfg = config_factory(
            consensus={"method": "average", "outlier_cutoff_lower": 2.0}
        )
        lines = format_metadata_lines(cfg, work_dir=tmp_path)
        assert lines[0].startswith("falcon version ")
        assert f"io.work_dir = {tmp_path}" in lines
        assert "clustering.precursor_tol = 20.0 ppm" in lines
        assert "consensus.outlier_cutoff_lower = 2.0" in lines

        schema = build_json_schema()
        serialized = json.dumps(schema)
        assert "extends" in schema["properties"]
        assert '"const": "none"' in serialized
        assert '"const": "auto"' in serialized
        assert "anyOf" in schema["properties"]["consensus"]
