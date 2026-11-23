"""Unit tests for falcon.config.loader TOML loading and merging."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon.config import (
    AverageConsensusConfig,
    ConfigError,
    FalconSettings,
    MedoidConsensusConfig,
    load_config,
    load_settings,
    load_toml_file,
    merge_config_dicts,
)


class TestMergeConfigDicts:
    def test_deep_merges_and_does_not_mutate(self):
        base = {
            "preprocessing": {"min_mz": 101.0, "max_mz": 1500.0},
            "clustering": {
                "precursor_tol": {"mass": 20.0, "mode": "ppm"},
                "precursor_charge_buckets": [[1], [2]],
            },
        }
        override = {
            "preprocessing": {"min_mz": 0.0},
            "clustering": {
                "precursor_tol": {"mass": 10.0},
                "precursor_charge_buckets": [[2, 3]],
            },
            "io": {"overwrite": True},
        }
        merged = merge_config_dicts(base, override)
        assert merged == {
            "preprocessing": {"min_mz": 0.0, "max_mz": 1500.0},
            "clustering": {
                "precursor_tol": {"mass": 10.0, "mode": "ppm"},
                "precursor_charge_buckets": [[2, 3]],
            },
            "io": {"overwrite": True},
        }
        # Originals remain untouched.
        assert base["preprocessing"]["min_mz"] == 101.0
        assert base["clustering"]["precursor_charge_buckets"] == [[1], [2]]
        merged["clustering"]["precursor_charge_buckets"].append("other")
        assert override["clustering"]["precursor_charge_buckets"] == [[2, 3]]

    def test_consensus_discriminator_merge_semantics(self):
        base = {
            "consensus": {
                "method": "average",
                "outlier_cutoff_lower": 2.0,
                "outlier_cutoff_upper": 3.0,
            }
        }
        # Partial override without method preserves average + sibling.
        partial = merge_config_dicts(
            base, {"consensus": {"outlier_cutoff_upper": 2.5}}
        )
        settings_avg = FalconSettings.model_validate(partial)
        assert settings_avg.consensus == AverageConsensusConfig(
            outlier_cutoff_lower=2.0, outlier_cutoff_upper=2.5
        )

        # Switching method to medoid drops stale average-only cutoffs.
        switched = merge_config_dicts(
            base, {"consensus": {"method": "medoid"}}
        )
        settings_med = FalconSettings.model_validate(switched)
        assert settings_med.consensus == MedoidConsensusConfig()

        # Same-layer method=medoid + outlier_cutoff fails validation.
        conflict = merge_config_dicts(
            base,
            {"consensus": {"method": "medoid", "outlier_cutoff_lower": 2.0}},
        )
        with pytest.raises(ValidationError):
            FalconSettings.model_validate(conflict)

    def test_consensus_reset_only_applies_at_top_level(self):
        nested_base = {
            "outer": {
                "consensus": {
                    "method": "average",
                    "keep": 1,
                }
            }
        }
        nested_override = {"outer": {"consensus": {"method": "medoid"}}}

        merged = merge_config_dicts(nested_base, nested_override)
        assert merged == {
            "outer": {
                "consensus": {
                    "method": "medoid",
                    "keep": 1,
                }
            }
        }


class TestLoadTomlFile:
    def test_no_extends_fast_path(self, tmp_path):
        standalone = tmp_path / "standalone.toml"
        standalone.write_text(
            "[preprocessing]\nmin_peaks = 9\n", encoding="utf-8"
        )
        assert load_toml_file(standalone) == {
            "preprocessing": {"min_peaks": 9}
        }

    def test_extends_diamond_inheritance_relative_to_declaring_file(
        self, tmp_path, monkeypatch
    ):
        presets = tmp_path / "presets"
        runs = tmp_path / "runs"
        other_cwd = tmp_path / "elsewhere"
        presets.mkdir()
        runs.mkdir()
        other_cwd.mkdir()

        (presets / "shared.toml").write_text(
            "[preprocessing]\nmin_mz = 50.0\nmax_mz = 1200.0\n",
            encoding="utf-8",
        )
        (presets / "left.toml").write_text(
            'extends = "shared.toml"\n'
            "[clustering]\nmin_matched_peaks = 6\ndistance_threshold = 0.2\n",
            encoding="utf-8",
        )
        (presets / "right.toml").write_text(
            'extends = "shared.toml"\n'
            "[clustering]\ndistance_threshold = 0.15\n"
            '[consensus]\nmethod = "average"\noutlier_cutoff_lower = 2.0\n',
            encoding="utf-8",
        )
        child = runs / "exp.toml"
        child.write_text(
            'extends = ["../presets/left.toml", "../presets/right.toml"]\n'
            "[preprocessing]\nmin_mz = 0.0\n",
            encoding="utf-8",
        )

        # Invoke from unrelated CWD; relative extends still resolve.
        monkeypatch.chdir(other_cwd)
        settings = load_settings(config_files=[child])
        assert settings.preprocessing.min_mz == 0.0
        assert settings.preprocessing.max_mz == 1200.0
        assert settings.clustering.min_matched_peaks == 6
        assert settings.clustering.distance_threshold == 0.15
        assert settings.consensus == AverageConsensusConfig(
            outlier_cutoff_lower=2.0, outlier_cutoff_upper=1.5
        )

    def test_extends_absolute_path(self, tmp_path):
        parent = tmp_path / "parent.toml"
        parent.write_text(
            "[clustering]\nmin_matched_peaks = 7\n", encoding="utf-8"
        )
        child = tmp_path / "child.toml"
        child.write_text(
            f'extends = "{parent}"\n[clustering]\ndistance_threshold = 0.2\n',
            encoding="utf-8",
        )
        settings = load_settings(config_files=[child])
        assert settings.clustering.min_matched_peaks == 7
        assert settings.clustering.distance_threshold == 0.2

    def test_extends_cycles_raise_config_error(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        a = tmp_path / "a.toml"
        b = sub / "b.toml"
        a.write_text('extends = "sub/b.toml"\n', encoding="utf-8")
        b.write_text('extends = "../a.toml"\n', encoding="utf-8")

        with pytest.raises(ConfigError, match="Cycle"):
            load_toml_file(a)

        self_ref = tmp_path / "self.toml"
        self_ref.write_text('extends = "self.toml"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="Cycle"):
            load_toml_file(self_ref)

    def test_symlink_loop_raises_config_error(self, tmp_path):
        loop = tmp_path / "loop.toml"
        loop.symlink_to(loop)
        with pytest.raises(
            ConfigError,
            match=r"Failed to resolve configuration file.*loop\.toml",
        ):
            load_toml_file(loop)

    def test_invalid_toml_files_raise_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match=r"missing\.toml"):
            load_settings(config_files=[tmp_path / "missing.toml"])

        adir = tmp_path / "adir"
        adir.mkdir()
        with pytest.raises(ConfigError, match="directory"):
            load_settings(config_files=[adir])

        bad_bytes = tmp_path / "binary.toml"
        bad_bytes.write_bytes(b"\xff\xfe[preprocessing]\n")
        with pytest.raises(ConfigError, match=r"binary\.toml"):
            load_settings(config_files=[bad_bytes])

        bad_syntax = tmp_path / "syntax.toml"
        bad_syntax.write_text("[preprocessing\nmin_peaks = 5\n")
        with pytest.raises(ConfigError, match=r"syntax\.toml"):
            load_settings(config_files=[bad_syntax])

        bad_parent = tmp_path / "child.toml"
        bad_parent.write_text('extends = "nope.toml"\n')
        with pytest.raises(
            ConfigError, match=r"nope\.toml.*referenced by.*child\.toml"
        ):
            load_settings(config_files=[bad_parent])

    @pytest.mark.parametrize(
        "extends_toml",
        [
            "extends = 123\n",
            "extends = []\n",
            'extends = ""\n',
            'extends = "   "\n',
            'extends = [""]\n',
            "extends = [1]\n",
        ],
    )
    def test_invalid_extends_types_raise_config_error(
        self, tmp_path, extends_toml
    ):
        bad = tmp_path / "bad_extends.toml"
        bad.write_text(extends_toml, encoding="utf-8")
        with pytest.raises(ConfigError, match="extends"):
            load_settings(config_files=[bad])


class TestLoadSettingsAndConfig:
    def test_sparse_toml_layering_and_overrides_precedence(self, tmp_path):
        base_toml = tmp_path / "base.toml"
        base_toml.write_text(
            "[preprocessing]\n"
            "min_mz = 0.0\n"
            "max_mz = 1000.0\n"
            "min_mz_range = 150.0\n"
            "[clustering]\n"
            "min_matched_peaks = 6\n"
            "rt_tol = 30.0\n"
            'precursor_charge_buckets = [[1], [2, 3], "other"]\n',
            encoding="utf-8",
        )
        run_toml = tmp_path / "run.toml"
        run_toml.write_text(
            "[io]\n"
            'input_filenames = ["a.mgf"]\n'
            'output_filename = "results"\n'
            "export_representatives = true\n"
            "[clustering]\n"
            "distance_threshold = 0.2\n"
            'rt_tol = "none"\n'
            'precursor_charge_buckets = "auto"\n',
            encoding="utf-8",
        )

        cfg = load_config(
            config_files=[base_toml, run_toml],
            overrides={"clustering": {"distance_threshold": 0.15}},
        )
        assert cfg.io.input_filenames == ("a.mgf",)
        assert cfg.io.output_filename == Path("results")
        assert cfg.io.export_representatives is True
        assert cfg.preprocessing.min_mz == 0.0
        assert cfg.preprocessing.max_mz == 1000.0
        assert cfg.preprocessing.min_mz_range == 150.0
        assert cfg.preprocessing.min_peaks == 5
        assert cfg.clustering.min_matched_peaks == 6
        assert cfg.clustering.distance_threshold == 0.15
        assert cfg.clustering.rt_tol is None
        assert cfg.clustering.precursor_charge_buckets is None

    def test_no_silent_autoload_of_cwd_config_files(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "config.ini").write_text("min_peaks = 99\n")
        (tmp_path / "falcon.toml").write_text(
            "[preprocessing]\nmin_peaks = 99\n"
        )
        monkeypatch.chdir(tmp_path)
        settings = load_settings()
        assert settings.preprocessing.min_peaks == 5

    def test_load_config_requires_io_paths(self):
        with pytest.raises(ValidationError):
            load_config()

    def test_metabolomics_example_config_loads_cleanly(self):
        src_root = Path(__file__).resolve().parents[2]
        metab_path = src_root / "configs" / "metabolomics.toml"
        settings = load_settings(config_files=[metab_path])
        assert settings.preprocessing.min_mz == 0.0
        assert settings.preprocessing.max_mz == 1000.0
        assert settings.preprocessing.min_mz_range == 150.0
        assert settings.preprocessing.min_peaks == 5
        assert settings.clustering.min_matched_peaks == 6
        assert settings.clustering.fragment_tol == 0.05
        assert settings.consensus == MedoidConsensusConfig()
