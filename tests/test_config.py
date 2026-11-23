"""Unit tests for falcon.config."""

import json
import os
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon.config import (
    STAGING_CACHE_SCHEMA_VERSION,
    AverageConsensusConfig,
    ClusteringConfig,
    ConfigError,
    FalconConfig,
    FalconSettings,
    IOConfig,
    MedoidConsensusConfig,
    PartialIOConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    build_json_schema,
    build_staging_manifest,
    deserialize_cached_buckets,
    diff_staging_manifests,
    load_config,
    load_settings,
    load_toml_file,
    merge_config_dicts,
)

# ----------------------------------------------------------------------
# Defaults, immutability, and strict type enforcement
# ----------------------------------------------------------------------


class TestConfigModels:
    def test_default_settings_and_immutability(self):
        cfg = FalconConfig(
            io=IOConfig(
                input_filenames=("in.mgf",),
                output_filename=Path("out"),
            )
        )
        assert cfg.io.input_filenames == ("in.mgf",)
        assert cfg.io.output_filename == Path("out")
        assert cfg.io.work_dir is None
        assert cfg.io.overwrite is False
        assert cfg.io.export_representatives is False
        assert cfg.preprocessing.min_peaks == 5
        assert cfg.preprocessing.min_mz == 101.0
        assert cfg.preprocessing.max_mz == 1500.0
        assert cfg.preprocessing.min_mz_range == 250.0
        assert cfg.preprocessing.remove_precursor_tol == 1.5
        assert cfg.preprocessing.min_intensity == 0.01
        assert cfg.preprocessing.max_peaks_used == 50
        assert cfg.preprocessing.scaling == "off"
        assert cfg.clustering.precursor_tol == PrecursorTolerance(
            mass=20.0, mode="ppm"
        )
        assert cfg.clustering.rt_tol is None
        assert cfg.clustering.fragment_tol == 0.05
        assert cfg.clustering.linkage == "complete"
        assert cfg.clustering.distance_threshold == 0.1
        assert cfg.clustering.min_matched_peaks == 0
        assert cfg.clustering.batch_size == 32768
        assert cfg.clustering.precursor_charge_buckets is None
        assert cfg.consensus == MedoidConsensusConfig()
        assert hash(cfg) == hash(cfg)
        with pytest.raises(ValidationError):
            cfg.clustering.batch_size = 10  # type: ignore[misc]

    def test_falcon_settings_defaults_without_io_paths(self):
        settings = FalconSettings()
        assert settings.io == PartialIOConfig()
        assert settings.io.input_filenames is None
        assert settings.io.output_filename is None
        assert isinstance(ConfigError("bad"), ValueError)

    def test_int_accepted_for_float_fields(self):
        cfg = PreprocessingConfig(
            min_mz=0, max_mz=1000, min_mz_range=150, remove_precursor_tol=2
        )
        assert cfg.min_mz == 0.0
        assert cfg.max_mz == 1000.0
        assert cfg.min_mz_range == 150.0
        assert cfg.remove_precursor_tol == 2.0

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (PreprocessingConfig, {"min_peaks": True}),
            (PreprocessingConfig, {"min_peaks": 5.0}),
            (PreprocessingConfig, {"min_peaks": "5"}),
            (PreprocessingConfig, {"min_mz": True}),
            (PreprocessingConfig, {"min_mz": "101.0"}),
            (PreprocessingConfig, {"max_peaks_used": False}),
            (PreprocessingConfig, {"max_peaks_used": 50.0}),
            (PreprocessingConfig, {"unknown_key": 1}),
            (ClusteringConfig, {"batch_size": True}),
            (ClusteringConfig, {"batch_size": 10.0}),
            (ClusteringConfig, {"min_matched_peaks": False}),
            (ClusteringConfig, {"distance_threshold": "0.1"}),
            (ClusteringConfig, {"distance_threshold": True}),
            (ClusteringConfig, {"extra_field": 123}),
            (PrecursorTolerance, {"mass": True}),
            (PrecursorTolerance, {"mass": "20"}),
            (PrecursorTolerance, {"mode": "PPM"}),
            (PrecursorTolerance, {"mode": "da"}),
            (PrecursorTolerance, {"extra": 1}),
            (PartialIOConfig, {"overwrite": 1}),
            (PartialIOConfig, {"overwrite": "true"}),
            (PartialIOConfig, {"export_representatives": "yes"}),
            (PartialIOConfig, {"work_dir": 123}),
            (PartialIOConfig, {"work_dir": True}),
            (MedoidConsensusConfig, {"outlier_cutoff_lower": 1.5}),
            (AverageConsensusConfig, {"outlier_cutoff_lower": True}),
            (AverageConsensusConfig, {"outlier_cutoff_upper": "1.5"}),
            (FalconSettings, {"unknown_section": {}}),
        ],
    )
    def test_strict_types_and_extra_forbid(self, model_cls, kwargs):
        with pytest.raises(ValidationError):
            model_cls.model_validate(kwargs)

    @pytest.mark.parametrize("scaling", ["off", "root", "log", "rank"])
    def test_valid_scaling_choices(self, scaling):
        assert PreprocessingConfig(scaling=scaling).scaling == scaling

    @pytest.mark.parametrize("linkage", ["single", "complete", "average"])
    def test_valid_linkage_choices(self, linkage):
        assert ClusteringConfig(linkage=linkage).linkage == linkage

    def test_consensus_discriminated_union(self):
        avg = FalconSettings.model_validate(
            {
                "consensus": {
                    "method": "average",
                    "outlier_cutoff_lower": 2.0,
                    "outlier_cutoff_upper": 3.0,
                }
            }
        )
        assert isinstance(avg.consensus, AverageConsensusConfig)
        assert avg.consensus.outlier_cutoff_lower == 2.0
        assert avg.consensus.outlier_cutoff_upper == 3.0

        with pytest.raises(ValidationError):
            FalconSettings.model_validate(
                {
                    "consensus": {
                        "method": "medoid",
                        "outlier_cutoff_lower": 2.0,
                    }
                }
            )


# ----------------------------------------------------------------------
# Numeric bounds, finiteness, and nullable TOML sentinels
# ----------------------------------------------------------------------


class TestBoundsAndSentinels:
    @pytest.mark.parametrize(
        "bad_float", [float("nan"), float("inf"), -float("inf")]
    )
    def test_non_finite_floats_rejected(self, bad_float):
        with pytest.raises(ValidationError):
            PrecursorTolerance(mass=bad_float, mode="ppm")
        with pytest.raises(ValidationError):
            ClusteringConfig(rt_tol=bad_float)
        with pytest.raises(ValidationError):
            ClusteringConfig(fragment_tol=bad_float)
        with pytest.raises(ValidationError):
            ClusteringConfig(distance_threshold=bad_float)
        with pytest.raises(ValidationError):
            PreprocessingConfig(min_mz_range=bad_float)
        with pytest.raises(ValidationError):
            PreprocessingConfig(min_mz=bad_float)
        with pytest.raises(ValidationError):
            PreprocessingConfig(max_mz=bad_float)
        with pytest.raises(ValidationError):
            PreprocessingConfig(remove_precursor_tol=bad_float)
        with pytest.raises(ValidationError):
            PreprocessingConfig(min_intensity=bad_float)
        with pytest.raises(ValidationError):
            AverageConsensusConfig(outlier_cutoff_lower=bad_float)
        with pytest.raises(ValidationError):
            AverageConsensusConfig(outlier_cutoff_upper=bad_float)

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (PrecursorTolerance, {"mass": -0.01, "mode": "Da"}),
            (ClusteringConfig, {"rt_tol": -1.0}),
            (ClusteringConfig, {"fragment_tol": 0.0}),
            (ClusteringConfig, {"fragment_tol": -0.05}),
            (ClusteringConfig, {"distance_threshold": 0.0}),
            (ClusteringConfig, {"distance_threshold": -0.1}),
            (ClusteringConfig, {"distance_threshold": 1.0}),
            (ClusteringConfig, {"distance_threshold": 1.2}),
            (ClusteringConfig, {"min_matched_peaks": -1}),
            (ClusteringConfig, {"batch_size": 0}),
            (ClusteringConfig, {"batch_size": -10}),
            (PreprocessingConfig, {"min_peaks": 0}),
            (PreprocessingConfig, {"min_mz": -1.0}),
            (PreprocessingConfig, {"max_mz": 0.0}),
            (PreprocessingConfig, {"min_mz_range": -0.1}),
            (PreprocessingConfig, {"remove_precursor_tol": 0.0}),
            (PreprocessingConfig, {"remove_precursor_tol": -1.0}),
            (PreprocessingConfig, {"min_intensity": -0.01}),
            (PreprocessingConfig, {"min_intensity": 1.0}),
            (PreprocessingConfig, {"min_intensity": 1.5}),
            (PreprocessingConfig, {"max_peaks_used": 0}),
            (AverageConsensusConfig, {"outlier_cutoff_lower": 0.0}),
            (AverageConsensusConfig, {"outlier_cutoff_lower": -1.0}),
            (AverageConsensusConfig, {"outlier_cutoff_upper": 0.0}),
            (AverageConsensusConfig, {"outlier_cutoff_upper": -0.5}),
        ],
    )
    def test_out_of_range_values_rejected(self, model_cls, kwargs):
        with pytest.raises(ValidationError):
            model_cls.model_validate(kwargs)

    def test_valid_edge_boundaries(self):
        tol = PrecursorTolerance(mass=0.0, mode="Da")
        assert tol.mass == 0.0
        clust = ClusteringConfig(
            rt_tol=0.0, distance_threshold=0.999, batch_size=1
        )
        assert clust.rt_tol == 0.0
        assert clust.batch_size == 1
        prep = PreprocessingConfig(
            min_peaks=1,
            min_mz=0.0,
            max_mz=10.0,
            min_mz_range=10.0,
            min_intensity=0.0,
            max_peaks_used=1,
        )
        assert prep.min_mz == 0.0
        assert prep.min_mz_range == 10.0

    def test_nullable_toml_sentinels_normalize_to_none(self):
        settings = FalconSettings.model_validate(
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
        assert settings.io.work_dir is None
        assert settings.preprocessing.remove_precursor_tol is None
        assert settings.preprocessing.max_peaks_used is None
        assert settings.clustering.rt_tol is None
        assert settings.clustering.precursor_charge_buckets is None

    @pytest.mark.parametrize(
        ("model_cls", "kwargs"),
        [
            (ClusteringConfig, {"rt_tol": "off"}),
            (ClusteringConfig, {"precursor_charge_buckets": "none"}),
            (ClusteringConfig, {"precursor_charge_buckets": "off"}),
            (PreprocessingConfig, {"remove_precursor_tol": "off"}),
            (PreprocessingConfig, {"max_peaks_used": "all"}),
        ],
    )
    def test_invalid_sentinel_strings_rejected(self, model_cls, kwargs):
        with pytest.raises(ValidationError):
            model_cls.model_validate(kwargs)


# ----------------------------------------------------------------------
# IOConfig path validation
# ----------------------------------------------------------------------


class TestIOConfig:
    def test_accepts_list_and_string_paths_preserving_dotted_prefix(self):
        io = IOConfig.model_validate(
            {
                "input_filenames": ["a.mgf", "b.mzml"],
                "output_filename": "dir/run_v1.2",
                "work_dir": "./none",
            }
        )
        assert io.input_filenames == ("a.mgf", "b.mzml")
        assert io.output_filename == Path("dir/run_v1.2")
        assert io.work_dir == Path("./none")

    @pytest.mark.parametrize(
        "payload",
        [
            {"input_filenames": [], "output_filename": "out"},
            {"input_filenames": [""], "output_filename": "out"},
            {"input_filenames": ["   "], "output_filename": "out"},
            {"input_filenames": "a.mgf", "output_filename": "out"},
            {"input_filenames": [123], "output_filename": "out"},
            {"input_filenames": ["a.mgf"], "output_filename": ""},
            {"input_filenames": ["a.mgf"], "output_filename": "   "},
            {"input_filenames": ["a.mgf"], "output_filename": Path(".")},
            {"input_filenames": ["a.mgf"], "output_filename": Path("/")},
            {
                "input_filenames": ["a.mgf"],
                "output_filename": "out",
                "work_dir": "",
            },
            {
                "input_filenames": ["a.mgf"],
                "output_filename": "out",
                "work_dir": "   ",
            },
        ],
    )
    def test_invalid_io_paths_rejected(self, payload):
        with pytest.raises(ValidationError):
            IOConfig.model_validate(payload)


# ----------------------------------------------------------------------
# Charge bucket validation and canonicalization
# ----------------------------------------------------------------------


class TestChargeBuckets:
    def test_canonicalizes_lists_tuples_and_sets(self):
        clust = ClusteringConfig.model_validate(
            {
                "precursor_charge_buckets": [
                    [3, 2],
                    {"unknown", -1},
                    (127, -128),
                    "other",
                ]
            }
        )
        assert clust.precursor_charge_buckets == (
            (2, 3),
            (-1, "unknown"),
            (-128, 127),
            "other",
        )

    @pytest.mark.parametrize(
        "buckets",
        [
            [],  # empty outer sequence
            [[]],  # empty inner bucket
            [[2, 2]],  # duplicate inside bucket
            [["unknown", "unknown"]],  # duplicate unknown inside bucket
            [[1, 2], [2, 3]],  # overlapping integer charge
            [[1, "unknown"], ["unknown"]],  # overlapping unknown
            [[1], "other", "other"],  # duplicate 'other'
            [["other"]],  # 'other' inside a bucket
            [[True]],  # bool inside bucket
            [[False]],  # bool inside bucket
            [[-129]],  # below int8 min
            [[128]],  # above int8 max
            [[1.5]],  # float charge
            [["abc"]],  # invalid string charge
            "12",  # bare string
            [1],  # bare int instead of bucket sequence
        ],
    )
    def test_invalid_charge_buckets_rejected(self, buckets):
        with pytest.raises(ValidationError):
            ClusteringConfig.model_validate(
                {"precursor_charge_buckets": buckets}
            )


# ----------------------------------------------------------------------
# Cross-field and cross-section invariants
# ----------------------------------------------------------------------


class TestCrossFieldInvariants:
    def test_preprocessing_cross_field_checks(self):
        with pytest.raises(ValidationError, match="min_mz"):
            PreprocessingConfig(min_mz=500.0, max_mz=500.0)
        with pytest.raises(ValidationError, match="min_mz"):
            PreprocessingConfig(min_mz=600.0, max_mz=500.0)
        with pytest.raises(ValidationError, match="min_mz_range"):
            PreprocessingConfig(min_mz=100.0, max_mz=200.0, min_mz_range=150.0)
        with pytest.raises(ValidationError, match="max_peaks_used"):
            PreprocessingConfig(min_peaks=10, max_peaks_used=5)

    def test_cross_section_min_matched_peaks_vs_max_peaks_used(self):
        with pytest.raises(ValidationError, match="min_matched_peaks"):
            FalconSettings(
                preprocessing=PreprocessingConfig(max_peaks_used=5),
                clustering=ClusteringConfig(min_matched_peaks=6),
            )
        # Valid when max_peaks_used is disabled (None).
        settings = FalconSettings(
            preprocessing=PreprocessingConfig(max_peaks_used=None),
            clustering=ClusteringConfig(min_matched_peaks=100),
        )
        assert settings.clustering.min_matched_peaks == 100

    def test_cross_section_fragment_tol_bin_overflow_rejected(self):
        with pytest.raises(ValidationError, match="fragment_tol"):
            FalconSettings(
                preprocessing=PreprocessingConfig(min_mz=0.0, max_mz=1500.0),
                clustering=ClusteringConfig(fragment_tol=1e-320),
            )
        with pytest.raises(ValidationError, match="bin"):
            FalconSettings(
                preprocessing=PreprocessingConfig(min_mz=0.0, max_mz=1500.0),
                clustering=ClusteringConfig(fragment_tol=1e-8),
            )


# ----------------------------------------------------------------------
# Hierarchical merging and TOML loading with extends
# ----------------------------------------------------------------------


class TestHierarchicalMergeAndLoading:
    def test_merge_config_dicts_deep_merges_and_does_not_mutate(self):
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
        }
        merged = merge_config_dicts(base, override)
        assert merged == {
            "preprocessing": {"min_mz": 0.0, "max_mz": 1500.0},
            "clustering": {
                "precursor_tol": {"mass": 10.0, "mode": "ppm"},
                "precursor_charge_buckets": [[2, 3]],
            },
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
        # Partial override without method preserves average + sibling cutoff.
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

        # Same-layer method=medoid + outlier_cutoff still fails validation.
        conflict = merge_config_dicts(
            base,
            {"consensus": {"method": "medoid", "outlier_cutoff_lower": 2.0}},
        )
        with pytest.raises(ValidationError):
            FalconSettings.model_validate(conflict)

    def test_sparse_toml_layering_and_cli_overrides_precedence(self, tmp_path):
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
        assert cfg.preprocessing.min_peaks == 5  # untouched default
        assert cfg.clustering.min_matched_peaks == 6
        assert cfg.clustering.distance_threshold == 0.15
        assert cfg.clustering.rt_tol is None
        assert cfg.clustering.precursor_charge_buckets is None

    def test_extends_single_and_list_relative_to_declaring_file(
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

        # Invoke from unrelated CWD; relative extends must still resolve.
        monkeypatch.chdir(other_cwd)
        settings = load_settings(config_files=[child])
        assert settings.preprocessing.min_mz == 0.0
        assert settings.preprocessing.max_mz == 1200.0
        assert settings.clustering.min_matched_peaks == 6
        assert settings.clustering.distance_threshold == 0.15
        assert settings.consensus == AverageConsensusConfig(
            outlier_cutoff_lower=2.0, outlier_cutoff_upper=1.5
        )

    def test_extends_indirect_cycle_raises_config_error(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        a = tmp_path / "a.toml"
        b = sub / "b.toml"
        a.write_text('extends = "sub/b.toml"\n', encoding="utf-8")
        b.write_text('extends = "../a.toml"\n', encoding="utf-8")

        with pytest.raises(ConfigError, match="Cycle"):
            load_toml_file(a)

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

    def test_invalid_toml_files_and_extends_raise_config_error(self, tmp_path):
        # Missing file
        with pytest.raises(ConfigError, match=r"missing\.toml"):
            load_settings(config_files=[tmp_path / "missing.toml"])

        # Directory path
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ConfigError, match="directory"):
            load_settings(config_files=[d])

        # Non-UTF-8 bytes
        bad_bytes = tmp_path / "binary.toml"
        bad_bytes.write_bytes(b"\xff\xfe[preprocessing]\n")
        with pytest.raises(ConfigError, match=r"binary\.toml"):
            load_settings(config_files=[bad_bytes])

        # Invalid TOML syntax
        bad_syntax = tmp_path / "syntax.toml"
        bad_syntax.write_text("[preprocessing\nmin_peaks = 5\n")
        with pytest.raises(ConfigError, match=r"syntax\.toml"):
            load_settings(config_files=[bad_syntax])

        # Missing extends target names both parent and child
        bad_parent = tmp_path / "child.toml"
        bad_parent.write_text('extends = "nope.toml"\n')
        with pytest.raises(ConfigError, match=r"nope\.toml"):
            load_settings(config_files=[bad_parent])

    @pytest.mark.parametrize(
        "extends_toml",
        [
            "extends = 123\n",
            "extends = []\n",
            'extends = ""\n',
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


# ----------------------------------------------------------------------
# TOML export, metadata lines, JSON Schema, and staging manifests
# ----------------------------------------------------------------------


class TestSerializationSchemaAndStaging:
    def test_to_toml_round_trip_and_commented_disabled_fields(self, tmp_path):
        default_settings = FalconSettings()
        default_toml = default_settings.to_toml()
        assert '# rt_tol = "none"' in default_toml
        assert '# precursor_charge_buckets = "auto"' in default_toml
        parsed_default = tomllib.loads(default_toml)
        assert load_settings(overrides=parsed_default) == default_settings

        custom = FalconConfig(
            io=IOConfig(
                input_filenames=("a.mgf", "b.mzml"),
                output_filename=Path("dir/run_v1.2"),
                work_dir=tmp_path / "falcon_work",
                overwrite=True,
                export_representatives=True,
            ),
            preprocessing=PreprocessingConfig(
                min_peaks=3,
                min_mz=0.0,
                max_mz=1000.0,
                min_mz_range=150.0,
                remove_precursor_tol=None,
                min_intensity=0.0,
                max_peaks_used=None,
                scaling="root",
            ),
            clustering=ClusteringConfig(
                precursor_tol=PrecursorTolerance(mass=0.05, mode="Da"),
                rt_tol=15.5,
                fragment_tol=0.02,
                linkage="average",
                distance_threshold=0.25,
                min_matched_peaks=6,
                batch_size=1024,
                precursor_charge_buckets=((1,), (2, 3), ("unknown",), "other"),
            ),
            consensus=AverageConsensusConfig(
                outlier_cutoff_lower=2.0, outlier_cutoff_upper=3.0
            ),
        )
        toml_text = custom.to_toml()
        assert '\nremove_precursor_tol = "none"\n' in toml_text
        assert '\nmax_peaks_used = "none"\n' in toml_text
        replayed = tmp_path / "replayed.toml"
        replayed.write_text(toml_text, encoding="utf-8")
        assert load_config(config_files=[replayed]) == custom

    def test_to_metadata_lines_medoid_and_average(self, tmp_path):
        work_dir = tmp_path / "wd"
        medoid_cfg = FalconConfig(
            io=IOConfig(input_filenames=("in.mgf",), output_filename=Path("o"))
        )
        med_lines = medoid_cfg.to_metadata_lines(work_dir=work_dir)
        assert any(line.startswith("falcon version ") for line in med_lines)
        assert f"io.work_dir = {work_dir}" in med_lines
        assert "clustering.precursor_tol = 20.0 ppm" in med_lines
        assert "consensus.method = medoid" in med_lines
        assert not any("outlier_cutoff" in line for line in med_lines)

        avg_cfg = FalconConfig(
            io=IOConfig(
                input_filenames=("in.mgf",), output_filename=Path("o")
            ),
            consensus=AverageConsensusConfig(
                outlier_cutoff_lower=2.0, outlier_cutoff_upper=3.5
            ),
        )
        avg_lines = avg_cfg.to_metadata_lines()
        assert "consensus.method = average" in avg_lines
        assert "consensus.outlier_cutoff_lower = 2.0" in avg_lines
        assert "consensus.outlier_cutoff_upper = 3.5" in avg_lines

    def test_build_json_schema_is_json_serializable_with_extends_and_sentinels(
        self,
    ):
        schema = build_json_schema()
        serialized = json.dumps(schema)
        assert isinstance(serialized, str)
        assert "extends" in schema["properties"]
        assert "io" in schema["properties"]
        assert "preprocessing" in schema["properties"]
        assert "clustering" in schema["properties"]
        assert "consensus" in schema["properties"]
        # Sentinels appear in schema definitions for TOML editors.
        assert '"const": "none"' in serialized
        assert '"const": "auto"' in serialized

    def test_staging_manifest_and_diff_detects_only_staging_changes(
        self, tmp_path
    ):
        f1 = tmp_path / "a.mgf"
        f2 = tmp_path / "b.mgf"
        f1.write_text("data1", encoding="utf-8")
        f2.write_text("data2", encoding="utf-8")

        base_cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(str(f1),),
                output_filename=tmp_path / "out1",
            )
        )
        base_manifest = build_staging_manifest(
            base_cfg, [f1], buckets=[(2,), ("unknown",), "other"]
        )
        assert base_manifest["schema_version"] == STAGING_CACHE_SCHEMA_VERSION
        assert base_manifest["buckets"] == [[2], ["unknown"], "other"]
        assert deserialize_cached_buckets(base_manifest["buckets"]) == [
            (2,),
            ("unknown",),
            "other",
        ]

        # Clustering/consensus/output changes must NOT invalidate cache.
        sweep_cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(str(f1),),
                output_filename=tmp_path / "out2",
                export_representatives=True,
            ),
            clustering=ClusteringConfig(
                distance_threshold=0.25,
                linkage="average",
                rt_tol=10.0,
                min_matched_peaks=6,
                batch_size=512,
            ),
            consensus=AverageConsensusConfig(),
        )
        assert (
            diff_staging_manifests(
                base_manifest, build_staging_manifest(sweep_cfg, [f1])
            )
            == []
        )

        # Preprocessing change invalidates cache.
        prep_cfg = FalconConfig(
            io=base_cfg.io,
            preprocessing=PreprocessingConfig(min_mz=50.0),
        )
        assert diff_staging_manifests(
            base_manifest, build_staging_manifest(prep_cfg, [f1])
        ) == ["preprocessing"]

        # fragment_tol change invalidates cache (affects get_dim bounds).
        frag_cfg = FalconConfig(
            io=base_cfg.io,
            clustering=ClusteringConfig(fragment_tol=0.02),
        )
        assert diff_staging_manifests(
            base_manifest, build_staging_manifest(frag_cfg, [f1])
        ) == ["fragment_tol"]

        # precursor_charge_buckets change invalidates cache.
        bucket_cfg = FalconConfig(
            io=base_cfg.io,
            clustering=ClusteringConfig(precursor_charge_buckets=((2,),)),
        )
        assert diff_staging_manifests(
            base_manifest, build_staging_manifest(bucket_cfg, [f1])
        ) == ["precursor_charge_buckets"]

        # Different input list invalidates cache.
        assert diff_staging_manifests(
            base_manifest, build_staging_manifest(base_cfg, [f1, f2])
        ) == ["inputs"]

        # Modified input file contents (size / mtime) invalidates cache.
        f1.write_text("modified_longer_content", encoding="utf-8")
        st = os.stat(f1)
        os.utime(f1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        assert diff_staging_manifests(
            base_manifest, build_staging_manifest(base_cfg, [f1])
        ) == ["inputs"]

        # Corrupted / non-dict / wrong schema version manifests fail.
        assert diff_staging_manifests(["legacy"], base_manifest) == [
            "schema_version"
        ]
        bad_ver = {**base_manifest, "schema_version": 999}
        assert diff_staging_manifests(bad_ver, base_manifest) == [
            "schema_version"
        ]

    def test_deserialize_cached_buckets_rejects_corrupted_entries(self):
        with pytest.raises(ValueError, match="buckets"):
            deserialize_cached_buckets("not-a-list")
        with pytest.raises(ValueError, match="bucket"):
            deserialize_cached_buckets([123])
        with pytest.raises(ValueError, match="bucket"):
            deserialize_cached_buckets([[True]])
