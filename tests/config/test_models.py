"""Unit tests for falcon.config.models models and validators."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon.config import (
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
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
)


class TestConfigModels:
    def test_default_settings_and_immutability(self):
        assert DEFAULT_MIN_MZ == 101.0
        assert DEFAULT_MAX_MZ == 1500.0
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
        assert cfg.preprocessing.min_mz == DEFAULT_MIN_MZ
        assert cfg.preprocessing.max_mz == DEFAULT_MAX_MZ
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


class TestBoundsAndSentinels:
    @pytest.mark.parametrize(
        "bad_float", [float("nan"), float("inf"), -float("inf")]
    )
    @pytest.mark.parametrize(
        ("model_cls", "field_name"),
        [
            (PrecursorTolerance, "mass"),
            (ClusteringConfig, "rt_tol"),
            (ClusteringConfig, "fragment_tol"),
            (ClusteringConfig, "distance_threshold"),
            (PreprocessingConfig, "min_mz_range"),
            (PreprocessingConfig, "min_mz"),
            (PreprocessingConfig, "max_mz"),
            (PreprocessingConfig, "remove_precursor_tol"),
            (PreprocessingConfig, "min_intensity"),
            (AverageConsensusConfig, "outlier_cutoff_lower"),
            (AverageConsensusConfig, "outlier_cutoff_upper"),
        ],
    )
    def test_non_finite_floats_rejected(
        self, model_cls, field_name, bad_float
    ):
        with pytest.raises(ValidationError):
            model_cls.model_validate({field_name: bad_float})

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

    def test_output_path_properties(self, tmp_path):
        io_cfg = IOConfig(
            input_filenames=("in.mgf",),
            output_filename=tmp_path / "run_v1.2",
        )
        assert io_cfg.csv_path == tmp_path / "run_v1.2.csv"
        assert io_cfg.mgf_path == tmp_path / "run_v1.2.mgf"

    def test_partial_io_config_accepts_optional_paths(self):
        partial = PartialIOConfig.model_validate(
            {
                "input_filenames": ["a.mgf"],
                "output_filename": "prefix",
                "work_dir": Path("work"),
            }
        )
        assert partial.input_filenames == ("a.mgf",)
        assert partial.output_filename == Path("prefix")
        assert partial.work_dir == Path("work")

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
            {"input_filenames": ["a.mgf"], "output_filename": Path("   ")},
            {"input_filenames": ["a.mgf"], "output_filename": Path(".")},
            {"input_filenames": ["a.mgf"], "output_filename": Path("..")},
            {"input_filenames": ["a.mgf"], "output_filename": Path("/")},
            {"input_filenames": ["a.mgf"], "output_filename": 123},
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
            {
                "input_filenames": ["a.mgf"],
                "output_filename": "out",
                "work_dir": Path("   "),
            },
        ],
    )
    def test_invalid_io_paths_rejected(self, payload):
        with pytest.raises(ValidationError):
            IOConfig.model_validate(payload)


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
            [],
            [[]],
            [[2, 2]],
            [["unknown", "unknown"]],
            [[1, 2], [2, 3]],
            [[1, "unknown"], ["unknown"]],
            [[1], "other", "other"],
            [["other"]],
            [[True]],
            [[False]],
            [[-129]],
            [[128]],
            [[1.5]],
            [["abc"]],
            "12",
            [1],
        ],
    )
    def test_invalid_charge_buckets_rejected(self, buckets):
        with pytest.raises(ValidationError):
            ClusteringConfig.model_validate(
                {"precursor_charge_buckets": buckets}
            )


class TestCrossFieldInvariants:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_mz": 500.0, "max_mz": 500.0}, "min_mz"),
            ({"min_mz": 600.0, "max_mz": 500.0}, "min_mz"),
            (
                {"min_mz": 100.0, "max_mz": 200.0, "min_mz_range": 150.0},
                "min_mz_range",
            ),
            ({"min_peaks": 10, "max_peaks_used": 5}, "max_peaks_used"),
        ],
    )
    def test_preprocessing_cross_field_checks(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            PreprocessingConfig(**kwargs)

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

    @pytest.mark.parametrize(
        ("fragment_tol", "match"),
        [
            (1e-320, "fragment_tol"),
            (1e-8, "bin"),
        ],
    )
    def test_cross_section_fragment_tol_bin_overflow_rejected(
        self, fragment_tol, match
    ):
        with pytest.raises(ValidationError, match=match):
            FalconSettings(
                preprocessing=PreprocessingConfig(min_mz=0.0, max_mz=1500.0),
                clustering=ClusteringConfig(fragment_tol=fragment_tol),
            )
