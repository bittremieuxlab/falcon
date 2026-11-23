"""Unit tests for falcon.config."""

import pytest

from falcon.config import Config, NewlineTextHelpFormatter


def _parse(args):
    """Parse CLI arguments with a fresh Config instance."""
    cfg = Config()
    cfg.parse(args)
    return cfg


# ----------------------------------------------------------------------
# Defaults, CLI overrides, and INI configuration files
# ----------------------------------------------------------------------


class TestConfigParsing:
    def test_defaults(self):
        cfg = _parse(["in.mgf", "out"])
        assert cfg.input_filenames == ["in.mgf"]
        assert cfg.output_filename == "out"
        assert cfg.work_dir is None
        assert cfg.overwrite is False
        assert cfg.export_representatives is False
        assert cfg.precursor_tol == [20.0, "ppm"]
        assert cfg.rt_tol is None
        assert cfg.fragment_tol == 0.05
        assert cfg.linkage == "complete"
        assert cfg.distance_threshold == 0.1
        assert cfg.min_matched_peaks == 0
        assert cfg.consensus_method == "medoid"
        assert cfg.outlier_cutoff_lower == 1.5
        assert cfg.outlier_cutoff_upper == 1.5
        assert cfg.batch_size == 2**15
        assert cfg.precursor_charge_buckets is None
        assert cfg.min_peaks == 5
        assert cfg.min_mz_range == 250.0
        assert cfg.min_mz == 101.0
        assert cfg.max_mz == 1500.0
        assert cfg.remove_precursor_tol == 1.5
        assert cfg.min_intensity == 0.01
        assert cfg.max_peaks_used == 50
        assert cfg.scaling == "off"

    def test_multiple_input_files_and_flags(self):
        cfg = _parse(
            [
                "a.mgf",
                "b.mzml",
                "results",
                "--work_dir",
                "custom_work_dir",
                "--overwrite",
                "--export_representatives",
                "--min_matched_peaks",
                "6",
                "--batch_size",
                "128",
                "--rt_tol",
                "30.5",
                "--outlier_cutoff_lower",
                "2.0",
                "--outlier_cutoff_upper",
                "3.5",
            ]
        )
        assert cfg.input_filenames == ["a.mgf", "b.mzml"]
        assert cfg.output_filename == "results"
        assert cfg.work_dir == "custom_work_dir"
        assert cfg.overwrite is True
        assert cfg.export_representatives is True
        assert cfg.min_matched_peaks == 6
        assert cfg.batch_size == 128
        assert cfg.rt_tol == 30.5
        assert cfg.outlier_cutoff_lower == 2.0
        assert cfg.outlier_cutoff_upper == 3.5

    @pytest.mark.parametrize(
        ("tol_val", "tol_mode", "expected_val"),
        [
            ("20", "ppm", 20.0),
            ("0.5", "Da", 0.5),
            ("0", "Da", 0.0),
        ],
    )
    def test_precursor_tol_valid(self, tol_val, tol_mode, expected_val):
        cfg = _parse(["in.mgf", "out", "--precursor_tol", tol_val, tol_mode])
        assert cfg.precursor_tol == [expected_val, tol_mode]

    @pytest.mark.parametrize("scaling", ["off", "root", "log", "rank"])
    def test_scaling_choices(self, scaling):
        cfg = _parse(["in.mgf", "out", "--scaling", scaling])
        assert cfg.scaling == scaling

    @pytest.mark.parametrize("linkage", ["single", "complete", "average"])
    def test_linkage_choices(self, linkage):
        cfg = _parse(["in.mgf", "out", "--linkage", linkage])
        assert cfg.linkage == linkage

    @pytest.mark.parametrize("method", ["medoid", "average"])
    def test_consensus_method_choices(self, method):
        cfg = _parse(["in.mgf", "out", "--consensus_method", method])
        assert cfg.consensus_method == method

    def test_config_file_loading_and_cli_precedence(self, tmp_path):
        cfg_file = tmp_path / "config.ini"
        cfg_file.write_text(
            "fragment_tol = 0.1\nmin_peaks = 9\nlinkage = average\n"
        )
        cfg = _parse(
            [
                "in.mgf",
                "out",
                "-c",
                str(cfg_file),
                "--fragment_tol",
                "0.03",
            ]
        )
        # CLI overrides INI; unspecified INI values override defaults.
        assert cfg.fragment_tol == 0.03
        assert cfg.min_peaks == 9
        assert cfg.linkage == "average"

    def test_help_formatter_preserves_newlines(self):
        formatter = NewlineTextHelpFormatter("falcon")
        text = "Line one\n\nLine two"
        formatted = formatter._fill_text(text, width=80, indent="")
        assert formatted == "Line one\n\nLine two"


# ----------------------------------------------------------------------
# Charge bucket parsing and validation
# ----------------------------------------------------------------------


class TestChargeBucketsParsing:
    def test_explicit_buckets_with_unknown_and_other(self):
        cfg = _parse(
            [
                "in.mgf",
                "out",
                "--precursor_charge_buckets",
                "[1]",
                "[2, 3]",
                "[unknown]",
                "other",
            ]
        )
        assert cfg.precursor_charge_buckets == [
            {1},
            {2, 3},
            {"unknown"},
            "other",
        ]

    def test_unbracketed_whitespace_and_mixed_unknown(self):
        cfg = _parse(
            [
                "in.mgf",
                "out",
                "--precursor_charge_buckets",
                " 1, unknown ",
                "[ -2 , , -1 ]",
            ]
        )
        assert cfg.precursor_charge_buckets == [
            {1, "unknown"},
            {-2, -1},
        ]

    @pytest.mark.parametrize(
        "bucket_args",
        [
            ["[1, 2]", "[2, 3]"],  # overlapping int charge
            ["[1, unknown]", "[unknown]"],  # overlapping unknown
            ["[1]", "other", "other"],  # duplicate catch-all 'other'
            ["[]"],  # empty bracketed bucket
            ["[ , ]"],  # only whitespace/commas
            [""],  # empty string
            ["[abc]"],  # non-integer token
            ["[1.5]"],  # float token
            ["[other]"],  # 'other' inside brackets
        ],
    )
    def test_invalid_charge_buckets_rejected(self, bucket_args):
        with pytest.raises(SystemExit) as exc_info:
            _parse(
                [
                    "in.mgf",
                    "out",
                    "--precursor_charge_buckets",
                    *bucket_args,
                ]
            )
        assert exc_info.value.code == 2


# ----------------------------------------------------------------------
# Parameter validation
# ----------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize(
        "bad_args",
        [
            ["--precursor_tol", "20", "mz"],  # unknown unit
            ["--precursor_tol", "20", "PPM"],  # case-sensitive mode
            ["--precursor_tol", "20", "da"],  # case-sensitive mode
            ["--precursor_tol", "abc", "ppm"],  # non-numeric mass
            ["--precursor_tol", "-1", "Da"],  # negative mass
            ["--precursor_tol", "nan", "ppm"],  # NaN mass
            ["--distance_threshold", "0"],  # zero threshold
            ["--distance_threshold", "-0.1"],  # negative threshold
            ["--distance_threshold", "nan"],  # NaN threshold
            ["--batch_size", "0"],  # zero batch size
            ["--batch_size", "-10"],  # negative batch size
            ["--linkage", "ward"],  # unsupported linkage
            ["--consensus_method", "median"],  # unsupported consensus
            ["--scaling", "zscore"],  # unsupported scaling
        ],
    )
    def test_invalid_cli_options_fail_fast(self, bad_args):
        with pytest.raises(SystemExit) as exc_info:
            _parse(["in.mgf", "out", *bad_args])
        assert exc_info.value.code == 2

    def test_validate_before_parse_raises(self):
        cfg = Config()
        with pytest.raises(RuntimeError, match="not been initialized"):
            cfg._validate()


# ----------------------------------------------------------------------
# Attribute and mapping accessors
# ----------------------------------------------------------------------


class TestConfigAccessors:
    def test_access_before_parse_raises(self):
        cfg = Config()
        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = cfg.fragment_tol
        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = cfg["fragment_tol"]

    def test_getattr_getitem_and_setattr_sync(self):
        cfg = _parse(["in.mgf", "out"])
        assert cfg["fragment_tol"] == cfg.fragment_tol == 0.05
        cfg.fragment_tol = 0.02
        assert cfg["fragment_tol"] == 0.02
        assert cfg.fragment_tol == 0.02

    def test_missing_option_raises_standard_errors(self):
        cfg = _parse(["in.mgf", "out"])
        assert not hasattr(cfg, "nonexistent_option")
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_option
        with pytest.raises(KeyError):
            _ = cfg["nonexistent_option"]
