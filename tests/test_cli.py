"""Unit tests for falcon.cli argument parsing and entry point."""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon import __version__, pipeline
from falcon.cli import (
    DumpSchemaAction,
    PrintConfigAction,
    RunAction,
    build_parser,
    format_validation_error,
    parse_cli,
)
from falcon.cli import (
    main as falcon_main,
)
from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    FalconConfig,
    FalconSettings,
    MedoidConsensusConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    load_config,
)


@pytest.fixture
def forbid_pipeline_run(monkeypatch):
    """Fail the test if ``falcon.pipeline.run`` is invoked."""

    def _fail_run(_cfg):
        raise AssertionError("pipeline.run() must not execute")

    monkeypatch.setattr(pipeline, "run", _fail_run)


class TestCliPositionalAndIO:
    def test_positional_inputs_and_dotted_output_prefix(self):
        result = parse_cli(["a.mgf", "b.mzml", "dir/run_v1.2"])
        assert isinstance(result, RunAction)
        assert result.config.io.input_filenames == ("a.mgf", "b.mzml")
        assert result.config.io.output_filename == Path("dir/run_v1.2")
        assert result.save_config is None

    def test_single_positional_argument_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["only_one.mgf"])
        assert exc_info.value.code == 2
        assert "output" in capsys.readouterr().err.lower()

    def test_zero_positionals_with_full_run_toml_and_output_override(
        self, tmp_path
    ):
        run_toml = tmp_path / "run.toml"
        run_toml.write_text(
            "[io]\n"
            'input_filenames = ["a.mgf"]\n'
            'output_filename = "out1"\n'
            "overwrite = true\n"
            "export_representatives = true\n",
            encoding="utf-8",
        )
        res = parse_cli(["-c", str(run_toml)])
        assert isinstance(res, RunAction)
        assert res.config.io.input_filenames == ("a.mgf",)
        assert res.config.io.output_filename == Path("out1")
        assert res.config.io.overwrite is True
        assert res.config.io.export_representatives is True

        # Named -o overrides only output_filename; --no-* unsets flags.
        res2 = parse_cli(
            [
                "-c",
                str(run_toml),
                "-o",
                "out_v2",
                "--no-overwrite",
                "--no-export-representatives",
            ]
        )
        assert isinstance(res2, RunAction)
        assert res2.config.io.input_filenames == ("a.mgf",)
        assert res2.config.io.output_filename == Path("out_v2")
        assert res2.config.io.overwrite is False
        assert res2.config.io.export_representatives is False

    def test_named_input_and_output_flags_and_mixing_rejection(self, capsys):
        res = parse_cli(["-i", "x.mgf", "y.mzml", "-o", "res"])
        assert isinstance(res, RunAction)
        assert res.config.io.input_filenames == ("x.mgf", "y.mzml")
        assert res.config.io.output_filename == Path("res")

        for argv in (
            ["a.mgf", "out", "-o", "other"],
            ["a.mgf", "out", "-i", "b.mgf"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                parse_cli(argv)
            assert exc_info.value.code == 2
            assert "positional" in capsys.readouterr().err.lower()


class TestCliLayeringAndScientificOptions:
    def test_repeatable_configs_and_cli_overrides(self, tmp_path):
        base = tmp_path / "base.toml"
        base.write_text(
            "[io]\n"
            'work_dir = "/some/work"\n'
            "[preprocessing]\n"
            "min_peaks = 7\n"
            "min_mz = 50.0\n"
            "max_mz = 1000.0\n"
            "min_mz_range = 150.0\n"
            "remove_precursor_tol = 2.0\n"
            "max_peaks_used = 40\n"
            'scaling = "root"\n'
            "[clustering]\n"
            "rt_tol = 20.0\n"
            "fragment_tol = 0.02\n"
            'linkage = "average"\n'
            "distance_threshold = 0.2\n"
            "min_matched_peaks = 6\n"
            "batch_size = 512\n"
            "precursor_charge_buckets = [[1], [2]]\n"
            "[consensus]\n"
            'method = "average"\n'
            "outlier_cutoff_lower = 2.0\n"
            "outlier_cutoff_upper = 3.0\n",
            encoding="utf-8",
        )
        overlay = tmp_path / "overlay.toml"
        overlay.write_text(
            "[clustering]\ndistance_threshold = 0.15\n",
            encoding="utf-8",
        )

        res = parse_cli(["in.mgf", "out", "-c", str(base), "-c", str(overlay)])
        assert isinstance(res, RunAction)
        assert res.config.preprocessing.min_peaks == 7
        assert res.config.clustering.distance_threshold == 0.15
        assert res.config.clustering.rt_tol == 20.0
        assert res.config.consensus == AverageConsensusConfig(
            outlier_cutoff_lower=2.0, outlier_cutoff_upper=3.0
        )

        # CLI flags override and unset nullable fields via sentinels.
        res2 = parse_cli(
            [
                "in.mgf",
                "out",
                "-c",
                str(base),
                "--work-dir",
                "none",
                "--rt-tol",
                "none",
                "--remove-precursor-tol",
                "none",
                "--max-peaks-used",
                "none",
                "--precursor-charge-buckets",
                "auto",
                "--precursor-tol",
                "10",
                "ppm",
                "--min-peaks",
                "3",
                "--min-mz",
                "10.0",
                "--max-mz",
                "900.0",
                "--min-mz-range",
                "100.0",
                "--min-intensity",
                "0.05",
                "--scaling",
                "log",
                "--fragment-tol",
                "0.04",
                "--linkage",
                "single",
                "--distance-threshold",
                "0.25",
                "--min-matched-peaks",
                "4",
                "--batch-size",
                "256",
                "--outlier-cutoff-upper",
                "2.5",
            ]
        )
        assert isinstance(res2, RunAction)
        cfg = res2.config
        assert cfg.io.work_dir is None
        assert cfg.clustering.rt_tol is None
        assert cfg.preprocessing.remove_precursor_tol is None
        assert cfg.preprocessing.max_peaks_used is None
        assert cfg.clustering.precursor_charge_buckets is None
        assert cfg.clustering.precursor_tol == PrecursorTolerance(
            mass=10.0, mode="ppm"
        )
        assert cfg.preprocessing.min_peaks == 3
        assert cfg.preprocessing.min_mz == 10.0
        assert cfg.preprocessing.max_mz == 900.0
        assert cfg.preprocessing.min_mz_range == 100.0
        assert cfg.preprocessing.min_intensity == 0.05
        assert cfg.preprocessing.scaling == "log"
        assert cfg.clustering.fragment_tol == 0.04
        assert cfg.clustering.linkage == "single"
        assert cfg.clustering.distance_threshold == 0.25
        assert cfg.clustering.min_matched_peaks == 4
        assert cfg.clustering.batch_size == 256
        assert cfg.consensus == AverageConsensusConfig(
            outlier_cutoff_lower=2.0, outlier_cutoff_upper=2.5
        )

    def test_charge_buckets_cli_parsing_including_negative_charges(self):
        res = parse_cli(
            [
                "in.mgf",
                "out",
                "--precursor-charge-buckets",
                "[-2, -1]",
                "[1]",
                "unknown",
                "other",
            ]
        )
        assert isinstance(res, RunAction)
        assert res.config.clustering.precursor_charge_buckets == (
            (-2, -1),
            (1,),
            ("unknown",),
            "other",
        )

    @pytest.mark.parametrize(
        ("tokens", "expected_err"),
        [
            (["auto", "[1]"], "auto"),
            (["[1, 2"], "unbalanced"),
            (["1, 2]"], "unbalanced"),
            (["[]"], "empty charge bucket"),
            (["[2,,3]"], "empty"),
            (["[abc]"], "invalid charge value"),
        ],
    )
    def test_charge_buckets_cli_errors_exit_2(
        self, tokens, expected_err, capsys
    ):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "--precursor-charge-buckets", *tokens])
        assert exc_info.value.code == 2
        assert expected_err in capsys.readouterr().err.lower()

    def test_consensus_switch_to_medoid_and_same_layer_conflict(
        self, tmp_path, capsys
    ):
        avg_toml = tmp_path / "avg.toml"
        avg_toml.write_text(
            '[consensus]\nmethod = "average"\n'
            "outlier_cutoff_lower = 2.0\noutlier_cutoff_upper = 3.0\n",
            encoding="utf-8",
        )
        res = parse_cli(
            [
                "in.mgf",
                "out",
                "-c",
                str(avg_toml),
                "--consensus-method",
                "medoid",
            ]
        )
        assert isinstance(res, RunAction)
        assert res.config.consensus == MedoidConsensusConfig()

        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "--outlier-cutoff-lower", "2.0"])
        assert exc_info.value.code == 2
        assert "outlier_cutoff_lower" in capsys.readouterr().err


class TestCliUtilityActionsAndErrorFormatting:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["--version"])
        assert exc_info.value.code == 0
        assert str(__version__) in capsys.readouterr().out

    def test_print_config_without_io_paths(self, tmp_path):
        overlay = tmp_path / "m.toml"
        overlay.write_text(
            "[clustering]\nmin_matched_peaks = 6\n", encoding="utf-8"
        )
        res = parse_cli(["-c", str(overlay), "--print-config"])
        assert isinstance(res, PrintConfigAction)
        assert isinstance(res.settings, FalconSettings)
        assert res.settings.clustering.min_matched_peaks == 6

    def test_dump_schema_and_save_config_actions(self, tmp_path):
        assert isinstance(parse_cli(["--dump-schema"]), DumpSchemaAction)

        dest = tmp_path / "saved.toml"
        res = parse_cli(["in.mgf", "out", "--save-config", str(dest)])
        assert isinstance(res, RunAction)
        assert res.save_config == dest

    @pytest.mark.parametrize(
        "argv",
        [
            ["--print-config", "--dump-schema"],
            ["--print-config", "--save-config", "s.toml"],
            ["--dump-schema", "--save-config", "s.toml"],
        ],
    )
    def test_mutually_exclusive_utility_flags_rejected_by_argparse(
        self, argv, capsys
    ):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(argv)
        assert exc_info.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("argv", "expected_err"),
        [
            (
                ["in.mgf", "out", "--distance-threshold", "-0.1"],
                "clustering.distance_threshold",
            ),
            (["in.mgf", "out", "--min-peaks", "abc"], "expected an integer"),
            (["in.mgf", "out", "--min-mz", "abc"], "expected a number"),
            (["in.mgf", "out", "--rt-tol", "abc"], "expected a number"),
            (
                ["in.mgf", "out", "--max-peaks-used", "abc"],
                "expected an integer",
            ),
            (
                ["in.mgf", "out", "--precursor-tol", "abc", "ppm"],
                "must be a number",
            ),
        ],
    )
    def test_invalid_cli_flag_values_exit_2(self, argv, expected_err, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(argv)
        assert exc_info.value.code == 2
        assert expected_err in capsys.readouterr().err

    def test_config_file_errors_exit_2_with_diagnostics(
        self, tmp_path, capsys
    ):
        missing = tmp_path / "nope.toml"
        bad = tmp_path / "bad.toml"
        bad.write_text("[clustering\n", encoding="utf-8")
        binary = tmp_path / "bad_bytes.toml"
        binary.write_bytes(b"\xff\xfe[clustering]\n")

        cases = [
            (missing, "nope.toml"),
            (tmp_path, "directory"),
            (bad, "bad.toml"),
            (binary, "bad_bytes.toml"),
        ]
        for path, expected in cases:
            with pytest.raises(SystemExit) as exc_info:
                parse_cli(["in.mgf", "out", "-c", str(path)])
            assert exc_info.value.code == 2
            assert expected in capsys.readouterr().err.lower()

    def test_format_validation_error_normalizes_paths_and_messages(self):
        with pytest.raises(ValidationError) as exc_union:
            FalconConfig.model_validate(
                {
                    "io": {
                        "input_filenames": ["a.mgf"],
                        "output_filename": "o",
                    },
                    "consensus": {
                        "method": "medoid",
                        "outlier_cutoff_lower": 1.0,
                    },
                }
            )
        formatted = format_validation_error(exc_union.value)
        assert "consensus.outlier_cutoff_lower" in formatted
        assert "MedoidConsensusConfig" not in formatted

        with pytest.raises(ValidationError) as exc_val:
            FalconSettings.model_validate(
                {"preprocessing": {"min_mz": 500.0, "max_mz": 100.0}}
            )
        formatted_val = format_validation_error(exc_val.value)
        assert "Value error, " not in formatted_val
        assert "min_mz" in formatted_val

        # Root-level cross-section error maps empty loc to "config".
        with pytest.raises(ValidationError) as exc_root:
            FalconSettings(
                preprocessing=PreprocessingConfig(max_peaks_used=5),
                clustering=ClusteringConfig(min_matched_peaks=6),
            )
        formatted_root = format_validation_error(exc_root.value)
        assert "  - config: " in formatted_root

    def test_build_parser_has_section_groups(self):
        help_text = build_parser().format_help()
        for heading in (
            "I/O",
            "Configuration",
            "Preprocessing",
            "Clustering",
            "Consensus",
        ):
            assert heading in help_text


class TestCliMain:
    def test_main_dump_schema_and_print_config(
        self, capsys, forbid_pipeline_run
    ):
        assert falcon_main(["--dump-schema"]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert "extends" in schema["properties"]

        assert falcon_main(["--print-config"]) == 0
        parsed = tomllib.loads(capsys.readouterr().out)
        assert parsed["consensus"]["method"] == "medoid"

    def test_main_runs_pipeline_with_and_without_save_config(
        self, tmp_path, monkeypatch
    ):
        seen: list[FalconConfig] = []

        def _record_run(cfg):
            seen.append(cfg)
            return 0

        monkeypatch.setattr(pipeline, "run", _record_run)
        assert falcon_main(["in.mgf", "out"]) == 0
        assert len(seen) == 1
        assert seen[0].io.input_filenames == ("in.mgf",)

        dest = tmp_path / "saved.toml"
        rc = falcon_main(["in.mgf", "out", "--save-config", str(dest)])

        assert rc == 0
        assert len(seen) == 2
        assert dest.is_file()
        assert load_config(config_files=[dest]) == seen[1]

    def test_main_returns_2_on_save_config_errors_without_running(
        self, tmp_path, capsys, forbid_pipeline_run
    ):
        # Directory destination rejected via write_saved_config in main.
        rc_dir = falcon_main(["in.mgf", "out", "--save-config", str(tmp_path)])
        assert rc_dir == 2
        assert "directory" in capsys.readouterr().err.lower()

        # Unwritable parent directory.
        unwritable = tmp_path / "no_such_dir" / "saved.toml"
        rc_write = falcon_main(
            ["in.mgf", "out", "--save-config", str(unwritable)]
        )
        assert rc_write == 2
        assert "Failed to write configuration" in capsys.readouterr().err

        # Input alias rejected before touching input or running.
        input_file = tmp_path / "sample.mgf"
        input_file.write_text("MGF\n", encoding="utf-8")
        rc_alias = falcon_main(
            [
                str(input_file),
                str(tmp_path / "out"),
                "--save-config",
                str(input_file),
            ]
        )
        assert rc_alias == 2
        assert "aliases input file" in capsys.readouterr().err
        assert input_file.read_text(encoding="utf-8") == "MGF\n"
