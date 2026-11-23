"""Unit tests for falcon.cli."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon import __version__
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
    ConfigError,
    FalconConfig,
    FalconSettings,
    MedoidConsensusConfig,
    PrecursorTolerance,
    load_config,
    write_saved_config,
)


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
        assert res.config is not None
        assert res.config.io.input_filenames == ("a.mgf",)
        assert res.config.io.output_filename == Path("out1")
        assert res.config.io.overwrite is True
        assert res.config.io.export_representatives is True

        # Named -o overrides only output_filename; --no-* disables booleans.
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
        assert res2.config is not None
        assert res2.config.io.input_filenames == ("a.mgf",)
        assert res2.config.io.output_filename == Path("out_v2")
        assert res2.config.io.overwrite is False
        assert res2.config.io.export_representatives is False

    def test_named_input_and_output_flags_and_mixing_rejection(self, capsys):
        res = parse_cli(["-i", "x.mgf", "y.mzml", "-o", "res"])
        assert res.config is not None
        assert res.config.io.input_filenames == ("x.mgf", "y.mzml")
        assert res.config.io.output_filename == Path("res")

        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["a.mgf", "out", "-o", "other"])
        assert exc_info.value.code == 2
        assert "positional" in capsys.readouterr().err.lower()

        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["a.mgf", "out", "-i", "b.mgf"])
        assert exc_info.value.code == 2


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

        # Omitted CLI flags preserve values from base and overlay.
        res = parse_cli(["in.mgf", "out", "-c", str(base), "-c", str(overlay)])
        assert res.config is not None
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
        cfg = res2.config
        assert cfg is not None
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

    def test_charge_buckets_cli_parsing_including_negative_charges(
        self, capsys
    ):
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
        assert res.config is not None
        assert res.config.clustering.precursor_charge_buckets == (
            (-2, -1),
            (1,),
            ("unknown",),
            "other",
        )

        # Reject empty token between commas.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(
                ["in.mgf", "out", "--precursor-charge-buckets", "[2,,3]"]
            )
        assert exc_info.value.code == 2
        assert "empty" in capsys.readouterr().err.lower()

        # Reject mixing 'auto' with other buckets.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(
                ["in.mgf", "out", "--precursor-charge-buckets", "auto", "[1]"]
            )
        assert exc_info.value.code == 2

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
        assert res.config is not None
        assert res.config.consensus == MedoidConsensusConfig()

        # Default medoid + --outlier-cutoff-lower fails cleanly with exit 2.
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

    def test_dump_schema_action(self):
        res = parse_cli(["--dump-schema"])
        assert isinstance(res, DumpSchemaAction)

    def test_mutually_exclusive_utility_flags_rejected(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["--print-config", "--dump-schema"])
        assert exc_info.value.code == 2
        assert capsys.readouterr().err

        with pytest.raises(SystemExit) as exc_info:
            parse_cli(
                ["--print-config", "--save-config", str(tmp_path / "s.toml")]
            )
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            parse_cli(
                ["--dump-schema", "--save-config", str(tmp_path / "s.toml")]
            )
        assert exc_info.value.code == 2

    def test_save_config_and_directory_rejection(self, tmp_path, capsys):
        dest = tmp_path / "saved.toml"
        res = parse_cli(["in.mgf", "out", "--save-config", str(dest)])
        assert isinstance(res, RunAction)
        assert res.save_config == dest
        write_saved_config(res.config, dest)
        assert load_config(config_files=[dest]) == res.config

        # Directory destination fails fast in parse_cli AND write_saved_config.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "--save-config", str(tmp_path)])
        assert exc_info.value.code == 2
        assert "directory" in capsys.readouterr().err.lower()

        with pytest.raises(ConfigError, match="directory"):
            write_saved_config(res.config, tmp_path)

    def test_validation_and_config_errors_exit_2_with_dotted_paths(
        self, tmp_path, capsys
    ):
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "--distance-threshold", "-0.1"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "clustering.distance_threshold" in err

        # Non-numeric precursor tolerance mass exits 2 cleanly.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "--precursor-tol", "abc", "ppm"])
        assert exc_info.value.code == 2

        # Missing config file exits 2 with clean error message.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "-c", str(tmp_path / "nope.toml")])
        assert exc_info.value.code == 2
        assert "nope.toml" in capsys.readouterr().err

        # Directory passed to -c exits 2 cleanly.
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "-c", str(tmp_path)])
        assert exc_info.value.code == 2
        assert "directory" in capsys.readouterr().err.lower()

        # Malformed TOML exits 2 cleanly.
        bad = tmp_path / "bad.toml"
        bad.write_text("[clustering\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "-c", str(bad)])
        assert exc_info.value.code == 2
        assert "bad.toml" in capsys.readouterr().err

    def test_format_validation_error_strips_internal_union_branch_names(self):
        with pytest.raises(ValidationError) as exc_info:
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
        formatted = format_validation_error(exc_info.value)
        assert "consensus.outlier_cutoff_lower" in formatted
        assert "MedoidConsensusConfig" not in formatted

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

    def test_non_utf8_config_file_exits_2_on_cli(self, tmp_path, capsys):
        binary = tmp_path / "bad_bytes.toml"
        binary.write_bytes(b"\xff\xfe[clustering]\n")
        with pytest.raises(SystemExit) as exc_info:
            parse_cli(["in.mgf", "out", "-c", str(binary)])
        assert exc_info.value.code == 2
        assert "bad_bytes.toml" in capsys.readouterr().err

    def test_format_validation_error_strips_value_error_prefix(self):
        with pytest.raises(ValidationError) as exc_info:
            FalconSettings.model_validate(
                {"preprocessing": {"min_mz": 500.0, "max_mz": 100.0}}
            )
        formatted = format_validation_error(exc_info.value)
        assert "Value error, " not in formatted
        assert "min_mz" in formatted

    def test_main_returns_2_when_save_config_unwritable(
        self, tmp_path, capsys
    ):
        dest = tmp_path / "no_such_dir" / "saved.toml"
        rc = falcon_main(["in.mgf", "out", "--save-config", str(dest)])
        assert rc == 2
        assert "Failed to write configuration" in capsys.readouterr().err


class TestSaveConfigInputAliasProtection:
    def test_main_rejects_direct_and_relative_input_alias_before_writing(
        self, tmp_path, capsys, monkeypatch
    ):
        from falcon import pipeline

        input_file = tmp_path / "sample.mgf"
        original = "BEGIN IONS\nPEPMASS=500.0\nEND IONS\n"
        input_file.write_text(original, encoding="utf-8")

        def _fail_run(_cfg):
            raise AssertionError("run() must not execute on alias error")

        monkeypatch.setattr(pipeline, "run", _fail_run)

        # Direct path alias.
        rc = falcon_main(
            [
                str(input_file),
                str(tmp_path / "out"),
                "--save-config",
                str(input_file),
            ]
        )
        assert rc == 2
        assert "aliases input file" in capsys.readouterr().err
        assert input_file.read_text(encoding="utf-8") == original

        # Non-normalized relative alias pointing to the same file.
        sub = tmp_path / "sub"
        sub.mkdir()
        rel_alias = sub / ".." / "sample.mgf"
        rc2 = falcon_main(
            [
                str(input_file),
                str(tmp_path / "out"),
                "--save-config",
                str(rel_alias),
            ]
        )
        assert rc2 == 2
        assert "aliases input file" in capsys.readouterr().err
        assert input_file.read_text(encoding="utf-8") == original

    def test_main_rejects_glob_selected_input_alias_before_writing(
        self, tmp_path, capsys, monkeypatch
    ):
        from falcon import pipeline

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        spec_a = data_dir / "a.mgf"
        spec_b = data_dir / "b.mgf"
        spec_a.write_text("MGF_A\n", encoding="utf-8")
        spec_b.write_text("MGF_B\n", encoding="utf-8")

        def _fail_run(_cfg):
            raise AssertionError("run() must not execute on alias error")

        monkeypatch.setattr(pipeline, "run", _fail_run)

        # CLI glob selecting b.mgf.
        rc = falcon_main(
            [
                str(data_dir / "*.mgf"),
                str(tmp_path / "out"),
                "--save-config",
                str(spec_b),
            ]
        )
        assert rc == 2
        assert "aliases input file" in capsys.readouterr().err
        assert spec_b.read_text(encoding="utf-8") == "MGF_B\n"

        # Glob declared inside a TOML config file.
        run_toml = tmp_path / "run.toml"
        run_toml.write_text(
            "[io]\n"
            f'input_filenames = ["{data_dir / "*.mgf"}"]\n'
            f'output_filename = "{tmp_path / "out"}"\n',
            encoding="utf-8",
        )
        rc2 = falcon_main(["-c", str(run_toml), "--save-config", str(spec_a)])
        assert rc2 == 2
        assert "aliases input file" in capsys.readouterr().err
        assert spec_a.read_text(encoding="utf-8") == "MGF_A\n"

    def test_rejects_symlink_and_hardlink_aliases_before_writing(
        self, tmp_path, capsys, monkeypatch
    ):
        import os

        from falcon import pipeline

        real_mgf = tmp_path / "real.mgf"
        real_mgf.write_text("REAL_MGF\n", encoding="utf-8")
        sym_dest = tmp_path / "sym_config.toml"
        sym_dest.symlink_to(real_mgf)
        hard_dest = tmp_path / "hard_config.toml"
        os.link(real_mgf, hard_dest)

        def _fail_run(_cfg):
            raise AssertionError("run() must not execute on alias error")

        monkeypatch.setattr(pipeline, "run", _fail_run)

        for dest in (sym_dest, hard_dest):
            rc = falcon_main(
                [
                    str(real_mgf),
                    str(tmp_path / "out"),
                    "--save-config",
                    str(dest),
                ]
            )
            assert rc == 2
            assert "aliases input file" in capsys.readouterr().err
            assert real_mgf.read_text(encoding="utf-8") == "REAL_MGF\n"

        # Reverse direction: symlinked input, real path as save target.
        sym_in = tmp_path / "in_link.mgf"
        sym_in.symlink_to(real_mgf)
        rc_rev = falcon_main(
            [
                str(sym_in),
                str(tmp_path / "out"),
                "--save-config",
                str(real_mgf),
            ]
        )
        assert rc_rev == 2
        assert "aliases input file" in capsys.readouterr().err
        assert real_mgf.read_text(encoding="utf-8") == "REAL_MGF\n"

    def test_rejects_nonexistent_literal_input_and_symlink_loop(
        self, tmp_path, capsys
    ):
        missing = tmp_path / "missing.mgf"
        rc = falcon_main(
            [
                str(missing),
                str(tmp_path / "out"),
                "--save-config",
                str(missing),
            ]
        )
        assert rc == 2
        assert "aliases input file" in capsys.readouterr().err
        assert not missing.exists()

        loop = tmp_path / "loop.toml"
        loop.symlink_to(loop)
        rc_loop = falcon_main(
            [
                str(tmp_path / "other.mgf"),
                str(tmp_path / "out"),
                "--save-config",
                str(loop),
            ]
        )
        assert rc_loop == 2
        assert (
            "Failed to resolve --save-config path" in capsys.readouterr().err
        )

    def test_distinct_destinations_remain_valid_with_glob_inputs(
        self, tmp_path, monkeypatch
    ):
        from falcon import pipeline

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.mgf").write_text("MGF_A\n", encoding="utf-8")
        # Distinct new destination in the same directory as the glob.
        new_dest = data_dir / "saved.toml"
        # Distinct existing TOML file may still be updated.
        existing_dest = tmp_path / "existing.toml"
        existing_dest.write_text("# old\n", encoding="utf-8")

        monkeypatch.setattr(pipeline, "run", lambda _cfg: 0)

        for dest in (new_dest, existing_dest):
            rc = falcon_main(
                [
                    str(data_dir / "*.mgf"),
                    str(tmp_path / "out"),
                    "--save-config",
                    str(dest),
                ]
            )
            assert rc == 0
            assert dest.is_file()
            loaded = load_config(config_files=[dest])
            assert loaded.io.input_filenames == (str(data_dir / "*.mgf"),)
        assert (data_dir / "a.mgf").read_text(encoding="utf-8") == "MGF_A\n"

    def test_unmatched_glob_and_broken_input_symlink_allow_distinct_save(
        self, tmp_path
    ):
        # Unmatched glob does not block saving to a distinct file.
        dest = tmp_path / "saved.toml"
        cfg = load_config(
            overrides={
                "io": {
                    "input_filenames": [str(tmp_path / "none_*.mgf")],
                    "output_filename": str(tmp_path / "out"),
                }
            }
        )
        write_saved_config(cfg, dest)
        assert dest.is_file()

        # A symlink-loop input does not crash saving elsewhere.
        bad_input = tmp_path / "bad_input.mgf"
        bad_input.symlink_to(bad_input)
        dest2 = tmp_path / "saved2.toml"
        cfg2 = load_config(
            overrides={
                "io": {
                    "input_filenames": [str(bad_input)],
                    "output_filename": str(tmp_path / "out"),
                }
            }
        )
        write_saved_config(cfg2, dest2)
        assert dest2.is_file()

        # Direct write_saved_config rejects dangling symlink to input.
        missing_in = tmp_path / "future.mgf"
        dangling = tmp_path / "dangling.toml"
        dangling.symlink_to(missing_in)
        cfg3 = load_config(
            overrides={
                "io": {
                    "input_filenames": [str(missing_in)],
                    "output_filename": str(tmp_path / "out"),
                }
            }
        )
        with pytest.raises(ConfigError, match="aliases input file"):
            write_saved_config(cfg3, dangling)
        assert not missing_in.exists()
