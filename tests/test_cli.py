"""Unit tests for falcon.cli argument parsing, actions, and entry point."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from falcon import cli
from falcon.config import AverageConsensusConfig, FalconConfig


class TestCliParsing:
    def test_positional_and_named_io_paths(self):
        action = cli.parse_cli(["a.mgf", "b.mzml", "dir/run.v2"])
        assert isinstance(action, cli.RunAction)
        assert action.config.io.input_filenames == ("a.mgf", "b.mzml")
        assert action.config.io.output_filename == Path("dir/run.v2")

        named = cli.parse_cli(
            ["-i", "a.mgf", "-i", "b.mzml", "c.mzxml", "-o", "out"]
        )
        assert isinstance(named, cli.RunAction)
        assert named.config.io.input_filenames == (
            "a.mgf",
            "b.mzml",
            "c.mzxml",
        )

        # Cannot mix positional and -i/-o or pass single positional.
        for bad in (["a.mgf"], ["a.mgf", "out", "-o", "other"]):
            with pytest.raises(SystemExit) as exc:
                cli.parse_cli(bad)
            assert exc.value.code == 2

    def test_layered_toml_and_cli_override_precedence(self, tmp_path):
        base = tmp_path / "base.toml"
        base.write_text(
            '[io]\ninput_filenames = ["a.mgf"]\noutput_filename = "toml_out"\n'
            "[preprocessing]\nremove_precursor_tol = 2.0\n"
            "[clustering]\ndistance_threshold = 0.2\n"
            '[consensus]\nmethod = "average"\noutlier_cutoff_lower = 2.5\n',
            encoding="utf-8",
        )
        action = cli.parse_cli(
            [
                "-c",
                str(base),
                "-o",
                "cli_out",
                "--remove-precursor-tol",
                "none",
                "--precursor-tol",
                "0.05",
                "Da",
                "--consensus-method",
                "medoid",
                "--export-representatives",
            ]
        )
        assert isinstance(action, cli.RunAction)
        cfg = action.config
        assert cfg.io.output_filename == Path("cli_out")
        assert cfg.io.export_representatives is True
        assert cfg.preprocessing.remove_precursor_tol is None
        assert cfg.clustering.precursor_tol.mass == 0.05
        assert cfg.clustering.precursor_tol.mode == "Da"
        assert cfg.clustering.distance_threshold == 0.2
        assert cfg.consensus.method == "medoid"

    def test_charge_buckets_cli_parsing_and_errors(self):
        action = cli.parse_cli(
            [
                "a.mgf",
                "out",
                "--precursor-charge-buckets",
                "[-2, -1]",
                "[2, 3]",
                "unknown",
                "other",
            ]
        )
        assert isinstance(action, cli.RunAction)
        assert action.config.clustering.precursor_charge_buckets == (
            (-2, -1),
            (2, 3),
            ("unknown",),
            "other",
        )

        auto_action = cli.parse_cli(
            ["a.mgf", "out", "--precursor-charge-buckets", "auto"]
        )
        assert isinstance(auto_action, cli.RunAction)
        assert auto_action.config.clustering.precursor_charge_buckets is None

        for bad in (
            ["a.mgf", "out", "--precursor-charge-buckets", "[1", "2]"],
            ["a.mgf", "out", "--precursor-charge-buckets", "auto", "[1]"],
            ["a.mgf", "out", "--precursor-charge-buckets", "[abc]"],
        ):
            with pytest.raises(SystemExit) as exc:
                cli.parse_cli(bad)
            assert exc.value.code == 2

    def test_utility_flags_and_mutual_exclusion(self, tmp_path):
        assert isinstance(
            cli.parse_cli(["--dump-schema"]), cli.DumpSchemaAction
        )
        printed = cli.parse_cli(
            ["--print-config", "--consensus-method", "average"]
        )
        assert isinstance(printed, cli.PrintConfigAction)
        assert isinstance(printed.settings.consensus, AverageConsensusConfig)

        saved = cli.parse_cli(
            ["a.mgf", "out", "--save-config", str(tmp_path / "cfg.toml")]
        )
        assert isinstance(saved, cli.RunAction)
        assert saved.save_config == tmp_path / "cfg.toml"

        with pytest.raises(SystemExit) as exc:
            cli.parse_cli(["--print-config", "--dump-schema"])
        assert exc.value.code == 2

    def test_format_validation_error_strips_union_tags(self):
        with pytest.raises(ValidationError) as exc_info:
            FalconConfig.model_validate(
                {
                    "io": {
                        "input_filenames": ["a.mgf"],
                        "output_filename": "o",
                    },
                    "consensus": {
                        "method": "average",
                        "outlier_cutoff_lower": -1.0,
                    },
                }
            )
        text = cli.format_validation_error(exc_info.value)
        assert "consensus.outlier_cutoff_lower" in text
        assert "AverageConsensusConfig" not in text


class TestCliMainAndStartup:
    def test_main_inspection_and_save_error(self, tmp_path, capsys):
        assert cli.main(["--dump-schema"]) == 0
        assert "extends" in json.loads(capsys.readouterr().out)["properties"]

        assert cli.main(["--print-config", "--min-peaks", "8"]) == 0
        parsed = tomllib.loads(capsys.readouterr().out)
        assert parsed["preprocessing"]["min_peaks"] == 8

        # Invalid save-config destination (directory) returns exit code 2.
        rc = cli.main(["a.mgf", "out", "--save-config", str(tmp_path)])
        assert rc == 2
        assert "falcon: error:" in capsys.readouterr().err

    def test_lightweight_cli_inspection_defers_heavy_imports(self):
        script = (
            "import sys\n"
            "import falcon, falcon.cli\n"
            "assert falcon.cli.main(['--print-config']) == 0\n"
            "assert falcon.cli.main(['--dump-schema']) == 0\n"
            "heavy = ('numpy', 'numba', 'scipy', 'lance', 'pyarrow', "
            "'pandas', 'fastcluster', 'joblib', 'spectrum_utils', "
            "'pyteomics')\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'Heavy modules loaded eagerly: {loaded}'\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
