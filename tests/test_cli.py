"""Unit tests for falcon.cli argument parsing and entry point."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from falcon import cli, pipeline


class TestCliParsingAndMain:
    def test_positional_named_and_layered_cli_overrides(
        self, tmp_path, monkeypatch
    ):
        captured = []
        monkeypatch.setattr(
            pipeline, "run", lambda cfg: captured.append(cfg) or 0
        )

        assert cli.main(["a.mgf", "b.mzml", "dir/run.v2"]) == 0
        assert captured[-1].io.input_filenames == ("a.mgf", "b.mzml")
        assert captured[-1].io.output_filename == Path("dir/run.v2")

        assert (
            cli.main(["-i", "a.mgf", "-i", "b.mzml", "c.mzxml", "-o", "out"])
            == 0
        )
        assert captured[-1].io.input_filenames == (
            "a.mgf",
            "b.mzml",
            "c.mzxml",
        )

        base = tmp_path / "base.toml"
        base.write_text(
            '[io]\ninput_filenames = ["a.mgf"]\noutput_filename = "toml_out"\n'
            "[preprocessing]\nremove_precursor_tol = 2.0\n"
            "[clustering]\ndistance_threshold = 0.2\n"
            '[consensus]\nmethod = "average"\noutlier_cutoff_lower = 2.5\n',
            encoding="utf-8",
        )
        assert (
            cli.main(
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
            == 0
        )
        cfg = captured[-1]
        assert cfg.io.output_filename == Path("cli_out")
        assert cfg.io.export_representatives is True
        assert cfg.preprocessing.remove_precursor_tol is None
        assert cfg.clustering.precursor_tol.mass == 0.05
        assert cfg.clustering.precursor_tol.mode == "Da"
        assert cfg.clustering.distance_threshold == 0.2
        assert cfg.consensus.method == "medoid"

    def test_charge_buckets_and_invalid_cli_arguments(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            pipeline, "run", lambda cfg: captured.append(cfg) or 0
        )

        assert (
            cli.main(
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
            == 0
        )
        assert captured[-1].clustering.precursor_charge_buckets == (
            (-2, -1),
            (2, 3),
            ("unknown",),
            "other",
        )

        assert (
            cli.main(["a.mgf", "out", "--precursor-charge-buckets", "auto"])
            == 0
        )
        assert captured[-1].clustering.precursor_charge_buckets is None

        for bad in (
            ["a.mgf"],
            ["a.mgf", "out", "-o", "other"],
            ["--print-config", "--dump-schema"],
            ["a.mgf", "out", "--precursor-charge-buckets", "[1", "2]"],
            ["a.mgf", "out", "--precursor-charge-buckets", "auto", "[1]"],
            ["a.mgf", "out", "--precursor-charge-buckets", "[abc]"],
            ["a.mgf", "out", "--distance-threshold", "1.5"],
        ):
            with pytest.raises(SystemExit) as exc:
                cli.main(bad)
            assert exc.value.code == 2

    def test_inspection_flags_and_save_error(self, tmp_path, capsys):
        assert cli.main(["--dump-schema"]) == 0
        assert "extends" in json.loads(capsys.readouterr().out)["properties"]

        assert (
            cli.main(
                [
                    "--print-config",
                    "--min-peaks",
                    "8",
                    "--consensus-method",
                    "average",
                ]
            )
            == 0
        )
        parsed = tomllib.loads(capsys.readouterr().out)
        assert parsed["preprocessing"]["min_peaks"] == 8
        assert parsed["consensus"]["method"] == "average"

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
