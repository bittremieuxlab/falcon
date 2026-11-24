"""Unit tests for falcon.config.serialize TOML, schema, and export."""

import json
import os
import tomllib
from pathlib import Path

import pytest

from falcon.config import (
    AverageConsensusConfig,
    ClusteringConfig,
    ConfigError,
    FalconConfig,
    FalconSettings,
    IOConfig,
    PartialIOConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    build_json_schema,
    format_metadata_lines,
    format_toml,
    load_config,
    load_settings,
    write_saved_config,
)
from falcon.config import serialize as config_serialize
from falcon.config.serialize import _toml_literal


class TestFormatTomlAndMetadata:
    def test_format_toml_round_trip_and_commented_disabled_fields(
        self, tmp_path
    ):
        default_settings = FalconSettings()
        default_toml = format_toml(default_settings)
        assert "# input_filenames = []" in default_toml
        assert '# output_filename = ""' in default_toml
        assert '# work_dir = "none"' in default_toml
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
        toml_text = format_toml(custom)
        assert '\nremove_precursor_tol = "none"\n' in toml_text
        assert '\nmax_peaks_used = "none"\n' in toml_text
        replayed = tmp_path / "replayed.toml"
        replayed.write_text(toml_text, encoding="utf-8")
        assert load_config(config_files=[replayed]) == custom

    def test_format_toml_work_dir_override(self, tmp_path):
        override_dir = tmp_path / "override_work"
        toml_text = format_toml(FalconSettings(), work_dir=override_dir)
        parsed = tomllib.loads(toml_text)
        assert parsed["io"]["work_dir"] == str(override_dir)

    def test_format_toml_round_trip_non_bmp_filenames(self, tmp_path):
        cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(
                    str(tmp_path / "sample_🧪.mgf"),
                    "spectra_𝄞/*.mzML",
                ),
                output_filename=tmp_path / "results_🧬" / "run_🧪",
                work_dir=tmp_path / "work_🔬",
            )
        )
        toml_text = format_toml(cfg)
        parsed = tomllib.loads(toml_text)
        assert parsed["io"]["input_filenames"] == list(cfg.io.input_filenames)
        assert parsed["io"]["output_filename"] == str(cfg.io.output_filename)
        assert parsed["io"]["work_dir"] == str(cfg.io.work_dir)

        saved = tmp_path / "saved_🧪.toml"
        saved.write_text(toml_text, encoding="utf-8")
        assert load_config(config_files=[saved]) == cfg

        settings = FalconSettings(
            io=PartialIOConfig(
                input_filenames=("inspect_🧪.mgf",),
                output_filename=Path("inspect_🧬"),
                work_dir=Path("work_🔬"),
            )
        )
        settings_toml = format_toml(settings)
        assert load_settings(overrides=tomllib.loads(settings_toml)) == (
            settings
        )

    def test_format_toml_escapes_quotes_backslashes_and_control_chars(
        self, tmp_path
    ):
        all_controls = "".join(chr(code) for code in range(0x20)) + "\x7f"
        special_str = f'in_\\"quote"\\path_{all_controls}_end.mgf'
        special_path = Path(f'dir_\\sub"q"_{all_controls}_out')
        special_work = Path('work_\\t\t"q"\n\r\b\f\x1b\x7f')
        cfg = FalconConfig(
            io=IOConfig(
                input_filenames=(special_str,),
                output_filename=special_path,
                work_dir=special_work,
            )
        )
        toml_text = format_toml(cfg)
        for expected_escape in (
            r"\\",
            r"\"",
            r"\b",
            r"\t",
            r"\n",
            r"\f",
            r"\r",
            r"\u0000",
            r"\u001b",
            r"\u007f",
        ):
            assert expected_escape in toml_text
        # No raw DEL or C0 control characters outside line breaks.
        for line in toml_text.splitlines():
            assert all(ord(ch) >= 0x20 and ord(ch) != 0x7F for ch in line)

        saved = tmp_path / "escaped.toml"
        saved.write_text(toml_text, encoding="utf-8")
        assert load_config(config_files=[saved]) == cfg

    def test_literal_dot_slash_none_work_dir_round_trips(self, tmp_path):
        cfg = FalconConfig(
            io=IOConfig.model_validate(
                {
                    "input_filenames": ["a.mgf"],
                    "output_filename": "out",
                    "work_dir": "./none",
                }
            )
        )
        toml_text = format_toml(cfg)
        assert 'work_dir = "./none"' in toml_text
        replayed = tmp_path / "replay_none.toml"
        replayed.write_text(toml_text, encoding="utf-8")
        reloaded = load_config(config_files=[replayed])
        assert reloaded.io.work_dir == Path("none")

    @pytest.mark.parametrize(
        "io_kwargs",
        [
            {
                "input_filenames": ("bad_\ud800.mgf",),
                "output_filename": Path("o"),
            },
            {
                "input_filenames": ("ok.mgf",),
                "output_filename": Path("bad_\udfff"),
            },
        ],
    )
    def test_format_toml_rejects_lone_surrogates(self, io_kwargs):
        cfg = FalconConfig(io=IOConfig(**io_kwargs))
        with pytest.raises(ConfigError, match="surrogate"):
            format_toml(cfg)

    def test_toml_literal_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported TOML value type"):
            _toml_literal({"nested": 1})

    def test_format_metadata_lines_medoid_and_average(
        self, tmp_path, config_factory
    ):
        work_dir = tmp_path / "wd"
        medoid_cfg = config_factory()
        med_lines = format_metadata_lines(medoid_cfg, work_dir=work_dir)
        assert any(line.startswith("falcon version ") for line in med_lines)
        assert f"io.work_dir = {work_dir}" in med_lines
        assert "clustering.precursor_tol = 20.0 ppm" in med_lines
        assert "consensus.method = medoid" in med_lines
        assert not any("outlier_cutoff" in line for line in med_lines)

        avg_cfg = config_factory(
            consensus={
                "method": "average",
                "outlier_cutoff_lower": 2.0,
                "outlier_cutoff_upper": 3.5,
            }
        )
        avg_lines = format_metadata_lines(avg_cfg)
        assert "consensus.method = average" in avg_lines
        assert "consensus.outlier_cutoff_lower = 2.0" in avg_lines
        assert "consensus.outlier_cutoff_upper = 3.5" in avg_lines

    def test_format_metadata_lines_for_falcon_settings(self):
        lines = format_metadata_lines(FalconSettings())
        assert lines[0].startswith("falcon version ")
        assert "io.work_dir = None" in lines
        assert "consensus.method = medoid" in lines

    def test_format_metadata_lines_sanitizes_control_chars_in_work_dir(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            config_serialize,
            "__version__",
            "1.0\ninjected\r\t\b\f\x00\x1b\x7f",
        )
        lines = format_metadata_lines(
            FalconSettings(),
            work_dir=Path("wd\ninjected\r\t\b\f\x00\x1b\x7fdir"),
        )
        for line in lines:
            assert all(ord(ch) >= 0x20 and ord(ch) != 0x7F for ch in line)

        version_line = lines[0]
        work_dir_line = next(
            line for line in lines if line.startswith("io.work_dir = ")
        )
        for target_line in (version_line, work_dir_line):
            for escaped in (
                "\\n",
                "\\r",
                "\\t",
                "\\b",
                "\\f",
                "\\x00",
                "\\x1b",
                "\\x7f",
            ):
                assert escaped in target_line


class TestJsonSchema:
    @staticmethod
    def _assert_core_schema_contracts(schema):
        serialized = json.dumps(schema)
        assert isinstance(serialized, str)
        for section in ("io", "preprocessing", "clustering", "consensus"):
            assert section in schema["properties"]
        assert '"const": "none"' in serialized
        assert '"const": "auto"' in serialized
        # Empty [consensus] table requires anyOf rather than oneOf.
        assert "anyOf" in schema["properties"]["consensus"]
        assert "oneOf" not in schema["properties"]["consensus"]

        # Every config field carries a non-empty description.
        for def_name, def_schema in schema["$defs"].items():
            for prop_name, prop_schema in def_schema["properties"].items():
                desc = prop_schema.get("description")
                assert isinstance(desc, str), (
                    f"{def_name}.{prop_name} missing description"
                )
                assert desc.strip(), (
                    f"{def_name}.{prop_name} has blank description"
                )

        io_defs = schema["$defs"]["PartialIOConfig"]["properties"]
        in_array = next(
            b
            for b in io_defs["input_filenames"]["anyOf"]
            if b.get("type") == "array"
        )
        assert in_array["minItems"] == 1
        assert in_array["items"]["minLength"] == 1
        for path_field in ("output_filename", "work_dir"):
            path_branch = next(
                b
                for b in io_defs[path_field]["anyOf"]
                if b.get("format") == "path"
            )
            assert path_branch["minLength"] == 1

        buckets_def = schema["$defs"]["ClusteringConfig"]["properties"][
            "precursor_charge_buckets"
        ]
        bucket_arr = next(
            b for b in buckets_def["anyOf"] if b.get("type") == "array"
        )
        assert bucket_arr["minItems"] == 1
        inner_arr = next(
            b for b in bucket_arr["items"]["anyOf"] if b.get("type") == "array"
        )
        assert inner_arr["minItems"] == 1
        assert inner_arr["uniqueItems"] is True
        int_elem = next(
            b
            for b in inner_arr["items"]["anyOf"]
            if b.get("type") == "integer"
        )
        assert (int_elem["minimum"], int_elem["maximum"]) == (-128, 127)

    def test_build_json_schema_and_native_model_schema_constraints(self):
        native_schema = FalconSettings.model_json_schema()
        self._assert_core_schema_contracts(native_schema)

        schema = build_json_schema()
        self._assert_core_schema_contracts(schema)
        assert "extends" in schema["properties"]
        assert schema["properties"]["extends"]["description"].strip()


class TestWriteSavedConfig:
    def test_round_trips_config_to_file(self, tmp_path, config_factory):
        cfg = config_factory(
            input_filenames=(str(tmp_path / "in.mgf"),),
            output_filename=tmp_path / "out",
        )
        dest = tmp_path / "saved.toml"
        write_saved_config(cfg, dest)
        assert load_config(config_files=[dest]) == cfg

    def test_rejects_directory_and_unwritable_destination(
        self, tmp_path, config_factory
    ):
        cfg = config_factory()
        with pytest.raises(
            ConfigError, match="Config output path is a directory"
        ):
            write_saved_config(cfg, tmp_path)

        for bad_dest, match in (
            ("", "must not be empty"),
            ("   ", "must not be empty"),
            (Path("   "), "must not be empty"),
            (f"{tmp_path}/   ", "non-empty file name"),
            (f"{tmp_path}/no_such_dir/", "directory"),
            (f"{tmp_path}/dir/.", "directory"),
            (f"{tmp_path}/bad\x00.toml", "Failed to resolve"),
        ):
            with pytest.raises(ConfigError, match=match):
                write_saved_config(cfg, bad_dest)

        missing_parent = tmp_path / "no_such_dir" / "saved.toml"
        with pytest.raises(ConfigError, match="Failed to write configuration"):
            write_saved_config(cfg, missing_parent)

    def test_rejects_direct_and_relative_input_aliases(
        self, tmp_path, config_factory
    ):
        input_file = tmp_path / "sample.mgf"
        original = "BEGIN IONS\nPEPMASS=500.0\nEND IONS\n"
        input_file.write_text(original, encoding="utf-8")
        cfg = config_factory(
            input_filenames=(str(input_file),),
            output_filename=tmp_path / "out",
        )

        with pytest.raises(
            ConfigError, match=r"Config output path .* aliases input file"
        ):
            write_saved_config(cfg, input_file)
        assert input_file.read_text(encoding="utf-8") == original

        sub = tmp_path / "sub"
        sub.mkdir()
        rel_alias = sub / ".." / "sample.mgf"
        with pytest.raises(
            ConfigError, match=r"Config output path .* aliases input file"
        ):
            write_saved_config(cfg, rel_alias)
        assert input_file.read_text(encoding="utf-8") == original

    def test_rejects_glob_selected_input_alias(self, tmp_path, config_factory):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        spec_a = data_dir / "a.mgf"
        spec_b = data_dir / "b.mgf"
        spec_a.write_text("MGF_A\n", encoding="utf-8")
        spec_b.write_text("MGF_B\n", encoding="utf-8")
        cfg = config_factory(
            input_filenames=(str(data_dir / "*.mgf"),),
            output_filename=tmp_path / "out",
        )

        for target, expected in ((spec_a, "MGF_A\n"), (spec_b, "MGF_B\n")):
            with pytest.raises(ConfigError, match="aliases input file"):
                write_saved_config(cfg, target)
            assert target.read_text(encoding="utf-8") == expected

    def test_rejects_symlink_and_hardlink_aliases(
        self, tmp_path, config_factory
    ):
        real_mgf = tmp_path / "real.mgf"
        real_mgf.write_text("REAL_MGF\n", encoding="utf-8")
        sym_dest = tmp_path / "sym_config.toml"
        sym_dest.symlink_to(real_mgf)
        hard_dest = tmp_path / "hard_config.toml"
        os.link(real_mgf, hard_dest)

        cfg = config_factory(
            input_filenames=(str(real_mgf),),
            output_filename=tmp_path / "out",
        )
        for dest in (sym_dest, hard_dest):
            with pytest.raises(ConfigError, match="aliases input file"):
                write_saved_config(cfg, dest)
            assert real_mgf.read_text(encoding="utf-8") == "REAL_MGF\n"

        # Reverse direction: symlinked input, real path as save target.
        sym_in = tmp_path / "in_link.mgf"
        sym_in.symlink_to(real_mgf)
        cfg_sym_in = config_factory(
            input_filenames=(str(sym_in),),
            output_filename=tmp_path / "out",
        )
        with pytest.raises(ConfigError, match="aliases input file"):
            write_saved_config(cfg_sym_in, real_mgf)
        assert real_mgf.read_text(encoding="utf-8") == "REAL_MGF\n"

    def test_rejects_output_csv_and_mgf_aliases(
        self, tmp_path, config_factory
    ):
        cfg = config_factory(
            input_filenames=(str(tmp_path / "in.mgf"),),
            output_filename=tmp_path / "out",
        )
        for output_target in (cfg.io.csv_path, cfg.io.mgf_path):
            with pytest.raises(ConfigError, match="aliases output file"):
                write_saved_config(cfg, output_target)
            assert not output_target.exists()

    def test_rejects_work_dir_alias(self, tmp_path, config_factory):
        work_dir = tmp_path / "planned_work_dir"
        cfg = config_factory(
            input_filenames=(str(tmp_path / "in.mgf"),),
            output_filename=tmp_path / "out",
            io={"work_dir": str(work_dir)},
        )
        with pytest.raises(ConfigError, match="aliases working directory"):
            write_saved_config(cfg, work_dir)
        assert not work_dir.exists()

        sub = tmp_path / "sub"
        sub.mkdir()
        rel_alias = sub / ".." / "planned_work_dir"
        with pytest.raises(ConfigError, match="aliases working directory"):
            write_saved_config(cfg, rel_alias)
        assert not work_dir.exists()

    def test_rejects_nonexistent_literal_dangling_symlink_and_symlink_loop(
        self, tmp_path, config_factory
    ):
        missing = tmp_path / "missing.mgf"
        cfg_missing = config_factory(
            input_filenames=(str(missing),),
            output_filename=tmp_path / "out",
        )
        with pytest.raises(ConfigError, match="aliases input file"):
            write_saved_config(cfg_missing, missing)
        assert not missing.exists()

        dangling = tmp_path / "dangling.toml"
        dangling.symlink_to(missing)
        with pytest.raises(ConfigError, match="aliases input file"):
            write_saved_config(cfg_missing, dangling)
        assert not missing.exists()

        loop = tmp_path / "loop.toml"
        loop.symlink_to(loop)
        cfg_other = config_factory(
            input_filenames=(str(tmp_path / "other.mgf"),),
            output_filename=tmp_path / "out",
        )
        with pytest.raises(
            ConfigError, match="Failed to resolve config output path"
        ):
            write_saved_config(cfg_other, loop)

    def test_allows_distinct_destinations_with_globs_and_broken_inputs(
        self, tmp_path, config_factory
    ):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.mgf").write_text("MGF_A\n", encoding="utf-8")
        glob_pattern = str(data_dir / "*.mgf")
        cfg_glob = config_factory(
            input_filenames=(glob_pattern,),
            output_filename=tmp_path / "out",
        )

        new_dest = data_dir / "saved.toml"
        existing_dest = tmp_path / "existing.toml"
        existing_dest.write_text("# old\n", encoding="utf-8")
        for dest in (new_dest, existing_dest):
            write_saved_config(cfg_glob, dest)
            assert dest.is_file()
            loaded = load_config(config_files=[dest])
            assert loaded.io.input_filenames == (glob_pattern,)
        assert (data_dir / "a.mgf").read_text(encoding="utf-8") == "MGF_A\n"

        # Unmatched glob does not block saving to a distinct file.
        unmatched_dest = tmp_path / "saved_unmatched.toml"
        cfg_unmatched = config_factory(
            input_filenames=(str(tmp_path / "none_*.mgf"),),
            output_filename=tmp_path / "out",
        )
        write_saved_config(cfg_unmatched, unmatched_dest)
        assert unmatched_dest.is_file()

        # A symlink-loop input does not crash saving elsewhere.
        bad_input = tmp_path / "bad_input.mgf"
        bad_input.symlink_to(bad_input)
        distinct_dest = tmp_path / "saved_distinct.toml"
        cfg_loop_in = config_factory(
            input_filenames=(str(bad_input),),
            output_filename=tmp_path / "out",
        )
        write_saved_config(cfg_loop_in, distinct_dest)
        assert distinct_dest.is_file()
