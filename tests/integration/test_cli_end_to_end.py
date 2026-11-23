"""End-to-end CLI integration tests on toy peak files."""

import json
import os
import tempfile
import tomllib

import pandas as pd
import pytest

import falcon
from falcon.falcon import main as falcon_main
from falcon.ms_io import ms_io

PEAKS_A = [
    (150.0, 100.0),
    (250.0, 200.0),
    (350.0, 150.0),
    (450.0, 300.0),
    (550.0, 250.0),
    (650.0, 120.0),
]
PEAKS_B = [
    (180.0, 100.0),
    (280.0, 200.0),
    (380.0, 150.0),
    (480.0, 300.0),
    (580.0, 250.0),
    (680.0, 120.0),
]
LOW_QUALITY_PEAKS = [(150.0, 100.0), (160.0, 200.0)]


class TestCliEndToEnd:
    def test_multi_charge_mgf_with_quality_filtering_and_representatives(
        self, tmp_path, mgf_writer, monkeypatch
    ):
        """Toy multi-charge MGF: clusters per charge, filters low-q
        spectra, natural-sorts CSV, exports readable MGF reps, and
        cleans up its temporary work_dir.
        """
        created_tmp_dirs: list[str] = []
        orig_mkdtemp = tempfile.mkdtemp

        def _tracking_mkdtemp(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created_tmp_dirs.append(d)
            return d

        monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "toy.mgf",
            [
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0),
                format_mgf_spectrum(
                    PEAKS_A, 500.002, charge=2, scan=10, rt=11.0
                ),
                format_mgf_spectrum(PEAKS_B, 600.0, charge=2, scan=2, rt=20.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=3, scan=3, rt=10.0),
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=None, scan=4, rt=10.0
                ),
                format_mgf_spectrum(
                    LOW_QUALITY_PEAKS, 500.0, charge=2, scan=5, rt=10.0
                ),
            ],
        )
        out_prefix = tmp_path / "results.v2"
        rc = falcon_main(
            [str(mgf), str(out_prefix), "--export-representatives"]
        )
        assert rc == 0
        assert len(created_tmp_dirs) == 1
        assert not os.path.exists(created_tmp_dirs[0])

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["identifier"]) == [
            "toy:scan:1",
            "toy:scan:2",
            "toy:scan:3",
            "toy:scan:4",
            "toy:scan:10",
        ]
        by_id = df.set_index("identifier")["cluster"].to_dict()
        assert by_id["toy:scan:1"] == by_id["toy:scan:10"]
        assert (
            len(
                {
                    by_id["toy:scan:1"],
                    by_id["toy:scan:2"],
                    by_id["toy:scan:3"],
                    by_id["toy:scan:4"],
                }
            )
            == 4
        )

        reps = list(ms_io.get_spectra(f"{out_prefix}.mgf"))
        assert len(reps) == 4
        assert sum(r.precursor_charge is None for r in reps) == 1

    def test_mixed_mzml_and_mzxml_inputs_cluster_across_formats(
        self, tmp_path, mzml_writer, mzxml_writer
    ):
        """Identical MS2 spectra across mzML and mzXML cluster."""
        mzs_a = [p[0] for p in PEAKS_A]
        ints_a = [p[1] for p in PEAKS_A]
        mzs_b = [p[0] for p in PEAKS_B]
        ints_b = [p[1] for p in PEAKS_B]

        mzml = mzml_writer(
            tmp_path / "run_a.mzml",
            [
                {
                    "scan": 1,
                    "ms_level": 1,
                    "mz": mzs_a,
                    "intensity": ints_a,
                },
                {
                    "scan": 2,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "rt": 60.0,
                    "mz": mzs_a,
                    "intensity": ints_a,
                },
            ],
        )
        mzxml = mzxml_writer(
            tmp_path / "run_b.mzxml",
            [
                {
                    "scan": 1,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "rt": 60,
                    "mz": mzs_a,
                    "intensity": ints_a,
                },
                {
                    "scan": 2,
                    "ms_level": 2,
                    "precursor_mz": 500.0,
                    "charge": 2,
                    "rt": 60,
                    "mz": mzs_b,
                    "intensity": ints_b,
                },
            ],
        )
        out_prefix = tmp_path / "mixed_out"
        rc = falcon_main([str(mzml), str(mzxml), str(out_prefix)])
        assert rc == 0

        by_id = (
            pd.read_csv(f"{out_prefix}.csv", comment="#")
            .set_index("identifier")["cluster"]
            .to_dict()
        )
        assert by_id["run_a:scan:2"] == by_id["run_b:scan:1"]
        assert by_id["run_a:scan:2"] != by_id["run_b:scan:2"]

    def test_explicit_charge_buckets_cross_charge_and_drop_unmapped(
        self, tmp_path, mgf_writer
    ):
        """Grouped bucket '[2, 3]' clusters charges 2 and 3 together;
        omitting 'other' drops charge 4 spectra.
        """
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "charges.mgf",
            [
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=3, scan=2),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=None, scan=3),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=4, scan=4),
            ],
        )
        out_prefix = tmp_path / "grouped"
        rc = falcon_main(
            [
                str(mgf),
                str(out_prefix),
                "--precursor-charge-buckets",
                "[2, 3]",
                "[unknown]",
            ]
        )
        assert rc == 0
        df = pd.read_csv(f"{out_prefix}.csv", comment="#").set_index(
            "identifier"
        )
        assert list(df.index) == [
            "charges:scan:1",
            "charges:scan:2",
            "charges:scan:3",
        ]
        assert (
            df.loc["charges:scan:1", "cluster"]
            == df.loc["charges:scan:2", "cluster"]
        )
        assert (
            df.loc["charges:scan:1", "cluster"]
            != df.loc["charges:scan:3", "cluster"]
        )

    def test_work_dir_cache_reuse_mismatch_overwrite_and_abort_cleanup(
        self, tmp_path, mgf_writer, monkeypatch
    ):
        created_tmp_dirs: list[str] = []
        orig_mkdtemp = tempfile.mkdtemp

        def _tracking_mkdtemp(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created_tmp_dirs.append(d)
            return d

        monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "cache.mgf",
            [
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=2, rt=11.0),
            ],
        )
        work_dir = tmp_path / "work"
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        # Run 1 builds staging cache in explicit work_dir.
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out1),
                    "--work-dir",
                    str(work_dir),
                    "--export-representatives",
                ]
            )
            == 0
        )
        assert (work_dir / "spectra" / "staging_cache.json").is_file()

        # Run 2 reuses cache cleanly across a distance_threshold sweep.
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out2),
                    "--work-dir",
                    str(work_dir),
                    "--distance-threshold",
                    "0.2",
                ]
            )
            == 0
        )
        df1 = pd.read_csv(f"{out1}.csv", comment="#")
        df2 = pd.read_csv(f"{out2}.csv", comment="#")
        pd.testing.assert_frame_equal(df1, df2)

        # Run 3 with changed preprocessing fails fast naming preprocessing.
        with pytest.raises(ValueError, match="preprocessing"):
            falcon_main(
                [
                    str(mgf),
                    str(tmp_path / "run3"),
                    "--work-dir",
                    str(work_dir),
                    "--min-mz",
                    "50.0",
                ]
            )

        # Run 4 without --overwrite aborts with exit code 1 AND cleans up
        # its automatic temporary work_dir.
        assert falcon_main([str(mgf), str(out1)]) == 1
        assert len(created_tmp_dirs) == 1
        assert not os.path.exists(created_tmp_dirs[0])

        # Run 5 with --overwrite refreshes cache and succeeds.
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out1),
                    "--work-dir",
                    str(work_dir),
                    "--overwrite",
                ]
            )
            == 0
        )

    def test_hierarchical_toml_extends_average_rt_tol_and_save_replay(
        self, tmp_path, mgf_writer
    ):
        """End-to-end run with hierarchical TOML extends, RT splitting,
        root scaling, averaged representatives, and --save-config replay.
        """
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "avg.mgf",
            [
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=2, rt=11.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=3, rt=12.0),
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=2, scan=4, rt=200.0
                ),
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=2, scan=5, rt=201.0
                ),
            ],
        )
        base_toml = tmp_path / "base.toml"
        base_toml.write_text(
            '[preprocessing]\nscaling = "root"\n'
            '[consensus]\nmethod = "average"\n',
            encoding="utf-8",
        )
        child_toml = tmp_path / "child.toml"
        child_toml.write_text(
            'extends = "base.toml"\n'
            "[io]\nexport_representatives = true\n"
            "[clustering]\nrt_tol = 10.0\n",
            encoding="utf-8",
        )
        out_prefix = tmp_path / "avg_out"
        saved_toml = tmp_path / "saved.toml"
        rc = falcon_main(
            [
                str(mgf),
                str(out_prefix),
                "-c",
                str(child_toml),
                "--save-config",
                str(saved_toml),
            ]
        )
        assert rc == 0
        assert saved_toml.is_file()

        by_id = (
            pd.read_csv(f"{out_prefix}.csv", comment="#")
            .set_index("identifier")["cluster"]
            .to_dict()
        )
        assert (
            by_id["avg:scan:1"] == by_id["avg:scan:2"] == by_id["avg:scan:3"]
        )
        assert by_id["avg:scan:4"] == by_id["avg:scan:5"]
        assert by_id["avg:scan:1"] != by_id["avg:scan:4"]

        reps = list(ms_io.get_spectra(f"{out_prefix}.mgf"))
        assert len(reps) == 2
        assert sorted(round(r.retention_time, 1) for r in reps) == [
            11.0,
            200.5,
        ]

        # Replay saved TOML directly with a new output prefix via -o.
        replayed_out = tmp_path / "replayed_out"
        assert (
            falcon_main(["-c", str(saved_toml), "-o", str(replayed_out)]) == 0
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(f"{out_prefix}.csv", comment="#"),
            pd.read_csv(f"{replayed_out}.csv", comment="#"),
        )

    def test_print_config_and_dump_schema_cli_actions(self, capsys):
        assert falcon_main(["--print-config", "--min-matched-peaks", "6"]) == 0
        printed = capsys.readouterr().out
        parsed = tomllib.loads(printed)
        assert parsed["clustering"]["min_matched_peaks"] == 6

        assert falcon_main(["--dump-schema"]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert "extends" in schema["properties"]

    def test_programmatic_top_level_api(self, tmp_path, mgf_writer):
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "api.mgf",
            [format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0)],
        )
        cfg = falcon.load_config(
            overrides={
                "io": {
                    "input_filenames": [str(mgf)],
                    "output_filename": str(tmp_path / "api_out"),
                }
            }
        )
        assert falcon.run(cfg) == 0
        df = pd.read_csv(f"{tmp_path / 'api_out'}.csv", comment="#")
        assert list(df["identifier"]) == ["api:scan:1"]

    def test_all_spectra_rejected_produces_valid_empty_outputs(
        self, tmp_path, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "lowq.mgf",
            [format_mgf_spectrum(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=1)],
        )
        out_prefix = tmp_path / "empty_run"
        rc = falcon_main(
            [str(mgf), str(out_prefix), "--export-representatives"]
        )
        assert rc == 0
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert len(df) == 0
        assert (tmp_path / "empty_run.mgf").is_file()

    def test_missing_input_glob_and_unwritable_output_fail_and_cleanup(
        self, tmp_path, mgf_writer, monkeypatch
    ):
        created_tmp_dirs: list[str] = []
        orig_mkdtemp = tempfile.mkdtemp

        def _tracking_mkdtemp(*args, **kwargs):
            d = orig_mkdtemp(*args, **kwargs)
            created_tmp_dirs.append(d)
            return d

        monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

        format_mgf_spectrum, write_mgf = mgf_writer
        with pytest.raises(FileNotFoundError):
            falcon_main(
                [str(tmp_path / "missing_*.mgf"), str(tmp_path / "out")]
            )
        assert not os.path.exists(created_tmp_dirs[-1])

        mgf = write_mgf(
            tmp_path / "ok.mgf",
            [format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1)],
        )
        with pytest.raises(FileNotFoundError):
            falcon_main([str(mgf), str(tmp_path / "no_such_dir" / "out")])
        assert not os.path.exists(created_tmp_dirs[-1])
