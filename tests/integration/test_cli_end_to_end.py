"""End-to-end CLI integration tests on toy peak files."""

import os

import pandas as pd
import pytest

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
        self, tmp_path, reset_global_config, mgf_writer
    ):
        """Toy multi-charge MGF: clusters per charge, filters low-q
        spectra, natural-sorts CSV, exports readable MGF reps, and
        cleans up its temporary work_dir.
        """
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "toy.mgf",
            [
                # Charge 2+: scans 1 and 10 identical -> cluster; scan 2
                # distant m/z -> singleton.
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0),
                format_mgf_spectrum(
                    PEAKS_A, 500.002, charge=2, scan=10, rt=11.0
                ),
                format_mgf_spectrum(PEAKS_B, 600.0, charge=2, scan=2, rt=20.0),
                # Charge 3+: same m/z and peaks as scan 1, must NOT
                # merge with charge 2+ in default auto mode.
                format_mgf_spectrum(PEAKS_A, 500.0, charge=3, scan=3, rt=10.0),
                # Missing charge -> own unknown bucket.
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=None, scan=4, rt=10.0
                ),
                # Low-quality spectrum (2 peaks) -> excluded from CSV.
                format_mgf_spectrum(
                    LOW_QUALITY_PEAKS, 500.0, charge=2, scan=5, rt=10.0
                ),
            ],
        )
        out_prefix = tmp_path / "results"
        rc = falcon_main(
            [str(mgf), str(out_prefix), "--export_representatives"]
        )
        assert rc == 0
        # Automatic temporary work_dir must be deleted on completion.
        assert not os.path.exists(reset_global_config.work_dir)

        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        # Natural sort order: scan:1, scan:2, scan:3, scan:4, scan:10.
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

        # Representatives MGF has one spectrum per unique cluster (4).
        reps = list(ms_io.get_spectra(f"{out_prefix}.mgf"))
        assert len(reps) == 4
        assert sum(r.precursor_charge is None for r in reps) == 1

    def test_mixed_mzml_and_mzxml_inputs_cluster_across_formats(
        self, tmp_path, reset_global_config, mzml_writer, mzxml_writer
    ):
        """Identical MS2 spectra across mzML and mzXML cluster."""
        mzs_a = [p[0] for p in PEAKS_A]
        ints_a = [p[1] for p in PEAKS_A]
        mzs_b = [p[0] for p in PEAKS_B]
        ints_b = [p[1] for p in PEAKS_B]

        mzml = mzml_writer(
            tmp_path / "run_a.mzml",
            [
                # MS1 scan ignored.
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
        self, tmp_path, reset_global_config, mgf_writer
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
                "--precursor_charge_buckets",
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
        self, tmp_path, reset_global_config, mgf_writer
    ):
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

        # Run 1 builds cache in explicit work_dir.
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out1),
                    "--work_dir",
                    str(work_dir),
                    "--export_representatives",
                ]
            )
            == 0
        )
        assert (work_dir / "spectra" / "charges.joblib").is_file()

        # Run 2 reuses cache cleanly for a new output prefix.
        assert (
            falcon_main([str(mgf), str(out2), "--work_dir", str(work_dir)])
            == 0
        )
        df1 = pd.read_csv(f"{out1}.csv", comment="#")
        df2 = pd.read_csv(f"{out2}.csv", comment="#")
        pd.testing.assert_frame_equal(df1, df2)

        # Run 3 with mismatched charge buckets fails fast.
        with pytest.raises(ValueError, match="Cached charge buckets"):
            falcon_main(
                [
                    str(mgf),
                    str(tmp_path / "run3"),
                    "--work_dir",
                    str(work_dir),
                    "--precursor_charge_buckets",
                    "[2]",
                ]
            )

        # Run 4 without --overwrite aborts with exit code 1 AND without
        # --work_dir must not leak its temporary directory.
        assert falcon_main([str(mgf), str(out1)]) == 1
        assert not os.path.exists(reset_global_config.work_dir)

        # Run 5 with --overwrite refreshes cache and succeeds.
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out1),
                    "--work_dir",
                    str(work_dir),
                    "--overwrite",
                ]
            )
            == 0
        )

    def test_ini_config_average_consensus_rt_tol_and_root_scaling(
        self, tmp_path, reset_global_config, mgf_writer
    ):
        """End-to-end run exercising -c INI config, RT splitting, root
        scaling, and averaged representative export.
        """
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "avg.mgf",
            [
                # Early RT cluster (3 spectra)
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1, rt=10.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=2, rt=11.0),
                format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=3, rt=12.0),
                # Late RT cluster (2 spectra)
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=2, scan=4, rt=200.0
                ),
                format_mgf_spectrum(
                    PEAKS_A, 500.0, charge=2, scan=5, rt=201.0
                ),
            ],
        )
        ini = tmp_path / "falcon.ini"
        ini.write_text(
            "consensus_method = average\n"
            "scaling = root\n"
            "rt_tol = 10.0\n"
            "export_representatives = true\n"
        )
        out_prefix = tmp_path / "avg_out"
        rc = falcon_main([str(mgf), str(out_prefix), "-c", str(ini)])
        assert rc == 0

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

    def test_all_spectra_rejected_produces_valid_empty_outputs(
        self, tmp_path, reset_global_config, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        mgf = write_mgf(
            tmp_path / "lowq.mgf",
            [format_mgf_spectrum(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=1)],
        )
        out_prefix = tmp_path / "empty_run"
        rc = falcon_main(
            [str(mgf), str(out_prefix), "--export_representatives"]
        )
        assert rc == 0
        assert not os.path.exists(reset_global_config.work_dir)
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert len(df) == 0
        assert (tmp_path / "empty_run.mgf").is_file()

    def test_missing_input_glob_and_unwritable_output_fail_and_cleanup(
        self, tmp_path, reset_global_config, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        with pytest.raises(FileNotFoundError):
            falcon_main(
                [str(tmp_path / "missing_*.mgf"), str(tmp_path / "out")]
            )
        assert not os.path.exists(reset_global_config.work_dir)

        mgf = write_mgf(
            tmp_path / "ok.mgf",
            [format_mgf_spectrum(PEAKS_A, 500.0, charge=2, scan=1)],
        )
        with pytest.raises(FileNotFoundError):
            falcon_main([str(mgf), str(tmp_path / "no_such_dir" / "out")])
        assert not os.path.exists(reset_global_config.work_dir)
