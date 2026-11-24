"""End-to-end CLI integration tests on toy peak files."""

import subprocess
import sys

import pandas as pd
import pytest

import falcon
from falcon import ms_io
from falcon.cli import main as falcon_main
from tests.helpers import (
    LOW_QUALITY_PEAKS,
    format_mgf_spectrum,
    write_mgf,
    write_toy_mzml,
    write_toy_mzxml,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("isolated_root_logger"),
]

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


class TestCliEndToEnd:
    def test_multi_charge_mgf_with_quality_filtering_and_representatives(
        self, tmp_path, track_mkdtemp
    ):
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
        assert len(track_mkdtemp) == 1
        assert not track_mkdtemp[0].exists()

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
        self, tmp_path
    ):
        mzs_a, ints_a = [p[0] for p in PEAKS_A], [p[1] for p in PEAKS_A]
        mzs_b, ints_b = [p[0] for p in PEAKS_B], [p[1] for p in PEAKS_B]

        mzml = write_toy_mzml(
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
        mzxml = write_toy_mzxml(
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
        assert falcon_main([str(mzml), str(mzxml), str(out_prefix)]) == 0

        by_id = (
            pd.read_csv(f"{out_prefix}.csv", comment="#")
            .set_index("identifier")["cluster"]
            .to_dict()
        )
        assert by_id["run_a:scan:2"] == by_id["run_b:scan:1"]
        assert by_id["run_a:scan:2"] != by_id["run_b:scan:2"]

    def test_explicit_charge_buckets_cross_charge_and_drop_unmapped(
        self, tmp_path
    ):
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

    def test_hierarchical_toml_extends_average_rt_tol_and_save_replay(
        self, tmp_path
    ):
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
        assert (
            falcon_main(
                [
                    str(mgf),
                    str(out_prefix),
                    "-c",
                    str(child_toml),
                    "--save-config",
                    str(saved_toml),
                ]
            )
            == 0
        )

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
        assert sorted(
            round(float(r.retention_time), 1)
            for r in reps
            if r.retention_time is not None
        ) == [11.0, 200.5]

        replayed_out = tmp_path / "replayed_out"
        assert (
            falcon_main(["-c", str(saved_toml), "-o", str(replayed_out)]) == 0
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(f"{out_prefix}.csv", comment="#"),
            pd.read_csv(f"{replayed_out}.csv", comment="#"),
        )

    def test_programmatic_api_empty_run_negative_mode_and_module_version(
        self, tmp_path
    ):
        # Programmatic top-level API + negative-ion precursor removal.
        peaks_with_prec = [*PEAKS_A, (500.0, 10_000.0)]
        mgf = write_mgf(
            tmp_path / "neg.mgf",
            [
                format_mgf_spectrum(
                    peaks_with_prec, 500.0, charge=-2, scan=1, rt=10.0
                ),
                format_mgf_spectrum(
                    peaks_with_prec, 500.002, charge=-2, scan=2, rt=12.0
                ),
            ],
        )
        out_prefix = tmp_path / "neg_out"
        cfg = falcon.load_config(
            overrides={
                "io": {
                    "input_filenames": [str(mgf)],
                    "output_filename": str(out_prefix),
                    "export_representatives": True,
                }
            }
        )
        assert falcon.run(cfg) == 0
        df = pd.read_csv(f"{out_prefix}.csv", comment="#")
        assert list(df["precursor_charge"]) == [-2, -2]
        assert df["cluster"].nunique() == 1
        reps = list(ms_io.get_spectra(f"{out_prefix}.mgf"))
        assert len(reps) == 1
        assert reps[0].precursor_charge == -2
        assert all(abs(float(mz) - 500.0) > 1.5 for mz in reps[0].mz)

        # All-rejected input produces valid empty CSV and MGF.
        lowq = write_mgf(
            tmp_path / "lowq.mgf",
            [format_mgf_spectrum(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=1)],
        )
        empty_out = tmp_path / "empty_run"
        assert (
            falcon_main(
                [str(lowq), str(empty_out), "--export-representatives"]
            )
            == 0
        )
        assert len(pd.read_csv(f"{empty_out}.csv", comment="#")) == 0
        assert (tmp_path / "empty_run.mgf").is_file()

        # python -m falcon --version smoke check.
        proc = subprocess.run(
            [sys.executable, "-m", "falcon", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"falcon {falcon.__version__}" in proc.stdout
