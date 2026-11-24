"""Unit tests for falcon.staging (preprocessing, ingestion, and cache)."""

import json
import math
import os

import lance
import numpy as np
import pytest
import spectrum_utils.spectrum as sus

from falcon import staging
from falcon.config import PreprocessingConfig
from tests.helpers import (
    LOW_QUALITY_PEAKS,
    TOY_PEAKS_5,
    format_mgf_spectrum,
    make_falcon_config,
    make_spectrum_row,
    write_mgf,
)


def _make_spectrum(
    mz,
    intensity,
    *,
    identifier="test:scan:1",
    precursor_mz=500.0,
    precursor_charge=2,
    retention_time=10.0,
):
    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        np.asarray(mz, dtype=np.float32),
        np.asarray(intensity, dtype=np.float32),
        None,
        retention_time,
    )


class TestPreprocessing:
    def test_normalization_quality_gates_and_inclusive_mz_bounds(self):
        prep = PreprocessingConfig(
            min_peaks=3,
            min_mz_range=250.0,
            min_mz=100.0,
            max_mz=500.0,
            remove_precursor_tol=None,
            min_intensity=0.0,
            max_peaks_used=None,
        )
        spec = _make_spectrum(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [1e20, 2e20, 3e20, 4e20, 5e20],
            retention_time=42.5,
        )
        out = staging.process_spectrum(spec, prep)
        assert out is not None
        assert out["identifier"] == "test:scan:1"
        assert out["precursor_charge"] == 2
        assert out["retention_time"] == pytest.approx(42.5)
        assert np.linalg.norm(out["intensity"]) == pytest.approx(1.0, abs=1e-5)

        # Narrower m/z span or post-restriction failure rejects spectrum.
        narrow = _make_spectrum([100.0, 200.0, 349.0], [10.0, 20.0, 30.0])
        assert staging.process_spectrum(narrow, prep) is None
        outside = _make_spectrum(
            [50.0, 200.0, 300.0, 400.0, 1600.0], [10.0] * 5
        )
        assert staging.process_spectrum(outside, prep) is None

    @pytest.mark.parametrize("charge", [2, -2, 0, None])
    def test_precursor_and_charge_reduced_removal(self, charge):
        reduced = 998.9927 if charge == 2 else 1001.0073
        mzs = [150.0, 300.0, 500.0, 700.0, reduced, 1100.0]
        spec = _make_spectrum(
            mzs,
            [10.0] * len(mzs),
            precursor_mz=500.0,
            precursor_charge=charge,
        )
        prep = PreprocessingConfig(
            min_peaks=3,
            min_mz_range=200.0,
            min_mz=100.0,
            max_mz=1500.0,
            remove_precursor_tol=1.5,
            min_intensity=0.0,
            max_peaks_used=None,
        )
        out = staging.process_spectrum(spec, prep)
        assert out is not None
        assert all(abs(mz - 500.0) > 1.5 for mz in out["mz"])
        if charge in (2, -2):
            assert all(abs(mz - reduced) > 1.5 for mz in out["mz"])
        else:
            assert any(abs(mz - reduced) < 0.01 for mz in out["mz"])

    @pytest.mark.parametrize("scaling", ["off", "root", "log", "rank"])
    def test_intensity_filter_and_scaling_match_reference(self, scaling):
        raw = np.array([10.0, 40.0, 90.0, 160.0, 250.0], dtype=np.float32)
        spec = _make_spectrum([100.0, 200.0, 300.0, 400.0, 500.0], raw)
        prep = PreprocessingConfig(
            min_peaks=3,
            min_mz_range=200.0,
            min_mz=50.0,
            max_mz=600.0,
            remove_precursor_tol=None,
            min_intensity=0.0,
            max_peaks_used=5,
            scaling=scaling,
        )
        out = staging.process_spectrum(spec, prep)
        assert out is not None
        if scaling == "off":
            expected = raw
        elif scaling == "root":
            expected = np.sqrt(raw)
        elif scaling == "log":
            expected = np.log2(raw + 1.0)
        else:
            expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(out["intensity"], expected, rtol=1e-5)

        # Intensity floor and max_peaks_used retain strongest peaks.
        spec_top = _make_spectrum(
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [100.0, 0.5, 50.0, 80.0, 60.0],
        )
        prep_top = PreprocessingConfig(
            min_peaks=3,
            min_mz_range=200.0,
            min_mz=50.0,
            max_mz=600.0,
            remove_precursor_tol=None,
            min_intensity=0.01,
            max_peaks_used=3,
        )
        out_top = staging.process_spectrum(spec_top, prep_top)
        assert out_top is not None
        np.testing.assert_allclose(out_top["mz"], [100.0, 400.0, 500.0])

    def test_rejects_invalid_metadata_and_non_finite_peaks(self):
        prep = PreprocessingConfig(
            min_peaks=3,
            min_mz_range=100.0,
            min_mz=50.0,
            max_mz=600.0,
            remove_precursor_tol=None,
            min_intensity=0.0,
            max_peaks_used=None,
        )
        good_mz = [100.0, 200.0, 300.0, 400.0, 500.0]
        good_int = [10.0] * 5
        for spec in (
            _make_spectrum(good_mz, good_int, precursor_mz=0.0),
            _make_spectrum(good_mz, good_int, precursor_mz=float("nan")),
            _make_spectrum(good_mz, good_int, retention_time=float("inf")),
            _make_spectrum(good_mz, good_int, precursor_charge=200),
            _make_spectrum(good_mz, [0.0] * 5),
            _make_spectrum(good_mz, [-1.0, 10.0, 10.0, 10.0, 10.0]),
            _make_spectrum(good_mz, [10.0, np.nan, 10.0, 10.0, 10.0]),
            _make_spectrum([100.0, np.nan, 300.0, 400.0, 500.0], good_int),
            _make_spectrum([], []),
        ):
            assert staging.process_spectrum(spec, prep) is None

        nan_rt = _make_spectrum(good_mz, good_int, retention_time=float("nan"))
        out = staging.process_spectrum(nan_rt, prep)
        assert out is not None
        assert math.isnan(out["retention_time"])


class TestStagingAndCache:
    def test_resolve_input_files_sorted_globs_brackets_and_errors(
        self, tmp_path
    ):
        for name in ("c.mgf", "a.mgf", "b.mgf", "sample_[1].mgf"):
            (tmp_path / name).write_text("", encoding="utf-8")

        expanded = staging.resolve_input_files(
            [str(tmp_path / "[a-c].mgf"), str(tmp_path / "b.mgf")]
        )
        assert expanded == [
            tmp_path / "a.mgf",
            tmp_path / "b.mgf",
            tmp_path / "c.mgf",
            tmp_path / "b.mgf",
        ]
        assert staging.resolve_input_files(
            [str(tmp_path / "sample_[1].mgf")]
        ) == [tmp_path / "sample_[1].mgf"]

        with pytest.raises(FileNotFoundError, match="matched no files"):
            staging.resolve_input_files([str(tmp_path / "missing_*.mgf")])
        with pytest.raises(IsADirectoryError, match="directory"):
            staging.resolve_input_files([str(tmp_path)])
        broken = tmp_path / "dangling.mgf"
        broken.symlink_to(tmp_path / "nonexistent.mgf")
        with pytest.raises(FileNotFoundError, match="regular file"):
            staging.resolve_input_files([str(broken)])

    def test_canonical_metadata_sort_with_nan_rt_ties(self, lance_dataset):
        nan = float("nan")
        rows = [
            make_spectrum_row(
                "z_high", 600.0, [100.0], [1.0], retention_time=1.0
            ),
            make_spectrum_row(
                "nan_b", 500.0, [100.0], [1.0], retention_time=nan
            ),
            make_spectrum_row(
                "rt20", 500.0, [100.0], [1.0], retention_time=20.0
            ),
            make_spectrum_row(
                "tie_b", 500.0, [100.0], [1.0], retention_time=10.0
            ),
            make_spectrum_row(
                "nan_a", 500.0, [100.0], [1.0], retention_time=nan
            ),
            make_spectrum_row(
                "tie_a", 500.0, [100.0], [1.0], retention_time=10.0
            ),
            make_spectrum_row(
                "low", 400.0, [100.0], [1.0], retention_time=99.0
            ),
        ]
        ds = lance_dataset(rows)
        expected = [
            "low",
            "tie_a",
            "tie_b",
            "rt20",
            "nan_a",
            "nan_b",
            "z_high",
        ]
        df = staging.read_canonical_metadata(
            ds, staging.METADATA_COLUMNS, with_row_index=True
        )
        assert df["identifier"].tolist() == expected
        taken = ds.take(
            df["index"].tolist(), columns=["identifier"]
        ).to_pydict()["identifier"]
        assert taken == expected

    def test_stage_spectra_multi_file_explicit_buckets_and_symlinks(
        self, tmp_path
    ):
        f1 = write_mgf(
            tmp_path / "run_a.mgf",
            [
                format_mgf_spectrum(TOY_PEAKS_5, 500.0, charge=3, scan=1),
                format_mgf_spectrum(TOY_PEAKS_5, 500.0, charge=None, scan=2),
                format_mgf_spectrum(
                    LOW_QUALITY_PEAKS, 500.0, charge=2, scan=3
                ),
            ],
        )
        f2 = write_mgf(
            tmp_path / "run_b.mgf",
            [
                format_mgf_spectrum(TOY_PEAKS_5, 600.0, charge=2, scan=1),
                format_mgf_spectrum(TOY_PEAKS_5, 650.0, charge=-1, scan=2),
            ],
        )
        wd = tmp_path / "work"
        cfg_auto = make_falcon_config(
            input_filenames=[str(f1), str(f2)],
            output_filename=tmp_path / "out",
        )
        buckets = staging.stage_spectra([f1, f2], wd, cfg_auto)
        assert buckets == [(-1,), (2,), (3,), ("unknown",)]
        for bucket in buckets:
            ds = lance.dataset(str(staging.bucket_dataset_path(wd, bucket)))
            assert ds.count_rows() == 1

        wd2 = tmp_path / "work_explicit"
        cfg_exp = make_falcon_config(
            input_filenames=[str(f1), str(f2)],
            output_filename=tmp_path / "out",
            clustering={"precursor_charge_buckets": [[2, 3], [5]]},
        )
        assert staging.stage_spectra([f1, f2], wd2, cfg_exp) == [(2, 3)]
        ds23 = lance.dataset(str(staging.bucket_dataset_path(wd2, (2, 3))))
        assert ds23.count_rows() == 2

        # Outward fragment_tol grid alignment retains peaks just outside
        # [min_mz, max_mz] but inside [101.00, 500.05].
        aligned_mgf = write_mgf(
            tmp_path / "aligned.mgf",
            [
                format_mgf_spectrum(
                    [
                        (100.9, 10.0),
                        (101.00, 10.0),
                        (300.0, 10.0),
                        (500.04, 10.0),
                        (500.10, 10.0),
                    ],
                    600.0,
                    charge=2,
                    scan=1,
                )
            ],
        )
        wd4 = tmp_path / "work_aligned"
        cfg_aligned = make_falcon_config(
            input_filenames=[str(aligned_mgf)],
            output_filename=tmp_path / "out",
            preprocessing={
                "min_peaks": 3,
                "min_mz_range": 100.0,
                "min_mz": 101.03,
                "max_mz": 500.02,
                "remove_precursor_tol": "none",
                "min_intensity": 0.0,
            },
            clustering={"fragment_tol": 0.05},
        )
        staging.stage_spectra([aligned_mgf], wd4, cfg_aligned)
        staged_mzs = (
            lance.dataset(str(staging.bucket_dataset_path(wd4, (2,))))
            .to_table(columns=["mz"])
            .to_pydict()["mz"][0]
        )
        np.testing.assert_allclose(staged_mzs, [101.00, 300.0, 500.04])

        # Symlink stem is preserved in identifiers.
        blob = tmp_path / "raw_blob"
        blob.write_text(f1.read_text(encoding="utf-8"), encoding="utf-8")
        link = tmp_path / "alias_run.mgf"
        link.symlink_to(blob)
        wd3 = tmp_path / "work_sym"
        cfg_sym = make_falcon_config(
            input_filenames=[str(link)],
            output_filename=tmp_path / "out",
            clustering={"precursor_charge_buckets": [[3]]},
        )
        staging.stage_spectra([link], wd3, cfg_sym)
        ids = (
            lance.dataset(str(staging.bucket_dataset_path(wd3, (3,))))
            .to_table(columns=["identifier"])
            .to_pydict()["identifier"]
        )
        assert ids == ["alias_run:scan:1"]

    def test_cache_reuse_invalidation_overwrite_and_corruption(
        self, tmp_path, monkeypatch
    ):
        mgf = write_mgf(
            tmp_path / "in.mgf",
            [format_mgf_spectrum(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
        )
        wd = tmp_path / "work"
        cfg = make_falcon_config(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(wd)},
        )
        assert staging.load_or_stage_buckets(cfg, wd) == [(2,)]
        cache_path = wd / "spectra" / staging.STAGING_CACHE_FILENAME
        assert cache_path.is_file()

        # Reuses cache across clustering/consensus sweeps without restaging.
        sweep_cfg = make_falcon_config(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out2",
            io={"work_dir": str(wd), "export_representatives": True},
            clustering={"distance_threshold": 0.25, "rt_tol": 10.0},
            consensus={"method": "average"},
        )
        monkeypatch.setattr(
            staging,
            "stage_spectra",
            lambda *_a, **_kw: pytest.fail("should not restage"),
        )
        assert staging.load_or_stage_buckets(sweep_cfg, wd) == [(2,)]

        # Changing preprocessing, fragment_tol, buckets, or input invalidates.
        for key, override in (
            ("preprocessing", {"preprocessing": {"min_mz": 50.0}}),
            ("fragment_tol", {"clustering": {"fragment_tol": 0.02}}),
            (
                "precursor_charge_buckets",
                {"clustering": {"precursor_charge_buckets": [[1]]}},
            ),
        ):
            changed = make_falcon_config(
                input_filenames=[str(mgf)],
                output_filename=tmp_path / "out",
                io={"work_dir": str(wd)},
                **override,
            )
            with pytest.raises(ValueError, match=key):
                staging.load_or_stage_buckets(changed, wd)

        # Touching input file invalidates cache.
        mgf.write_text(
            mgf.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        st = os.stat(mgf)
        os.utime(mgf, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        with pytest.raises(ValueError, match="inputs"):
            staging.load_or_stage_buckets(cfg, wd)

        # Overwrite clears stale files and restages cleanly.
        monkeypatch.undo()
        ow_cfg = make_falcon_config(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(wd), "overwrite": True},
        )
        assert staging.load_or_stage_buckets(ow_cfg, wd) == [(2,)]

        # Corrupted JSON or orphaned Lance dataset without manifest fail fast.
        cache_path.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted staging cache"):
            staging.load_or_stage_buckets(cfg, wd)

        cache_path.unlink()
        with pytest.raises(ValueError, match="unverified staged"):
            staging.load_or_stage_buckets(cfg, wd)

        # Manifest referencing missing bucket dir fails fast.
        manifest = json.loads(
            json.dumps(staging._build_manifest(cfg, [mgf], buckets=[(9,)]))
        )
        cache_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="do not match cached buckets"):
            staging.load_or_stage_buckets(cfg, wd)
