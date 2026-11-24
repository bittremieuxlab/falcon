"""Unit tests for falcon.staging (dataset, routing, ingestion, cache)."""

import functools
import json
import os

import lance
import numpy as np
import pytest

from falcon import preprocessing, staging
from tests.helpers import LOW_QUALITY_PEAKS, TOY_PEAKS_5

_TOY_PROC = functools.partial(
    preprocessing.process_spectrum, min_peaks=5, min_mz_range=250.0
)


class TestDatasetHelpersAndMetadata:
    def test_bucket_naming_and_uri_labels(self, tmp_path):
        assert (
            staging.bucket_dataset_name((1, 2)) == "spectra_charge_1_2.lance"
        )
        assert (
            staging.bucket_dataset_path(tmp_path, ("unknown",))
            == tmp_path / "spectra" / "spectra_charge_unknown.lance"
        )
        assert staging.format_bucket_label((1, 2)) == "1, 2"
        assert staging.format_bucket_label("other") == "other"
        assert (
            staging.bucket_label_from_uri(
                "/work/spectra/spectra_charge_3_unknown.lance", "fb"
            )
            == "3, unknown"
        )
        assert (
            staging.bucket_label_from_uri("/work/custom.lance", "fb") == "fb"
        )

    def test_canonical_metadata_sort_with_nan_rt_ties(
        self, lance_dataset, spectrum_row
    ):
        nan = float("nan")
        rows = [
            spectrum_row("z_high", 600.0, [100.0], [1.0], retention_time=1.0),
            spectrum_row("nan_b", 500.0, [100.0], [1.0], retention_time=nan),
            spectrum_row("rt20", 500.0, [100.0], [1.0], retention_time=20.0),
            spectrum_row("tie_b", 500.0, [100.0], [1.0], retention_time=10.0),
            spectrum_row("nan_a", 500.0, [100.0], [1.0], retention_time=nan),
            spectrum_row("tie_a", 500.0, [100.0], [1.0], retention_time=10.0),
            spectrum_row("low", 400.0, [100.0], [1.0], retention_time=99.0),
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


class TestChargeRouterAndStaging:
    def test_charge_router_auto_explicit_and_other(self):
        auto = staging.ChargeRouter(None)
        assert auto.route(2) == (2,)
        assert auto.route(np.int8(-1)) == (-1,)
        assert auto.route(None) == ("unknown",)

        explicit = staging.ChargeRouter([(1,), (3, 2), ("unknown",), "other"])
        assert explicit.canonical_buckets == (
            (1,),
            (2, 3),
            ("unknown",),
            "other",
        )
        assert explicit.route(3) == (2, 3)
        assert explicit.route(None) == ("unknown",)
        assert explicit.route(5) == "other"

        strict = staging.ChargeRouter([(2,)])
        assert strict.route(3) is None

        with pytest.raises(ValueError, match="int8 range"):
            auto.route(200)

    def test_stage_spectra_single_multi_file_explicit_and_symlinks(
        self, tmp_path, mgf_block, mgf_writer
    ):
        f1 = mgf_writer(
            tmp_path / "run_a.mgf",
            [
                mgf_block(TOY_PEAKS_5, 500.0, charge=3, scan=1),
                mgf_block(TOY_PEAKS_5, 500.0, charge=None, scan=2),
                mgf_block(LOW_QUALITY_PEAKS, 500.0, charge=2, scan=3),
            ],
        )
        f2 = mgf_writer(
            tmp_path / "run_b.mgf",
            [
                mgf_block(TOY_PEAKS_5, 600.0, charge=2, scan=1),
                mgf_block(TOY_PEAKS_5, 650.0, charge=-1, scan=2),
            ],
        )
        wd = tmp_path / "work"
        buckets = staging.stage_spectra([f1, f2], wd, _TOY_PROC, None)
        assert buckets == [(-1,), (2,), (3,), ("unknown",)]
        for bucket in buckets:
            ds = lance.dataset(str(staging.bucket_dataset_path(wd, bucket)))
            assert ds.count_rows() == 1

        # Explicit buckets preserve order and drop unmatched charges.
        wd2 = tmp_path / "work_explicit"
        exp_buckets = staging.stage_spectra(
            [f1, f2], wd2, _TOY_PROC, [(2, 3), (5,)]
        )
        assert exp_buckets == [(2, 3)]
        ds23 = lance.dataset(str(staging.bucket_dataset_path(wd2, (2, 3))))
        assert ds23.count_rows() == 2

        # Symlink stem is preserved in identifiers.
        blob = tmp_path / "raw_blob"
        blob.write_text(f1.read_text(encoding="utf-8"), encoding="utf-8")
        link = tmp_path / "alias_run.mgf"
        link.symlink_to(blob)
        wd3 = tmp_path / "work_sym"
        staging.stage_spectra([link], wd3, _TOY_PROC, [(3,)])
        ids = (
            lance.dataset(str(staging.bucket_dataset_path(wd3, (3,))))
            .to_table(columns=["identifier"])
            .to_pydict()["identifier"]
        )
        assert ids == ["alias_run:scan:1"]


class TestStagingCache:
    def test_cache_reuse_invalidation_and_overwrite(
        self, tmp_path, mgf_block, mgf_writer, config_factory, monkeypatch
    ):
        mgf = mgf_writer(
            tmp_path / "in.mgf",
            [mgf_block(TOY_PEAKS_5, 500.0, charge=2, scan=1)],
        )
        wd = tmp_path / "work"
        cfg = config_factory(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(wd)},
        )
        buckets = staging.load_or_stage_buckets(cfg, wd, _TOY_PROC)
        assert buckets == [(2,)]
        assert (wd / "spectra" / staging.STAGING_CACHE_FILENAME).is_file()

        # Reuses cache across clustering/consensus sweeps without restaging.
        sweep_cfg = config_factory(
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
        assert staging.load_or_stage_buckets(sweep_cfg, wd, _TOY_PROC) == [
            (2,)
        ]

        # Changing preprocessing, fragment_tol, buckets, or input invalidates.
        for section, override in (
            ("preprocessing", {"preprocessing": {"min_mz": 50.0}}),
            ("fragment_tol", {"clustering": {"fragment_tol": 0.02}}),
            (
                "precursor_charge_buckets",
                {"clustering": {"precursor_charge_buckets": [[1]]}},
            ),
        ):
            changed = config_factory(
                input_filenames=[str(mgf)],
                output_filename=tmp_path / "out",
                io={"work_dir": str(wd)},
                **override,
            )
            with pytest.raises(ValueError, match=section):
                staging.load_or_stage_buckets(changed, wd, _TOY_PROC)

        # Touching input file invalidates cache.
        mgf.write_text(
            mgf.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        st = os.stat(mgf)
        os.utime(mgf, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        with pytest.raises(ValueError, match="inputs"):
            staging.load_or_stage_buckets(cfg, wd, _TOY_PROC)

        # Overwrite clears stale files and restages cleanly.
        monkeypatch.undo()
        (wd / "spectra" / "charges.joblib").write_text("old", encoding="utf-8")
        ow_cfg = config_factory(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(wd), "overwrite": True},
        )
        assert staging.load_or_stage_buckets(ow_cfg, wd, _TOY_PROC) == [(2,)]
        assert not (wd / "spectra" / "charges.joblib").exists()

    def test_corrupted_legacy_and_missing_dataset_caches_fail_fast(
        self, tmp_path, config_factory
    ):
        mgf = tmp_path / "in.mgf"
        mgf.write_text("", encoding="utf-8")
        wd = tmp_path / "work"
        s_dir = staging.spectra_dir(wd)
        s_dir.mkdir(parents=True)
        cfg = config_factory(
            input_filenames=[str(mgf)],
            output_filename=tmp_path / "out",
            io={"work_dir": str(wd)},
        )

        # Orphaned legacy file without manifest fails fast.
        legacy = s_dir / "charges.joblib"
        legacy.write_bytes(b"legacy")
        with pytest.raises(ValueError, match="overwrite"):
            staging.load_or_stage_buckets(cfg, wd, _TOY_PROC)
        legacy.unlink()

        # Corrupted JSON fails fast.
        cache_path = s_dir / staging.STAGING_CACHE_FILENAME
        cache_path.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted staging cache"):
            staging.load_or_stage_buckets(cfg, wd, _TOY_PROC)

        # Valid manifest missing on-disk Lance dir fails fast.
        manifest = staging.build_staging_manifest(cfg, [mgf], buckets=[(2,)])
        cache_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="do not match cached buckets"):
            staging.load_or_stage_buckets(cfg, wd, _TOY_PROC)
