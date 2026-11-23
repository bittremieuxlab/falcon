"""Unit tests for falcon.spectra_io."""

import functools
import queue
import threading
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pytest

from falcon import spectra_io
from falcon.cluster import spectrum
from falcon.config import FalconConfig, IOConfig


def _make_spec(identifier, charge):
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    intensity = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    intensity /= np.linalg.norm(intensity)
    return {
        "identifier": identifier,
        "precursor_mz": np.float32(500.0),
        "precursor_charge": charge,
        "mz": mz,
        "intensity": intensity,
        "retention_time": np.float32(0.0),
    }


def _run_write_worker(
    spectra,
    charge_to_bucket,
    catch_other_charges,
    work_dir,
    schema,
    auto_bucket=False,
):
    q = queue.Queue()
    for spec in spectra:
        q.put(spec)
    q.put(None)

    locks = spectra_io._PerChargeLockRegistry()
    spectra_io._write_spectra_lance(
        q,
        locks,
        schema,
        charge_to_bucket,
        catch_other_charges,
        auto_bucket,
        work_dir,
    )

    spectra_dir = work_dir / "spectra"
    counts = {}
    if spectra_dir.exists():
        for p in spectra_dir.iterdir():
            if p.suffix == ".lance":
                counts[p.name] = lance.dataset(str(p)).count_rows()
    return counts


@pytest.fixture
def staging_work_dir(tmp_path):
    """Create a temporary working directory with a ``spectra`` folder."""
    (tmp_path / "spectra").mkdir()
    return tmp_path


# ----------------------------------------------------------------------
# Input glob resolution, bucket_key_to_str, and lock registry
# ----------------------------------------------------------------------


class TestResolveInputFiles:
    def test_expands_globs_sorted_deterministically_and_preserves_duplicates(
        self, tmp_path
    ):
        (tmp_path / "b.mgf").write_text("")
        (tmp_path / "a.mgf").write_text("")
        resolved = spectra_io.resolve_input_files(
            [str(tmp_path / "*.mgf"), str(tmp_path / "a.mgf")]
        )
        assert resolved == [
            tmp_path / "a.mgf",
            tmp_path / "b.mgf",
            tmp_path / "a.mgf",
        ]

    def test_unmatched_glob_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            spectra_io.resolve_input_files([str(tmp_path / "missing_*.mgf")])


class TestBucketKeyToStr:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ((1,), "_1"),
            ((1, 2), "_1_2"),
            ((-2, -1), "_-2_-1"),
            ((3, "unknown"), "_3_unknown"),
            ("other", "_other"),
            ("unknown", "_unknown"),
            ({2, 1}, "_1_2"),
            ({"unknown", 3}, "_3_unknown"),
        ],
    )
    def test_valid_keys(self, key, expected):
        assert spectra_io.bucket_key_to_str(key) == expected

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported bucket key type"):
            spectra_io.bucket_key_to_str(42)  # type: ignore[arg-type]


class TestPerChargeLockRegistry:
    def test_returns_same_lock_per_key_and_distinct_across_keys(self):
        reg = spectra_io._PerChargeLockRegistry()
        assert reg.get((1,)) is reg.get((1,))
        assert reg.get((1,)) is not reg.get((2,))

    def test_thread_safe_concurrent_get(self):
        reg = spectra_io._PerChargeLockRegistry()
        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(reg.get((2,)))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(lock) for lock in results}) == 1


# ----------------------------------------------------------------------
# Lance dataset creation and writes
# ----------------------------------------------------------------------


class TestWriteToDataset:
    def test_creates_and_appends_at_explicit_work_dir(
        self, tmp_path, spectrum_schema
    ):
        explicit_dir = tmp_path / "target"
        (explicit_dir / "spectra").mkdir(parents=True)

        lock = threading.Lock()
        count1 = spectra_io._write_to_dataset(
            [_make_spec("s1", 2), _make_spec("s2", 2)],
            (2,),
            lock,
            spectrum_schema,
            explicit_dir,
        )
        count2 = spectra_io._write_to_dataset(
            [_make_spec("s3", 2)],
            (2,),
            lock,
            spectrum_schema,
            explicit_dir,
        )
        assert count1 == 2
        assert count2 == 1
        target_ds = explicit_dir / "spectra" / "spectra_charge_2.lance"
        assert target_ds.exists()
        assert lance.dataset(str(target_ds)).count_rows() == 3


# ----------------------------------------------------------------------
# Charge routing and batch flushing
# ----------------------------------------------------------------------


class TestChargeRouting:
    def test_explicit_single_multi_unknown_and_other_routing(
        self, staging_work_dir, spectrum_schema
    ):
        charge_to_bucket = {
            1: (1,),
            2: (2, 3),
            3: (2, 3),
            "unknown": ("unknown",),
        }
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", 3),
            _make_spec("d", None),
            _make_spec("e", 5),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=True,
            work_dir=staging_work_dir,
            schema=spectrum_schema,
        )
        assert counts == {
            "spectra_charge_1.lance": 1,
            "spectra_charge_2_3.lance": 2,
            "spectra_charge_unknown.lance": 1,
            "spectra_charge_other.lance": 1,
        }

    def test_unmapped_charge_dropped_without_other(
        self, staging_work_dir, spectrum_schema
    ):
        counts = _run_write_worker(
            [_make_spec("x", 3), _make_spec("u", None)],
            charge_to_bucket={1: (1,)},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            schema=spectrum_schema,
        )
        assert counts == {}

    def test_auto_bucket_isolates_all_charges_including_negative_and_unknown(
        self, staging_work_dir, spectrum_schema
    ):
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", 1),
            _make_spec("d", -2),
            _make_spec("u", None),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            schema=spectrum_schema,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_1.lance": 2,
            "spectra_charge_2.lance": 1,
            "spectra_charge_-2.lance": 1,
            "spectra_charge_unknown.lance": 1,
        }

    def test_batch_flush_threshold_writes_all_rows_across_flushes(
        self, staging_work_dir, spectrum_schema, monkeypatch
    ):
        """Mid-stream flushes at batch size plus final sentinel."""
        monkeypatch.setattr(spectra_io, "_LANCE_WRITE_BATCH_SIZE", 3)
        spectra = [
            *(_make_spec(f"c2_{i}", 2) for i in range(6)),
            *(_make_spec(f"c1_{i}", 1) for i in range(4)),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            schema=spectrum_schema,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_2.lance": 6,
            "spectra_charge_1.lance": 4,
        }


# ----------------------------------------------------------------------
# Auto bucket discovery, reading, staging, and CSV export
# ----------------------------------------------------------------------


class TestDiscoverAutoBuckets:
    def test_sorts_numeric_ascending_unknown_last_and_ignores_other_files(
        self, staging_work_dir, spectrum_schema
    ):
        _run_write_worker(
            [
                _make_spec("a", 3),
                _make_spec("b", -1),
                _make_spec("c", None),
                _make_spec("d", 1),
            ],
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=staging_work_dir,
            schema=spectrum_schema,
            auto_bucket=True,
        )
        spectra_dir = staging_work_dir / "spectra"
        (spectra_dir / "staging_cache.json").write_text("{}")
        (spectra_dir / "notes.txt").write_text("ignore")

        assert spectra_io._discover_auto_buckets(str(spectra_dir)) == [
            (-1,),
            (1,),
            (3,),
            ("unknown",),
        ]


class TestReadAndPrepareSpectra:
    def test_read_spectra_counts_accepted_and_rejected(
        self, tmp_path, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        good_peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        lowq_peaks = [(150.0, 100.0), (160.0, 200.0)]
        mgf = write_mgf(
            tmp_path / "mix.mgf",
            [
                format_mgf_spectrum(good_peaks, 500.0, charge=2, scan=1),
                format_mgf_spectrum(lowq_peaks, 500.0, charge=2, scan=2),
            ],
        )
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        accepted, rejected = spectra_io._read_spectra(str(mgf), proc)
        assert len(accepted) == 1
        assert accepted[0]["identifier"] == "mix:scan:1"
        assert rejected == 1

    def test_prepare_spectra_explicit_buckets_skips_empty_and_rejects_overlap(
        self, staging_work_dir, mgf_writer
    ):
        format_mgf_spectrum, write_mgf = mgf_writer
        peaks = [
            (150.0, 100.0),
            (250.0, 200.0),
            (350.0, 150.0),
            (450.0, 300.0),
            (550.0, 250.0),
        ]
        mgf = write_mgf(
            staging_work_dir / "sample.mgf",
            [
                format_mgf_spectrum(peaks, 500.0, charge=2, scan=1),
                format_mgf_spectrum(peaks, 500.0, charge=4, scan=2),
            ],
        )
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        # Bucket (1,) has no spectra and is skipped; charge 4 goes to "other".
        valid = spectra_io._prepare_spectra(
            [mgf], staging_work_dir, proc, [(1,), (2, 3), "other"]
        )
        assert valid == [(2, 3), "other"]

        with pytest.raises(ValueError, match="more than one bucket"):
            spectra_io._prepare_spectra(
                [mgf], staging_work_dir, proc, [{1, 2}, {2, 3}]
            )

        with pytest.raises(ValueError, match="other"):
            spectra_io._prepare_spectra(
                [mgf], staging_work_dir, proc, [{1}, "other", "other"]
            )

    def test_prepare_spectra_empty_input_list_raises_file_not_found(
        self, staging_work_dir
    ):
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            spectra_io._prepare_spectra([], staging_work_dir, proc, None)


class TestWriteClusterInfo:
    def test_writes_metadata_comments_parseable_csv_and_dotted_prefix(
        self, staging_work_dir
    ):
        cfg = FalconConfig(
            io=IOConfig(
                input_filenames=("in.mgf",),
                output_filename=staging_work_dir / "run_v1.2",
            )
        )
        df = pd.DataFrame(
            {
                "identifier": ["f:scan:1", "f:scan:2"],
                "precursor_charge": [2, 2],
                "precursor_mz": [500.0, 500.05],
                "retention_time": [10.0, 11.0],
                "cluster": [0, 0],
            }
        )
        spectra_io._write_cluster_info(df, cfg, staging_work_dir)
        csv_path = Path(f"{cfg.io.output_filename}.csv")
        assert csv_path == staging_work_dir / "run_v1.2.csv"
        raw_lines = csv_path.read_text().splitlines()
        assert raw_lines[0].startswith("# falcon version ")
        assert any(
            line == "# clustering.linkage = complete" for line in raw_lines
        )
        assert any(
            line == f"# io.work_dir = {staging_work_dir}" for line in raw_lines
        )
        parsed = pd.read_csv(csv_path, comment="#")
        assert list(parsed.columns) == [
            "identifier",
            "precursor_charge",
            "precursor_mz",
            "retention_time",
            "cluster",
        ]
        assert list(parsed["identifier"]) == ["f:scan:1", "f:scan:2"]
        assert list(parsed["cluster"]) == [0, 0]
