"""Unit tests for falcon.spectra_io."""

import functools
import queue
import threading

import lance
import numpy as np
import pandas as pd
import pytest

from falcon import spectra_io
from falcon.cluster import spectrum


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
    )

    spectra_dir = work_dir / "spectra"
    counts = {}
    if spectra_dir.exists():
        for p in spectra_dir.iterdir():
            if p.suffix == ".lance":
                counts[p.name] = lance.dataset(str(p)).count_rows()
    return counts


@pytest.fixture
def configured_work_dir(tmp_path, reset_global_config):
    """Initialize global config with a temp work_dir for staging."""
    (tmp_path / "spectra").mkdir()
    reset_global_config.parse(["dummy.mgf", str(tmp_path / "out")])
    reset_global_config.work_dir = str(tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# bucket_key_to_str and lock registry
# ----------------------------------------------------------------------


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
        self, tmp_path, reset_global_config, spectrum_schema
    ):
        """_write_to_dataset must honor its work_dir argument directly,
        independent of global config.work_dir.
        """
        explicit_dir = tmp_path / "target"
        (explicit_dir / "spectra").mkdir(parents=True)
        other_dir = tmp_path / "other"
        (other_dir / "spectra").mkdir(parents=True)
        reset_global_config.parse(["dummy.mgf", str(tmp_path / "out")])
        reset_global_config.work_dir = str(other_dir)

        lock = threading.Lock()
        count1 = spectra_io._write_to_dataset(
            [_make_spec("s1", 2), _make_spec("s2", 2)],
            (2,),
            lock,
            spectrum_schema,
            str(explicit_dir),
        )
        count2 = spectra_io._write_to_dataset(
            [_make_spec("s3", 2)],
            (2,),
            lock,
            spectrum_schema,
            str(explicit_dir),
        )
        assert count1 == 2
        assert count2 == 1
        target_ds = explicit_dir / "spectra" / "spectra_charge_2.lance"
        assert target_ds.exists()
        assert lance.dataset(str(target_ds)).count_rows() == 3
        assert not (other_dir / "spectra" / "spectra_charge_2.lance").exists()

    def test_works_when_global_config_uninitialized(
        self, tmp_path, reset_global_config, spectrum_schema
    ):
        reset_global_config._namespace = None
        (tmp_path / "spectra").mkdir()
        lock = threading.Lock()
        count = spectra_io._write_to_dataset(
            [_make_spec("s1", 2)],
            (2,),
            lock,
            spectrum_schema,
            str(tmp_path),
        )
        assert count == 1
        assert (tmp_path / "spectra" / "spectra_charge_2.lance").exists()


# ----------------------------------------------------------------------
# Charge routing and batch flushing
# ----------------------------------------------------------------------


class TestChargeRouting:
    def test_explicit_single_multi_unknown_and_other_routing(
        self, configured_work_dir, spectrum_schema
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
            work_dir=configured_work_dir,
            schema=spectrum_schema,
        )
        assert counts == {
            "spectra_charge_1.lance": 1,
            "spectra_charge_2_3.lance": 2,
            "spectra_charge_unknown.lance": 1,
            "spectra_charge_other.lance": 1,
        }

    def test_unmapped_charge_dropped_without_other(
        self, configured_work_dir, spectrum_schema
    ):
        counts = _run_write_worker(
            [_make_spec("x", 3), _make_spec("u", None)],
            charge_to_bucket={1: (1,)},
            catch_other_charges=False,
            work_dir=configured_work_dir,
            schema=spectrum_schema,
        )
        assert counts == {}

    def test_auto_bucket_isolates_all_charges_including_negative_and_unknown(
        self, configured_work_dir, spectrum_schema
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
            work_dir=configured_work_dir,
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
        self, configured_work_dir, spectrum_schema, monkeypatch
    ):
        """Mid-stream flushes at batch size plus final sentinel."""
        monkeypatch.setattr(spectra_io, "_LANCE_WRITE_BATCH_SIZE", 3)
        # Charge 2: 6 rows (2 full batches, empty on sentinel).
        # Charge 1: 4 rows (1 batch + 1 remainder on sentinel).
        spectra = [
            *(_make_spec(f"c2_{i}", 2) for i in range(6)),
            *(_make_spec(f"c1_{i}", 1) for i in range(4)),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=configured_work_dir,
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
        self, configured_work_dir, spectrum_schema
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
            work_dir=configured_work_dir,
            schema=spectrum_schema,
            auto_bucket=True,
        )
        spectra_dir = configured_work_dir / "spectra"
        # Non-bucket artifacts must be ignored.
        (spectra_dir / "charges.joblib").write_bytes(b"placeholder")
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
        self, configured_work_dir, mgf_writer
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
            configured_work_dir / "sample.mgf",
            [
                format_mgf_spectrum(peaks, 500.0, charge=2, scan=1),
                format_mgf_spectrum(peaks, 500.0, charge=4, scan=2),
            ],
        )
        spectra_io.config.input_filenames = [str(mgf)]
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )

        # Bucket {1} has no spectra and should be skipped; charge 4 goes
        # to "other".
        valid = spectra_io._prepare_spectra(proc, [{1}, {2, 3}, "other"])
        assert valid == [{2, 3}, "other"]

        # Overlapping charges passed programmatically raise ValueError.
        with pytest.raises(ValueError, match="more than one bucket"):
            spectra_io._prepare_spectra(proc, [{1, 2}, {2, 3}])

        # Duplicate "other" passed programmatically raises ValueError.
        with pytest.raises(ValueError, match="other"):
            spectra_io._prepare_spectra(proc, [{1}, "other", "other"])

    def test_prepare_spectra_unmatched_glob_raises_file_not_found(
        self, configured_work_dir
    ):
        spectra_io.config.input_filenames = [
            str(configured_work_dir / "no_match_*.mgf")
        ]
        proc = functools.partial(
            spectrum.process_spectrum, min_peaks=5, min_mz_range=250.0
        )
        with pytest.raises(FileNotFoundError, match="No input peak files"):
            spectra_io._prepare_spectra(proc, None)


class TestWriteClusterInfo:
    def test_writes_metadata_comments_and_parseable_csv(
        self, configured_work_dir
    ):
        df = pd.DataFrame(
            {
                "identifier": ["f:scan:1", "f:scan:2"],
                "precursor_charge": [2, 2],
                "precursor_mz": [500.0, 500.05],
                "retention_time": [10.0, 11.0],
                "cluster": [0, 0],
            }
        )
        spectra_io._write_cluster_info(df)
        csv_path = configured_work_dir / "out.csv"
        raw_lines = csv_path.read_text().splitlines()
        assert raw_lines[0].startswith("# falcon version ")
        assert any(
            line.startswith("# linkage = complete") for line in raw_lines
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
