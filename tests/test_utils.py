"""Unit tests for falcon.utils logging, path, glob, worker, and RAM code."""

import inspect
import logging
import os
import sys
from pathlib import Path

import joblib
import pytest

from falcon import utils


class TestConfigureLogger:
    def test_attaches_stderr_handler_when_root_empty(
        self, isolated_root_logger
    ):
        root = isolated_root_logger
        for handler in list(root.handlers):
            root.removeHandler(handler)

        logger = utils.configure_logger()

        assert logger.name == "falcon"
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr
        assert handler.level == logging.DEBUG
        assert logging.getLogger("numba").level == logging.WARNING

    def test_does_not_duplicate_existing_handlers(self, isolated_root_logger):
        root = isolated_root_logger
        for handler in list(root.handlers):
            root.removeHandler(handler)
        existing = logging.NullHandler()
        root.addHandler(existing)

        utils.configure_logger()

        assert root.handlers == [existing]


class TestPathGlobAndTimeHelpers:
    def test_resolve_path_handles_existing_missing_and_symlink_loop(
        self, tmp_path
    ):
        existing = tmp_path / "file.txt"
        existing.write_text("ok", encoding="utf-8")
        assert utils.resolve_path(existing) == existing.resolve()

        missing = tmp_path / "does_not_exist.txt"
        assert utils.resolve_path(missing) == missing.resolve(strict=False)

        loop = tmp_path / "loop"
        loop.symlink_to(loop)
        with pytest.raises((OSError, RuntimeError)):
            utils.resolve_path(loop)

    def test_expand_input_globs_sorted_and_preserves_repeated_matches(
        self, tmp_path
    ):
        for name in ("c.mgf", "a.mgf", "b.mgf"):
            (tmp_path / name).write_text("", encoding="utf-8")

        expanded = utils.expand_input_globs(
            [str(tmp_path / "*.mgf"), str(tmp_path / "b.mgf")]
        )
        assert expanded == [
            tmp_path / "a.mgf",
            tmp_path / "b.mgf",
            tmp_path / "c.mgf",
            tmp_path / "b.mgf",
        ]

    def test_expand_input_globs_unmatched_literals_flag(self, tmp_path):
        missing_literal = tmp_path / "missing.mgf"
        missing_glob = str(tmp_path / "none_*.mgf")

        assert (
            utils.expand_input_globs(
                [str(missing_literal), missing_glob],
                keep_unmatched_literals=False,
            )
            == []
        )
        assert utils.expand_input_globs(
            [str(missing_literal), missing_glob],
            keep_unmatched_literals=True,
        ) == [missing_literal]

    def test_expand_input_globs_retains_existing_literal_with_brackets(
        self, tmp_path
    ):
        bracketed = tmp_path / "sample_[1].mgf"
        bracketed.write_text("", encoding="utf-8")

        assert utils.expand_input_globs([str(bracketed)]) == [bracketed]

    def test_paths_alias_covers_symlinks_hardlinks_and_loops(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "real.mgf"
        target.write_text("MGF\n", encoding="utf-8")
        sym = tmp_path / "sym.mgf"
        sym.symlink_to(target)
        hard = tmp_path / "hard.mgf"
        os.link(target, hard)
        other = tmp_path / "other.mgf"
        other.write_text("OTHER\n", encoding="utf-8")
        loop = tmp_path / "loop.mgf"
        loop.symlink_to(loop)

        assert utils.paths_alias(target, sym)
        assert utils.paths_alias(target, hard)
        assert not utils.paths_alias(target, other)
        assert not utils.paths_alias(loop, target)
        assert not utils.paths_alias(target, loop)
        assert not utils.paths_alias(tmp_path / "bad\x00a", target)
        assert not utils.paths_alias(target, tmp_path / "bad\x00b")

        def _fail_samefile(_a, _b):
            raise OSError("vanished")

        monkeypatch.setattr(os.path, "samefile", _fail_samefile)
        assert not utils.paths_alias(target, other)

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (-10.5, "0:00:00"),
            (-0.1, "0:00:00"),
            (float("nan"), "0:00:00"),
            (float("inf"), "0:00:00"),
            (float("-inf"), "0:00:00"),
            (0, "0:00:00"),
            (59.9, "0:00:59"),
            (3661, "1:01:01"),
        ],
    )
    def test_format_hms(self, seconds, expected):
        assert utils.format_hms(seconds) == expected

    def test_effective_n_workers_returns_positive_int(self, monkeypatch):
        workers = utils.effective_n_workers()
        assert isinstance(workers, int)
        assert workers >= 1

        monkeypatch.setattr(joblib, "effective_n_jobs", lambda _n: 0)
        assert utils.effective_n_workers() == 1

    def test_find_path_alias_matches_symlinks_hardlinks_and_nonexistent(
        self, tmp_path
    ):
        real = tmp_path / "real.mgf"
        real.write_text("MGF\n", encoding="utf-8")
        sym = tmp_path / "sym.mgf"
        sym.symlink_to(real)
        hard = tmp_path / "hard.mgf"
        os.link(real, hard)
        other = tmp_path / "other.mgf"
        other.write_text("OTHER\n", encoding="utf-8")
        loop = tmp_path / "loop.mgf"
        loop.symlink_to(loop)

        # Returns first matching candidate, skipping broken/loop entries.
        assert (
            utils.find_path_alias(
                real, [other, loop, tmp_path / "x\x00y", sym, hard]
            )
            == sym
        )
        assert utils.find_path_alias(real, [other, hard]) == hard
        assert utils.find_path_alias(real, [other]) is None
        assert utils.find_path_alias(real, []) is None

        # Pre-resolved target works without re-resolving.
        assert (
            utils.find_path_alias(
                real, [other, sym], resolved_target=real.resolve()
            )
            == sym
        )

        # Non-existent target matches non-existent canonical alias.
        missing = tmp_path / "missing.mgf"
        sub = tmp_path / "sub"
        sub.mkdir()
        rel_missing = sub / ".." / "missing.mgf"
        assert (
            utils.find_path_alias(missing, [other, rel_missing]) == rel_missing
        )
        assert utils.find_path_alias(missing, [other]) is None

    def test_find_path_alias_strict_and_non_strict_errors(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "target.mgf"
        target.write_text("ok\n", encoding="utf-8")
        loop = tmp_path / "loop.mgf"
        loop.symlink_to(loop)
        nul_path = tmp_path / "bad\x00path.mgf"

        assert utils.find_path_alias(loop, [target], strict=False) is None
        with pytest.raises((OSError, RuntimeError)):
            utils.find_path_alias(loop, [target], strict=True)

        assert utils.find_path_alias(nul_path, [target], strict=False) is None
        with pytest.raises(ValueError, match="null"):
            utils.find_path_alias(nul_path, [target], strict=True)

        # OSError from target.exists() falls back to target_exists=False.
        def _fail_exists(self_path):
            raise OSError(f"cannot stat {self_path}")

        monkeypatch.setattr(Path, "exists", _fail_exists)
        assert utils.find_path_alias(target, [target]) == target
        assert utils.find_path_alias(target, [tmp_path / "other.mgf"]) is None

    def test_loky_unordered_pool_config_and_clamping(self):
        pool_3 = utils.loky_unordered_pool(3)
        assert isinstance(pool_3, joblib.Parallel)
        assert pool_3.n_jobs == 3
        assert pool_3.return_as == "generator_unordered"
        assert pool_3.verbose == 0
        assert type(pool_3._backend).__name__ == "LokyBackend"

        for requested in (0, -4):
            clamped = utils.loky_unordered_pool(requested)
            assert clamped.n_jobs == 1

        gen = utils.loky_unordered_pool(1)(
            joblib.delayed(abs)(v) for v in (-3, -1, -2)
        )
        assert inspect.isgenerator(gen)
        assert sorted(gen) == [1, 2, 3]


class TestMemoryAndCgroupLimits:
    def test_cgroup_memory_limit_fallback_paths(self, tmp_path, monkeypatch):
        missing = tmp_path / "missing"
        cg_max = tmp_path / "memory.max"
        cg_zero = tmp_path / "memory.zero"
        cg_num = tmp_path / "memory.num"
        cg_max.write_text("max\n", encoding="ascii")
        cg_zero.write_text("0\n", encoding="ascii")
        cg_num.write_text("1048576\n", encoding="ascii")

        # Fallback paths when /proc/self/cgroup is unreadable.
        monkeypatch.setattr(utils, "_PROC_CGROUP_PATH", str(missing))
        monkeypatch.setattr(
            utils, "_CGROUP_MEMORY_PATHS", (str(missing), str(cg_max))
        )
        assert utils._cgroup_memory_limit() is None

        monkeypatch.setattr(
            utils,
            "_CGROUP_MEMORY_PATHS",
            (str(missing), str(cg_max), str(cg_zero), str(cg_num)),
        )
        assert utils._cgroup_memory_limit() == 1048576

    def test_cgroup_memory_limit_v2_v1_hierarchy_and_dedup(
        self, tmp_path, monkeypatch
    ):
        # Non-root cgroup v2 & v1 traversal (e.g. Slurm / systemd) picks
        # the minimum finite limit across ancestors and hierarchies.
        v2_root = tmp_path / "cgv2"
        v2_job = v2_root / "system.slice" / "job_42"
        v2_step = v2_job / "step_0"
        v2_step.mkdir(parents=True)
        (v2_root / "memory.max").write_text("max\n", encoding="ascii")
        (v2_job / "memory.max").write_text("8388608\n", encoding="ascii")
        (v2_step / "memory.max").write_text("max\n", encoding="ascii")

        v1_root = tmp_path / "cgv1_mem"
        v1_job = v1_root / "slurm" / "job_42"
        v1_job.mkdir(parents=True)
        (v1_root / "memory.limit_in_bytes").write_text(
            "9223372036854771712\n", encoding="ascii"
        )
        (v1_job / "memory.limit_in_bytes").write_text(
            "4194304\n", encoding="ascii"
        )

        proc_cg = tmp_path / "proc_cgroup"
        proc_cg.write_text(
            "malformed_line\n2:cpu:/ignored\n0::/system.slice/job_42/step_0\n",
            encoding="ascii",
        )
        monkeypatch.setattr(utils, "_PROC_CGROUP_PATH", str(proc_cg))
        monkeypatch.setattr(utils, "_CGROUP_V2_ROOT", str(v2_root))
        monkeypatch.setattr(utils, "_CGROUP_V1_MEMORY_ROOT", str(v1_root))
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_PATHS", ())
        assert utils._cgroup_memory_limit() == 8388608

        # Duplicate v2 root in fallback list is not re-read, and an
        # unseen fallback path still contributes.
        read_paths: list[Path] = []
        real_read = utils._read_positive_int

        def _tracking_read(path: Path) -> int | None:
            read_paths.append(path)
            return real_read(path)

        extra_fb = tmp_path / "extra_limit"
        extra_fb.write_text("2097152\n", encoding="ascii")
        monkeypatch.setattr(utils, "_read_positive_int", _tracking_read)
        monkeypatch.setattr(
            utils,
            "_CGROUP_MEMORY_PATHS",
            (str(v2_root / "memory.max"), str(extra_fb)),
        )
        proc_cg.write_text(
            "0::/system.slice/job_42/step_0\n"
            "0::/system.slice/job_42\n"
            "4:cpu,memory:/slurm/job_42\n",
            encoding="ascii",
        )
        assert utils._cgroup_memory_limit() == 2097152
        assert read_paths.count(v2_root / "memory.max") == 1
        assert read_paths.count(v2_job / "memory.max") == 1

        monkeypatch.setattr(utils, "_read_positive_int", real_read)
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_PATHS", ())
        assert utils._cgroup_memory_limit() == 4194304

        # Non-ASCII bytes in /proc/self/cgroup or a limit file never raise.
        (v2_step / "memory.max").write_bytes(b"\xff\xfe\n")
        proc_cg.write_bytes(
            b"3:cpu:/caf\xc3\xa9\n0::/system.slice/job_42/step_0\n"
        )
        assert utils._cgroup_memory_limit() == 8388608

        # Relative '.' / '..' components never escape cgroup root.
        proc_cg.write_text(
            "0::/../../system.slice/./job_42/step_0\n", encoding="ascii"
        )
        assert utils._cgroup_memory_limit() == 8388608

    def test_effective_memory_bytes_clamping_and_sysconf_errors(
        self, monkeypatch
    ):
        monkeypatch.setattr(utils, "_cgroup_memory_limit", lambda: None)
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 4096 if name == "SC_PAGE_SIZE" else 1000,
        )
        assert utils.effective_memory_bytes() == 4_096_000

        # Clamped when cgroup limit is below host RAM.
        monkeypatch.setattr(utils, "_cgroup_memory_limit", lambda: 1_000_000)
        assert utils.effective_memory_bytes() == 1_000_000

        # Host RAM retained when cgroup limit exceeds host RAM.
        monkeypatch.setattr(utils, "_cgroup_memory_limit", lambda: 9_999_999)
        assert utils.effective_memory_bytes() == 4_096_000

        # Non-positive sysconf values return None.
        for bad_page, bad_phys in (
            (0, 1000),
            (-1, 1000),
            (4096, 0),
            (4096, -1),
        ):
            monkeypatch.setattr(
                os,
                "sysconf",
                lambda name, p=bad_page, q=bad_phys: (
                    p if name == "SC_PAGE_SIZE" else q
                ),
            )
            assert utils.effective_memory_bytes() is None

        # Unsupported or failing sysconf returns None.
        for exc_cls in (ValueError, OSError, AttributeError):

            def _fail(_name, _exc=exc_cls):
                raise _exc("unsupported")

            monkeypatch.setattr(os, "sysconf", _fail)
            assert utils.effective_memory_bytes() is None


class TestAtomicWriteText:
    def test_writes_atomically_and_cleans_temp_file_on_error(self, tmp_path):
        target = tmp_path / "result.txt"
        with utils.atomic_write_text(target, prefix=".test_") as f_out:
            f_out.write("hello\n")
            assert not target.exists()
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert list(tmp_path.glob(".test_*.tmp")) == []

        def _fail_write() -> None:
            with utils.atomic_write_text(target, prefix=".test_") as f_out:
                f_out.write("partial")
                raise RuntimeError("write failed")

        with pytest.raises(RuntimeError, match="write failed"):
            _fail_write()
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert list(tmp_path.glob(".test_*.tmp")) == []


class TestRequireRegularFile:
    def test_accepts_regular_file_and_rejects_dir_missing_fifo_and_loop(
        self, tmp_path
    ):
        regular = tmp_path / "ok.mgf"
        regular.write_text("data", encoding="utf-8")
        utils.require_regular_file(
            regular, dir_message="is dir", missing_message="is missing"
        )

        with pytest.raises(IsADirectoryError, match="is dir"):
            utils.require_regular_file(
                tmp_path, dir_message="is dir", missing_message="is missing"
            )

        with pytest.raises(FileNotFoundError, match="is missing"):
            utils.require_regular_file(
                tmp_path / "missing.mgf",
                dir_message="is dir",
                missing_message="is missing",
            )

        fifo = tmp_path / "stream.mgf"
        os.mkfifo(fifo)
        with pytest.raises(FileNotFoundError, match="is missing"):
            utils.require_regular_file(
                fifo, dir_message="is dir", missing_message="is missing"
            )

        loop = tmp_path / "loop.mgf"
        loop.symlink_to(loop)
        with pytest.raises(OSError, match=r".+") as exc_info:
            utils.require_regular_file(
                loop, dir_message="is dir", missing_message="is missing"
            )
        assert not isinstance(exc_info.value, FileNotFoundError)
