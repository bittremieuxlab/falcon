"""Unit tests for falcon.utils."""

import logging
import os
import sys

import pytest

from falcon import utils


class TestConfigureLogger:
    def test_attaches_stderr_handler_once(self, isolated_root_logger):
        root = isolated_root_logger
        for handler in list(root.handlers):
            root.removeHandler(handler)

        logger = utils.configure_logger()
        assert logger.name == "falcon"
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert root.handlers[0].stream is sys.stderr
        assert logging.getLogger("numba").level == logging.WARNING

        # Second call does not attach duplicate handlers.
        utils.configure_logger()
        assert len(root.handlers) == 1


class TestInputGlobsAndAtomicWrite:
    def test_resolve_input_files_sorted_globs_brackets_and_errors(
        self, tmp_path
    ):
        for name in ("c.mgf", "a.mgf", "b.mgf", "sample_[1].mgf"):
            (tmp_path / name).write_text("", encoding="utf-8")

        expanded = utils.resolve_input_files(
            [str(tmp_path / "[a-c].mgf"), str(tmp_path / "b.mgf")]
        )
        assert expanded == [
            tmp_path / "a.mgf",
            tmp_path / "b.mgf",
            tmp_path / "c.mgf",
            tmp_path / "b.mgf",
        ]
        assert utils.resolve_input_files(
            [str(tmp_path / "sample_[1].mgf")]
        ) == [tmp_path / "sample_[1].mgf"]

        with pytest.raises(FileNotFoundError, match="No input peak files"):
            utils.resolve_input_files([])
        with pytest.raises(FileNotFoundError, match="matched no files"):
            utils.resolve_input_files([str(tmp_path / "missing_*.mgf")])

        subdir = tmp_path / "nested"
        subdir.mkdir()
        with pytest.raises(IsADirectoryError, match="directory"):
            utils.resolve_input_files([str(subdir)])

        broken = tmp_path / "dangling.mgf"
        broken.symlink_to(tmp_path / "nonexistent.mgf")
        with pytest.raises(FileNotFoundError, match="regular file"):
            utils.resolve_input_files([str(broken)])

    def test_atomic_write_text_replaces_and_cleans_temp_on_error(
        self, tmp_path
    ):
        target = tmp_path / "result.txt"
        with utils.atomic_write_text(target, prefix=".test_") as f_out:
            f_out.write("hello\n")
            assert not target.exists()
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert list(tmp_path.glob(".test_*.tmp")) == []

        def _fail_write() -> None:
            with utils.atomic_write_text(target, prefix=".test_") as f_out:
                f_out.write("partial")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _fail_write()
        assert target.read_text(encoding="utf-8") == "hello\n"
        assert list(tmp_path.glob(".test_*.tmp")) == []


class TestRuntimeHelpers:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (-5.0, "0:00:00"),
            (float("nan"), "0:00:00"),
            (0, "0:00:00"),
            (59.9, "0:00:59"),
            (3661, "1:01:01"),
        ],
    )
    def test_format_hms(self, seconds, expected):
        assert utils.format_hms(seconds) == expected

    def test_effective_n_workers_and_memory_bytes(self, tmp_path, monkeypatch):
        assert utils.effective_n_workers() >= 1

        cg_limit = tmp_path / "memory.max"
        cg_limit.write_text("1048576\n", encoding="ascii")
        monkeypatch.setattr(utils, "_CGROUP_MEMORY_PATHS", (str(cg_limit),))
        monkeypatch.setattr(
            os,
            "sysconf",
            lambda name: 4096 if name == "SC_PAGE_SIZE" else 1000,
        )
        assert utils.effective_memory_bytes() == 1_048_576
