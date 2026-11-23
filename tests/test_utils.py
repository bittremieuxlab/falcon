"""Unit tests for falcon.utils logging, path, glob, and worker code."""

import logging
import sys

import joblib
import pytest

from falcon import utils


@pytest.fixture
def isolated_root_logger():
    """Preserve and restore root and third-party logging state."""
    root = logging.getLogger()
    numba_logger = logging.getLogger("numba")
    numexpr_logger = logging.getLogger("numexpr")
    prev_handlers = list(root.handlers)
    prev_root_level = root.level
    prev_numba_level = numba_logger.level
    prev_numexpr_level = numexpr_logger.level
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in prev_handlers:
        root.addHandler(handler)
    root.setLevel(prev_root_level)
    numba_logger.setLevel(prev_numba_level)
    numexpr_logger.setLevel(prev_numexpr_level)
    logging.captureWarnings(False)


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
        assert logging.getLogger("numexpr").level == logging.WARNING

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

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (-10.5, "0:00:00"),
            (-0.1, "0:00:00"),
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
