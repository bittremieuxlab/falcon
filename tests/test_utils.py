"""Unit tests for falcon.utils and package metadata."""

import logging
import random
import sys

import numpy as np
import pytest

import falcon
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


class TestPackageMetadata:
    def test_version_is_non_empty_string(self):
        assert isinstance(falcon.__version__, str)
        assert falcon.__version__


class TestSetSeeds:
    def test_reproduces_python_and_numpy_sequences(self):
        utils.set_seeds(123)
        py_first = [random.random() for _ in range(5)]  # noqa: S311
        np_first = np.random.rand(5)

        utils.set_seeds(123)
        py_second = [random.random() for _ in range(5)]  # noqa: S311
        np_second = np.random.rand(5)

        assert py_first == py_second
        np.testing.assert_array_equal(np_first, np_second)

    def test_different_seeds_diverge(self):
        utils.set_seeds(1)
        a = (random.random(), float(np.random.rand()))  # noqa: S311
        utils.set_seeds(2)
        b = (random.random(), float(np.random.rand()))  # noqa: S311
        assert a != b


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
