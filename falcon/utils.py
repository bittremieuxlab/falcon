"""Logging configuration and random seed helpers."""

import logging
import random
import sys

import numpy as np


def set_seeds(my_seed: int = 42) -> None:
    """
    Seed Python's and NumPy's legacy global random generators.

    Parameters
    ----------
    my_seed : int, default=42
        Random seed applied to ``random`` and ``numpy.random``.
    """
    random.seed(my_seed)
    np.random.seed(my_seed)


def configure_logger() -> logging.Logger:
    """
    Configure root and Falcon logging handlers for CLI execution.

    Attaches a debug-level stderr handler if the root logger has no
    handlers yet, enables warning capture, and suppresses verbose debug
    logs from ``numba`` and ``numexpr``.

    Returns
    -------
    logging.Logger
        The ``"falcon"`` application logger.
    """
    logger = logging.getLogger("falcon")
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "{asctime} {levelname} [{name}/{processName}] "
                "{module}.{funcName} : {message}",
                style="{",
            )
        )
        root.addHandler(handler)
    # Silence verbose third-party debug/info logs.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return logger
