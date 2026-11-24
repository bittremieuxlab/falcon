"""Shared pytest fixtures for the Falcon test suite.

Keeps heavy scientific dependencies (``lance``, ``numpy``, ``pyarrow``)
out of module-level imports so pure-stdlib unit tests start quickly
without polluting ``sys.modules``.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path

import pytest

from tests.helpers import (
    format_mgf_spectrum,
    make_consensus_record,
    make_da_clustering,
    make_falcon_config,
    make_spectrum_row,
    make_spectrum_tuple,
    write_mgf,
    write_toy_mzml,
    write_toy_mzxml,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def isolated_root_logger():
    """Preserve and restore root and third-party logging state."""
    root = logging.getLogger()
    numba_logger = logging.getLogger("numba")
    prev_handlers = list(root.handlers)
    prev_root_level = root.level
    prev_numba_level = numba_logger.level
    prev_showwarning = warnings.showwarning
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in prev_handlers:
        root.addHandler(handler)
    root.setLevel(prev_root_level)
    numba_logger.setLevel(prev_numba_level)
    warnings.showwarning = prev_showwarning


@pytest.fixture(scope="session")
def data_dir():
    """Return path to checked-in peak-file fixtures."""
    return DATA_DIR


@pytest.fixture(scope="session")
def spec_factory():
    """Expose ``make_spectrum_tuple`` as a session fixture."""
    return make_spectrum_tuple


@pytest.fixture(scope="session")
def consensus_factory():
    """Expose ``make_consensus_record`` as a session fixture."""
    return make_consensus_record


@pytest.fixture(scope="session")
def config_factory():
    """Expose ``make_falcon_config`` as a session fixture."""
    return make_falcon_config


@pytest.fixture(scope="session")
def da_clustering_factory():
    """Expose ``make_da_clustering`` as a session fixture."""
    return make_da_clustering


@pytest.fixture(scope="session")
def spectrum_row():
    """Expose ``make_spectrum_row`` as a session fixture."""
    return make_spectrum_row


@pytest.fixture(scope="session")
def mgf_block():
    """Expose ``format_mgf_spectrum`` as a session fixture."""
    return format_mgf_spectrum


@pytest.fixture(scope="session")
def mgf_writer():
    """Expose ``write_mgf`` as a session fixture."""
    return write_mgf


@pytest.fixture(scope="session")
def mzml_writer():
    """Expose ``write_toy_mzml`` as a session fixture."""
    return write_toy_mzml


@pytest.fixture(scope="session")
def mzxml_writer():
    """Expose ``write_toy_mzxml`` as a session fixture."""
    return write_toy_mzxml


@pytest.fixture
def lance_dataset(tmp_path):
    """Factory fixture writing rows to a collision-free Lance dataset."""
    counter = 0

    def _make(rows, charge=2, name=None):
        nonlocal counter
        import lance
        import pyarrow as pa

        from falcon.staging import SPECTRUM_SCHEMA, bucket_dataset_name

        filename = name if name is not None else bucket_dataset_name((charge,))
        target = tmp_path / filename
        if target.exists():
            counter += 1
            target = tmp_path / f"ds_{counter}" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        return lance.write_dataset(
            pa.Table.from_pylist(rows, SPECTRUM_SCHEMA),
            str(target),
            mode="overwrite",
            data_storage_version="stable",
        )

    return _make


@pytest.fixture
def track_mkdtemp(monkeypatch):
    """Track directories allocated via ``tempfile.mkdtemp``."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)
    return created
