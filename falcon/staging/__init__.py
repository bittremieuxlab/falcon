"""Input discovery, charge routing, Lance staging, and cache provenance.

This package owns input peak-file expansion, precursor charge routing,
parallel spectrum reading and batch-routed Lance writes, and
``staging_cache.json`` manifest validation. Shared Lance schemas and
bucket naming conventions live in :mod:`falcon.dataset`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..utils import resolve_input_files
    from .cache import (
        STAGING_CACHE_FILENAME,
        STAGING_CACHE_SCHEMA_VERSION,
        build_staging_manifest,
        deserialize_cached_buckets,
        diff_staging_manifests,
        load_or_stage_buckets,
        serialize_cached_buckets,
    )
    from .ingest import (
        ChargeRouter,
        stage_spectra,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "STAGING_CACHE_FILENAME": ".cache",
    "STAGING_CACHE_SCHEMA_VERSION": ".cache",
    "build_staging_manifest": ".cache",
    "deserialize_cached_buckets": ".cache",
    "diff_staging_manifests": ".cache",
    "load_or_stage_buckets": ".cache",
    "serialize_cached_buckets": ".cache",
    "ChargeRouter": ".ingest",
    "stage_spectra": ".ingest",
    "resolve_input_files": "..utils",
}

__all__ = [
    "STAGING_CACHE_FILENAME",
    "STAGING_CACHE_SCHEMA_VERSION",
    "ChargeRouter",
    "build_staging_manifest",
    "deserialize_cached_buckets",
    "diff_staging_manifests",
    "load_or_stage_buckets",
    "resolve_input_files",
    "serialize_cached_buckets",
    "stage_spectra",
]


def __getattr__(name: str) -> Any:
    """Lazily expose staging cache, input, and ingestion APIs."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed API names in ``dir(falcon.staging)``."""
    return sorted({*globals(), *__all__})
