"""Input discovery, charge routing, Lance staging, and cache provenance.

This package owns input peak-file expansion, precursor charge routing,
parallel spectrum reading and batch-routed Lance writes, and
``staging_cache.json`` manifest validation. Shared Lance schemas and
bucket naming conventions live in :mod:`falcon.dataset`.
"""

from __future__ import annotations

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
    resolve_input_files,
    stage_spectra,
)

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
