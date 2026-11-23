"""Spectrum clustering, distances, refinement, and consensus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import generate_clusters, generate_clusters_multi

__all__ = ["generate_clusters", "generate_clusters_multi"]


def __getattr__(name: str) -> Any:
    """Lazily expose top-level clustering entry points."""
    if name in {"generate_clusters", "generate_clusters_multi"}:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
