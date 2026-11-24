"""Spectrum clustering, distances, refinement, and consensus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import generate_clusters, generate_clusters_multi

__all__ = ["generate_clusters", "generate_clusters_multi"]

_ENGINE_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Lazily expose top-level clustering entry points."""
    if name in _ENGINE_EXPORTS:
        from . import engine

        value = getattr(engine, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy entry points in ``dir(falcon.cluster)``."""
    return sorted({*globals(), *__all__})
