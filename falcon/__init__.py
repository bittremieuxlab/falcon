"""Large-scale tandem mass spectrum clustering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

pkg_name = "falcon-ms"

try:
    __version__ = version(pkg_name)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

if TYPE_CHECKING:
    from .config import (
        FalconConfig,
        FalconSettings,
        load_config,
        load_settings,
    )
    from .falcon import run

__all__ = [
    "FalconConfig",
    "FalconSettings",
    "__version__",
    "load_config",
    "load_settings",
    "run",
]


def __getattr__(name: str) -> Any:
    """Lazily expose configuration and pipeline APIs at package level."""
    if name in {
        "FalconConfig",
        "FalconSettings",
        "load_config",
        "load_settings",
    }:
        from . import config

        return getattr(config, name)
    if name == "run":
        from .falcon import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
