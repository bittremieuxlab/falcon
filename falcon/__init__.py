"""Large-scale tandem mass spectrum clustering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

_PKG_NAME = "falcon-ms"

try:
    __version__ = version(_PKG_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

if TYPE_CHECKING:
    from .config import (
        FalconConfig,
        FalconSettings,
        load_config,
        load_settings,
    )
    from .pipeline import run

__all__ = [
    "FalconConfig",
    "FalconSettings",
    "__version__",
    "load_config",
    "load_settings",
    "run",
]


_CONFIG_EXPORTS = frozenset(
    {
        "FalconConfig",
        "FalconSettings",
        "load_config",
        "load_settings",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose configuration and pipeline APIs at module level."""
    if name in _CONFIG_EXPORTS:
        from . import config

        value = getattr(config, name)
        globals()[name] = value
        return value
    if name == "run":
        from .pipeline import run

        globals()[name] = run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed public API names in ``dir(falcon)``."""
    return sorted({*globals(), *__all__})
