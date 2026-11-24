"""Large-scale tandem mass spectrum clustering."""

from __future__ import annotations

import importlib
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


_LAZY_EXPORTS: dict[str, str] = {
    "FalconConfig": ".config",
    "FalconSettings": ".config",
    "load_config": ".config",
    "load_settings": ".config",
    "run": ".pipeline",
}


def __getattr__(name: str) -> Any:
    """Lazily expose configuration and pipeline APIs at module level."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed public API names in ``dir(falcon)``."""
    return sorted({*globals(), *__all__})
