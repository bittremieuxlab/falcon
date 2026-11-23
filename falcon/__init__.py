"""Large-scale tandem mass spectrum clustering."""

from importlib.metadata import PackageNotFoundError, version

pkg_name = "falcon-ms"

try:
    __version__ = version(pkg_name)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
