"""Hierarchical TOML loading and deep-merging for Falcon configs."""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import resolve_path
from .models import (
    AverageConsensusConfig,
    ConfigError,
    FalconConfig,
    FalconSettings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "load_config",
    "load_settings",
    "load_toml_file",
    "merge_config_dicts",
]

_AVERAGE_ONLY_KEYS: frozenset[str] = frozenset(
    AverageConsensusConfig.model_fields
) - {"method"}


def _copy_config_value(value: Any) -> Any:
    """Deep-copy config values, converting arbitrary ``Mapping``s."""
    if isinstance(value, Mapping):
        return {k: _copy_config_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    top_level: bool,
) -> dict[str, Any]:
    """Recursively merge config mappings with variant reset logic."""
    merged: dict[str, Any] = {}
    for key, existing in base.items():
        if key not in override:
            merged[key] = _copy_config_value(existing)
            continue
        val = override[key]
        if isinstance(existing, Mapping) and isinstance(val, Mapping):
            if (
                top_level
                and key == "consensus"
                and val.get("method") == "medoid"
                and existing.get("method") in (None, "average")
            ):
                cleaned = {
                    k: v
                    for k, v in existing.items()
                    if k not in _AVERAGE_ONLY_KEYS
                }
                merged[key] = _merge_config_dicts(
                    cleaned, val, top_level=False
                )
            else:
                merged[key] = _merge_config_dicts(
                    existing, val, top_level=False
                )
        else:
            merged[key] = _copy_config_value(val)
    for key, val in override.items():
        if key not in base:
            merged[key] = _copy_config_value(val)
    return merged


def merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two config mappings without mutating inputs.

    Each leaf value is deep-copied at most once. Lists and scalar values
    replace lower-layer entries atomically. When ``override`` switches
    ``consensus.method`` to a different discriminated variant than
    ``base`` at the top level (or selects ``"medoid"`` over a partial
    ``[consensus]`` table containing average-only keys), stale
    variant-specific keys from ``base["consensus"]`` are discarded.

    Parameters
    ----------
    base : Mapping[str, Any]
        Lower-precedence configuration mapping.
    override : Mapping[str, Any]
        Higher-precedence sparse override mapping.

    Returns
    -------
    dict[str, Any]
        Freshly deep-copied merged configuration dictionary.
    """
    return _merge_config_dicts(base, override, top_level=True)


def _parse_extends_entries(raw_extends: Any, source: Path) -> list[str]:
    """Validate and normalize a TOML ``extends`` value into paths."""
    if isinstance(raw_extends, str):
        if not raw_extends.strip():
            raise ConfigError(
                f"Invalid 'extends' in {source}: path string must not be empty"
            )
        return [raw_extends]
    if isinstance(raw_extends, list):
        if len(raw_extends) == 0:
            raise ConfigError(
                f"Invalid 'extends' in {source}: list must not be empty"
            )
        entries: list[str] = []
        for item in raw_extends:
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(
                    f"Invalid 'extends' entry {item!r} in {source}: "
                    "expected non-empty path string"
                )
            entries.append(item)
        return entries
    raise ConfigError(
        f"Invalid 'extends' in {source}: expected a path string or list of "
        f"path strings, got {type(raw_extends).__name__}"
    )


def _load_toml_file(
    path: Path,
    *,
    stack: tuple[Path, ...],
    referenced_by: Path | None,
) -> dict[str, Any]:
    """Recursively load a TOML file with canonical cycle detection."""
    origin = (
        f" (referenced by {referenced_by})"
        if referenced_by is not None
        else ""
    )
    try:
        resolved = resolve_path(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(
            f"Failed to resolve configuration file {path}{origin}: {exc}"
        ) from exc
    if resolved in stack:
        cycle_start = stack.index(resolved)
        cycle_trace = " -> ".join(
            str(p) for p in (*stack[cycle_start:], resolved)
        )
        raise ConfigError(f"Cycle detected in config 'extends': {cycle_trace}")

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}{origin}")
    if path.is_dir():
        raise ConfigError(
            f"Configuration path is a directory, expected a TOML file: "
            f"{path}{origin}"
        )

    try:
        raw_text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            f"Failed to read configuration file {path}{origin}: {exc}"
        ) from exc

    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Invalid TOML syntax in {path}{origin}: {exc}"
        ) from exc

    if "extends" not in data:
        return data

    extends_entries = _parse_extends_entries(data.pop("extends"), path)
    next_stack = (*stack, resolved)
    merged: dict[str, Any] = {}
    for entry in extends_entries:
        parent_path = Path(entry)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent_data = _load_toml_file(
            parent_path,
            stack=next_stack,
            referenced_by=path,
        )
        merged = merge_config_dicts(merged, parent_data)

    return merge_config_dicts(merged, data)


def load_toml_file(path: str | Path) -> dict[str, Any]:
    """
    Load a TOML configuration file, resolving ``extends`` recursively.

    Relative ``extends`` paths resolve against the declaring file's
    resolved parent directory. Cycle detection tracks the active
    resolution stack so valid diamond inheritance succeeds while true
    cycles raise ``ConfigError``.

    Parameters
    ----------
    path : str | Path
        Path to the TOML configuration file.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary with ``extends`` consumed.

    Raises
    ------
    ConfigError
        If the file is missing, is a directory, cannot be decoded as
        UTF-8 TOML, has invalid ``extends`` entries, or forms a cycle.
    """
    if not str(path).strip():
        raise ConfigError(
            "Configuration file path must not be empty or whitespace"
        )
    return _load_toml_file(Path(path), stack=(), referenced_by=None)


def _resolve_layers(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit TOML files left-to-right, then apply overrides."""
    if isinstance(config_files, (str, bytes, os.PathLike)):
        raise ConfigError(
            "config_files must be a sequence of paths (e.g. [path]), "
            f"not a single {type(config_files).__name__}"
        )
    merged: dict[str, Any] = {}
    for cfg_file in config_files:
        layer = load_toml_file(cfg_file)
        merged = merge_config_dicts(merged, layer)
    if overrides is not None:
        merged = merge_config_dicts(merged, overrides)
    # Apply default consensus method after merging so extends and
    # repeatable -c handle partial [consensus] tables identically.
    consensus = merged.get("consensus")
    if isinstance(consensus, dict) and "method" not in consensus:
        consensus["method"] = "medoid"
    return merged


def load_settings(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> FalconSettings:
    """
    Load and validate Falcon settings from TOML layers and overrides.

    Parameters
    ----------
    config_files : Sequence[str | Path], default=()
        Explicit TOML files applied left-to-right (later files override
        earlier ones). No file in the working directory is auto-loaded.
    overrides : Mapping[str, Any] | None, default=None
        Highest-precedence sparse override dictionary (e.g., from CLI).

    Returns
    -------
    FalconSettings
        Validated immutable settings (run I/O paths may be omitted).
    """
    merged = _resolve_layers(config_files=config_files, overrides=overrides)
    return FalconSettings.model_validate(merged)


def load_config(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> FalconConfig:
    """
    Load and validate a complete runnable ``FalconConfig``.

    Parameters
    ----------
    config_files : Sequence[str | Path], default=()
        Explicit TOML files applied left-to-right.
    overrides : Mapping[str, Any] | None, default=None
        Highest-precedence sparse override dictionary (e.g., from CLI).

    Returns
    -------
    FalconConfig
        Validated immutable run configuration with required I/O paths.
    """
    merged = _resolve_layers(config_files=config_files, overrides=overrides)
    merged.setdefault("io", {})
    return FalconConfig.model_validate(merged)
