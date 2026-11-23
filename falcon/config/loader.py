"""Hierarchical TOML loading and deep-merging for Falcon configs."""

from __future__ import annotations

import copy
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..utils import resolve_path
from .models import ConfigError, FalconConfig, FalconSettings


def merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two config mappings without mutating inputs.

    Each leaf value is deep-copied at most once. Lists and scalar values
    replace lower-layer entries atomically. When ``override`` switches
    ``consensus.method`` to a different discriminated variant than
    ``base``, stale variant-specific keys from ``base["consensus"]`` are
    discarded.

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
    merged: dict[str, Any] = {}
    for key, existing in base.items():
        if key not in override:
            merged[key] = copy.deepcopy(existing)
            continue
        val = override[key]
        if isinstance(existing, Mapping) and isinstance(val, Mapping):
            if (
                key == "consensus"
                and "method" in val
                and "method" in existing
                and val["method"] != existing["method"]
            ):
                merged[key] = copy.deepcopy(dict(val))
            else:
                merged[key] = merge_config_dicts(existing, val)
        else:
            merged[key] = copy.deepcopy(val)
    for key, val in override.items():
        if key not in base:
            merged[key] = copy.deepcopy(val)
    return merged


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


def load_toml_file(
    path: Path,
    *,
    _stack: tuple[Path, ...] = (),
    _referenced_by: Path | None = None,
) -> dict[str, Any]:
    """
    Load a TOML configuration file, resolving ``extends`` recursively.

    Relative ``extends`` paths resolve against the declaring file's
    parent directory. Cycle detection tracks the active resolution
    stack so valid diamond inheritance succeeds while true cycles raise
    ``ConfigError``.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.
    _stack : tuple[Path, ...], optional
        Internal active canonical path stack for cycle detection.
    _referenced_by : Path | None, optional
        Internal declaring file path for clearer error messages.

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
    origin = (
        f" (referenced by {_referenced_by})"
        if _referenced_by is not None
        else ""
    )
    try:
        resolved = resolve_path(path)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Failed to resolve configuration file {path}{origin}: {exc}"
        ) from exc
    if resolved in _stack:
        cycle_start = _stack.index(resolved)
        cycle_trace = " -> ".join(
            str(p) for p in (*_stack[cycle_start:], resolved)
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

    merged: dict[str, Any] = {}
    if "extends" in data:
        extends_entries = _parse_extends_entries(data.pop("extends"), path)
        next_stack = (*_stack, resolved)
        for entry in extends_entries:
            parent_path = Path(entry)
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            parent_data = load_toml_file(
                parent_path,
                _stack=next_stack,
                _referenced_by=path,
            )
            merged = merge_config_dicts(merged, parent_data)

    return merge_config_dicts(merged, data)


def _resolve_layers(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge explicit TOML files left-to-right, then apply overrides."""
    # Seed the default consensus discriminator so partial consensus
    # overrides (or empty [consensus] tables) resolve against medoid
    # rather than failing with a missing-discriminator error.
    merged: dict[str, Any] = {"consensus": {"method": "medoid"}}
    for cfg_file in config_files:
        layer = load_toml_file(Path(cfg_file))
        merged = merge_config_dicts(merged, layer)
    if overrides is not None:
        merged = merge_config_dicts(merged, overrides)
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
    return FalconConfig.model_validate(merged)
