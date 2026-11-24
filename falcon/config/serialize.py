"""Deterministic TOML, metadata, JSON Schema, and config export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..types import INT8_MAX, INT8_MIN, NONE_SENTINEL
from ..utils import expand_input_globs, paths_alias, resolve_path
from .models import (
    ConfigError,
    FalconSettings,
    PrecursorTolerance,
    _AllowSentinelSchema,
    validate_file_path,
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from .models import FalconConfig

__all__ = [
    "build_json_schema",
    "format_metadata_lines",
    "format_toml",
    "write_saved_config",
]

_TOML_ESCAPES: dict[str, str] = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}

_IO_CUSTOM_FIELDS: frozenset[str] = frozenset(
    {"input_filenames", "output_filename", "work_dir"}
)
_CLUSTERING_NESTED_FIELDS: frozenset[str] = frozenset({"precursor_tol"})


def _toml_string(value: str) -> str:
    """Encode ``value`` as a valid TOML 1.0 basic string literal."""
    parts: list[str] = ['"']
    for ch in value:
        escaped = _TOML_ESCAPES.get(ch)
        if escaped is not None:
            parts.append(escaped)
            continue
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise ConfigError(
                f"Cannot serialize lone surrogate code point U+{code:04X} "
                "to TOML."
            )
        if code < 0x20 or code == 0x7F:
            parts.append(f"\\u{code:04x}")
        else:
            parts.append(ch)
    parts.append('"')
    return "".join(parts)


def _toml_literal(value: Any) -> str:
    """Format a validated scalar or nested list as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (Path, str)):
        return _toml_string(str(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _format_optional_toml(
    key: str,
    value: Any,
    sentinel: str,
    *,
    comment_when_none: bool,
) -> str:
    """
    Format an optional TOML field with round-trip-safe sentinel output.

    Fields whose model default is non-``None`` (``remove_precursor_tol``
    and ``max_peaks_used``) must emit an uncommented sentinel when
    disabled (``comment_when_none=False``), because omitting them would
    revert to the non-``None`` default upon reload. Fields whose default
    is ``None`` (``work_dir``, ``rt_tol``, ``precursor_charge_buckets``)
    emit commented sentinels (``comment_when_none=True``) while
    preserving exact round-trip fidelity.
    """
    if value is not None:
        return f"{key} = {_toml_literal(value)}"
    prefix = "# " if comment_when_none else ""
    return f"{prefix}{key} = {_toml_literal(sentinel)}"


def _sentinel_for_field(field_info: FieldInfo) -> str | None:
    """Return the configured TOML sentinel for ``field_info``."""
    for meta in field_info.metadata:
        if isinstance(meta, _AllowSentinelSchema):
            return meta.sentinel
    return None


def _format_model_toml_lines(
    section: BaseModel,
    *,
    skip: frozenset[str] = frozenset(),
) -> list[str]:
    """Render scalar/list fields of a config section to TOML."""
    lines: list[str] = []
    for name, field_info in type(section).model_fields.items():
        if name in skip:
            continue
        value = getattr(section, name)
        sentinel = _sentinel_for_field(field_info)
        if sentinel is not None:
            lines.append(
                _format_optional_toml(
                    name,
                    value,
                    sentinel,
                    comment_when_none=field_info.default is None,
                )
            )
        else:
            lines.append(f"{name} = {_toml_literal(value)}")
    return lines


def _format_metadata_value(value: Any) -> str:
    """Format a metadata value as a single-line safe string."""
    text = str(value)
    if all(ord(ch) >= 0x20 and ord(ch) != 0x7F for ch in text):
        return text
    parts: list[str] = []
    for ch in text:
        if ch in _TOML_ESCAPES and ch not in ('"', "\\"):
            parts.append(_TOML_ESCAPES[ch])
        else:
            code = ord(ch)
            if code < 0x20 or code == 0x7F:
                parts.append(f"\\x{code:02x}")
            else:
                parts.append(ch)
    return "".join(parts)


def _format_model_metadata_lines(
    prefix: str,
    section: BaseModel,
    *,
    skip: frozenset[str] = frozenset(),
) -> list[str]:
    """Render ``section.field = value`` lines from a Pydantic model."""
    lines: list[str] = []
    for name in type(section).model_fields:
        if name in skip:
            continue
        value = getattr(section, name)
        if isinstance(value, PrecursorTolerance):
            lines.append(f"{prefix}.{name} = {value.mass} {value.mode}")
        else:
            lines.append(f"{prefix}.{name} = {value}")
    return lines


def format_toml(
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> str:
    """
    Render a validated config as canonical round-trippable TOML.

    Parameters
    ----------
    cfg : FalconSettings | FalconConfig
        Validated ``FalconSettings`` or ``FalconConfig`` instance.
    work_dir : Path | None, default=None
        Optional effective working directory override.

    Returns
    -------
    str
        Deterministic TOML text terminated by a trailing newline.
    """
    effective_work_dir = cfg.io.work_dir if work_dir is None else work_dir
    if effective_work_dir is None:
        work_dir_value: str | None = None
    elif effective_work_dir == Path(NONE_SENTINEL):
        # Preserve a literal relative directory named "none" as "./none"
        # so reloading does not interpret it as the disable sentinel.
        work_dir_value = f"./{NONE_SENTINEL}"
    else:
        work_dir_value = str(effective_work_dir)

    lines: list[str] = ["[io]"]
    if cfg.io.input_filenames is not None:
        lines.append(
            f"input_filenames = {_toml_literal(cfg.io.input_filenames)}"
        )
    else:
        lines.append("# input_filenames = []")
    if cfg.io.output_filename is not None:
        lines.append(
            f"output_filename = {_toml_literal(cfg.io.output_filename)}"
        )
    else:
        lines.append('# output_filename = ""')
    lines.append(
        _format_optional_toml(
            "work_dir", work_dir_value, NONE_SENTINEL, comment_when_none=True
        )
    )
    lines.extend(_format_model_toml_lines(cfg.io, skip=_IO_CUSTOM_FIELDS))

    lines.extend(["", "[preprocessing]"])
    lines.extend(_format_model_toml_lines(cfg.preprocessing))

    lines.extend(["", "[clustering]"])
    lines.extend(
        _format_model_toml_lines(
            cfg.clustering, skip=_CLUSTERING_NESTED_FIELDS
        )
    )

    lines.extend(["", "[clustering.precursor_tol]"])
    lines.extend(_format_model_toml_lines(cfg.clustering.precursor_tol))

    lines.extend(["", "[consensus]"])
    lines.extend(_format_model_toml_lines(cfg.consensus))

    return "\n".join(lines) + "\n"


def format_metadata_lines(
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> list[str]:
    """
    Return canonical metadata provenance lines for logs and CSV headers.

    Parameters
    ----------
    cfg : FalconSettings | FalconConfig
        Validated ``FalconSettings`` or ``FalconConfig`` instance.
    work_dir : Path | None, default=None
        Optional effective working directory override.

    Returns
    -------
    list[str]
        Version banner followed by ``section.key = value`` lines.
    """
    effective_work_dir = cfg.io.work_dir if work_dir is None else work_dir
    return [
        f"falcon version {__version__}",
        f"io.work_dir = {_format_metadata_value(effective_work_dir)}",
        *_format_model_metadata_lines("io", cfg.io, skip=_IO_CUSTOM_FIELDS),
        *_format_model_metadata_lines("preprocessing", cfg.preprocessing),
        *_format_model_metadata_lines("clustering", cfg.clustering),
        *_format_model_metadata_lines("consensus", cfg.consensus),
    ]


def _enrich_json_schema_constraints(schema: dict[str, Any]) -> None:
    """Attach runtime validator constraints to the Pydantic schema."""
    # Allow empty [consensus] tables (defaulting to medoid) by relaxing
    # oneOf -> anyOf while preserving per-variant additionalProperties.
    consensus_prop = schema.get("properties", {}).get("consensus")
    if isinstance(consensus_prop, dict) and "oneOf" in consensus_prop:
        consensus_prop["anyOf"] = consensus_prop.pop("oneOf")

    defs = schema.get("$defs", {})
    io_props = defs.get("PartialIOConfig", {}).get("properties", {})
    if "input_filenames" in io_props:
        for branch in io_props["input_filenames"].get("anyOf", ()):
            if branch.get("type") == "array":
                branch["minItems"] = 1
                branch["items"] = {"type": "string", "minLength": 1}
    for path_field in ("output_filename", "work_dir"):
        if path_field in io_props:
            for branch in io_props[path_field].get("anyOf", ()):
                if branch.get("type") == "string" and "const" not in branch:
                    branch["minLength"] = 1

    clust_props = defs.get("ClusteringConfig", {}).get("properties", {})
    buckets_schema = clust_props.get("precursor_charge_buckets", {})
    for branch in buckets_schema.get("anyOf", ()):
        if branch.get("type") != "array":
            continue
        branch["minItems"] = 1
        for item_branch in branch.get("items", {}).get("anyOf", ()):
            if item_branch.get("type") != "array":
                continue
            item_branch["minItems"] = 1
            item_branch["uniqueItems"] = True
            for elem in item_branch.get("items", {}).get("anyOf", ()):
                if elem.get("type") == "integer":
                    elem["minimum"] = INT8_MIN
                    elem["maximum"] = INT8_MAX


def build_json_schema() -> dict[str, Any]:
    """
    Build a JSON Schema for sparse hierarchical TOML configs.

    Includes optional top-level ``extends`` (single path or list of
    paths) alongside optional ``[io]`` paths, runtime validator bounds,
    and TOML string sentinels (``"none"`` and ``"auto"``) so editor
    tooling validates partial overlay files cleanly.

    Returns
    -------
    dict[str, Any]
        JSON-serializable schema dictionary.
    """
    schema = FalconSettings.model_json_schema()
    schema["properties"] = {
        "extends": {
            "title": "Extends",
            "description": (
                "Optional parent TOML configuration file path or list of "
                "paths resolved relative to the declaring file."
            ),
            "anyOf": [
                {"type": "string", "minLength": 1},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
            ],
        },
        **schema.get("properties", {}),
    }
    _enrich_json_schema_constraints(schema)
    return schema


def _validate_save_destination(
    config: FalconConfig, destination: str | Path
) -> Path:
    """Ensure ``destination`` is a file path distinct from I/O files."""
    try:
        dest = validate_file_path(destination, label="Config output path")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        resolved_dest = resolve_path(dest)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(
            f"Failed to resolve config output path {dest}: {exc}"
        ) from exc
    if dest.is_dir():
        raise ConfigError(
            f"Config output path is a directory, expected a file path: {dest}"
        )

    dest_exists = dest.exists()
    for candidate in expand_input_globs(
        config.io.input_filenames, keep_unmatched_literals=True
    ):
        if paths_alias(
            dest,
            candidate,
            resolved_target=resolved_dest,
            target_exists=dest_exists,
        ):
            raise ConfigError(
                f"Config output path {dest} aliases input file "
                f"{candidate}; choose a distinct destination"
            )
    for out_path in (config.io.csv_path, config.io.mgf_path):
        if paths_alias(
            dest,
            out_path,
            resolved_target=resolved_dest,
            target_exists=dest_exists,
        ):
            raise ConfigError(
                f"Config output path {dest} aliases output file "
                f"{out_path}; choose a distinct destination"
            )
    return dest


def write_saved_config(config: FalconConfig, destination: str | Path) -> None:
    """
    Write a resolved ``FalconConfig`` as TOML to ``destination``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration to serialize.
    destination : str | Path
        Target TOML file path (must not be a directory or alias any
        configured input or output file).

    Raises
    ------
    ConfigError
        If ``destination`` is a directory, aliases an input or output
        file, or cannot be written.
    """
    dest = _validate_save_destination(config, destination)
    try:
        dest.write_text(format_toml(config), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Failed to write configuration to {dest}: {exc}"
        ) from exc
