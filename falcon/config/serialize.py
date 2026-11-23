"""Deterministic TOML, metadata, JSON Schema, and config export."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..types import AUTO_SENTINEL, NONE_SENTINEL
from ..utils import expand_input_globs, resolve_path
from .models import (
    AverageConsensusConfig,
    ConfigError,
    FalconSettings,
)

if TYPE_CHECKING:
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
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


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
    lines.append(f"overwrite = {_toml_literal(cfg.io.overwrite)}")
    lines.append(
        f"export_representatives = "
        f"{_toml_literal(cfg.io.export_representatives)}"
    )

    prep = cfg.preprocessing
    lines.extend(
        [
            "",
            "[preprocessing]",
            f"min_peaks = {_toml_literal(prep.min_peaks)}",
            f"min_mz_range = {_toml_literal(prep.min_mz_range)}",
            f"min_mz = {_toml_literal(prep.min_mz)}",
            f"max_mz = {_toml_literal(prep.max_mz)}",
            _format_optional_toml(
                "remove_precursor_tol",
                prep.remove_precursor_tol,
                NONE_SENTINEL,
                comment_when_none=False,
            ),
            f"min_intensity = {_toml_literal(prep.min_intensity)}",
            _format_optional_toml(
                "max_peaks_used",
                prep.max_peaks_used,
                NONE_SENTINEL,
                comment_when_none=False,
            ),
            f"scaling = {_toml_literal(prep.scaling)}",
        ]
    )

    clust = cfg.clustering
    lines.extend(
        [
            "",
            "[clustering]",
            _format_optional_toml(
                "rt_tol", clust.rt_tol, NONE_SENTINEL, comment_when_none=True
            ),
            f"fragment_tol = {_toml_literal(clust.fragment_tol)}",
            f"linkage = {_toml_literal(clust.linkage)}",
            f"distance_threshold = {_toml_literal(clust.distance_threshold)}",
            f"min_matched_peaks = {_toml_literal(clust.min_matched_peaks)}",
            f"batch_size = {_toml_literal(clust.batch_size)}",
            _format_optional_toml(
                "precursor_charge_buckets",
                clust.precursor_charge_buckets,
                AUTO_SENTINEL,
                comment_when_none=True,
            ),
            "",
            "[clustering.precursor_tol]",
            f"mass = {_toml_literal(clust.precursor_tol.mass)}",
            f"mode = {_toml_literal(clust.precursor_tol.mode)}",
            "",
            "[consensus]",
            f"method = {_toml_literal(cfg.consensus.method)}",
        ]
    )
    if isinstance(cfg.consensus, AverageConsensusConfig):
        lines.append(
            f"outlier_cutoff_lower = "
            f"{_toml_literal(cfg.consensus.outlier_cutoff_lower)}"
        )
        lines.append(
            f"outlier_cutoff_upper = "
            f"{_toml_literal(cfg.consensus.outlier_cutoff_upper)}"
        )
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
    prep = cfg.preprocessing
    clust = cfg.clustering
    lines = [
        f"falcon version {__version__}",
        f"io.work_dir = {effective_work_dir}",
        f"io.overwrite = {cfg.io.overwrite}",
        f"io.export_representatives = {cfg.io.export_representatives}",
        f"preprocessing.min_peaks = {prep.min_peaks}",
        f"preprocessing.min_mz_range = {prep.min_mz_range}",
        f"preprocessing.min_mz = {prep.min_mz}",
        f"preprocessing.max_mz = {prep.max_mz}",
        f"preprocessing.remove_precursor_tol = {prep.remove_precursor_tol}",
        f"preprocessing.min_intensity = {prep.min_intensity}",
        f"preprocessing.max_peaks_used = {prep.max_peaks_used}",
        f"preprocessing.scaling = {prep.scaling}",
        (
            f"clustering.precursor_tol = "
            f"{clust.precursor_tol.mass} {clust.precursor_tol.mode}"
        ),
        f"clustering.rt_tol = {clust.rt_tol}",
        f"clustering.fragment_tol = {clust.fragment_tol}",
        f"clustering.linkage = {clust.linkage}",
        f"clustering.distance_threshold = {clust.distance_threshold}",
        f"clustering.min_matched_peaks = {clust.min_matched_peaks}",
        f"clustering.batch_size = {clust.batch_size}",
        (
            f"clustering.precursor_charge_buckets = "
            f"{clust.precursor_charge_buckets}"
        ),
        f"consensus.method = {cfg.consensus.method}",
    ]
    if isinstance(cfg.consensus, AverageConsensusConfig):
        lines.append(
            f"consensus.outlier_cutoff_lower = "
            f"{cfg.consensus.outlier_cutoff_lower}"
        )
        lines.append(
            f"consensus.outlier_cutoff_upper = "
            f"{cfg.consensus.outlier_cutoff_upper}"
        )
    return lines


def build_json_schema() -> dict[str, Any]:
    """
    Build a JSON Schema for sparse hierarchical TOML configs.

    Includes optional top-level ``extends`` (single path or list of
    paths) alongside optional ``[io]`` paths and TOML string sentinels
    (``"none"`` and ``"auto"``) so editor tooling validates partial
    overlay files cleanly.

    Returns
    -------
    dict[str, Any]
        JSON-serializable schema dictionary.
    """
    schema = FalconSettings.model_json_schema()
    properties = dict(schema.get("properties", {}))
    properties = {
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
        **properties,
    }
    schema["properties"] = properties
    return schema


def _paths_alias(
    destination: Path,
    resolved_dest: Path,
    dest_exists: bool,
    candidate: Path,
) -> bool:
    """Return True if ``destination`` aliases ``candidate``."""
    try:
        if resolved_dest == resolve_path(candidate):
            return True
    except (OSError, RuntimeError):
        return False
    if dest_exists and candidate.exists():
        try:
            return os.path.samefile(destination, candidate)
        except OSError:
            return False
    return False


def _validate_save_destination(
    config: FalconConfig, destination: Path
) -> None:
    """Ensure ``destination`` is a file path distinct from inputs."""
    if destination.exists() and destination.is_dir():
        raise ConfigError(
            "Config output path is a directory, expected a file path: "
            f"{destination}"
        )
    try:
        resolved_dest = resolve_path(destination)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Failed to resolve config output path {destination}: {exc}"
        ) from exc

    dest_exists = destination.exists()
    candidates = expand_input_globs(
        config.io.input_filenames, keep_unmatched_literals=True
    )
    for candidate in candidates:
        if _paths_alias(destination, resolved_dest, dest_exists, candidate):
            raise ConfigError(
                f"Config output path {destination} aliases input file "
                f"{candidate}; choose a distinct destination"
            )


def write_saved_config(config: FalconConfig, destination: Path) -> None:
    """
    Write a resolved ``FalconConfig`` as TOML to ``destination``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration to serialize.
    destination : Path
        Target TOML file path (must not be a directory or alias any
        configured input peak file).

    Raises
    ------
    ConfigError
        If ``destination`` is a directory, aliases an input file, or
        cannot be written.
    """
    _validate_save_destination(config, destination)
    try:
        destination.write_text(format_toml(config), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Failed to write configuration to {destination}: {exc}"
        ) from exc
