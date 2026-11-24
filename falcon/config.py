"""Strict hierarchical Pydantic/TOML configuration for Falcon."""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from . import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import GetJsonSchemaHandler
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

INT8_MIN = -128
INT8_MAX = 127

NONE_SENTINEL = "none"
AUTO_SENTINEL = "auto"
UNKNOWN_CHARGE: Literal["unknown"] = "unknown"
OTHER_BUCKET: Literal["other"] = "other"

PrecursorTolMode = Literal["ppm", "Da"]
ScalingMethod = Literal["off", "root", "log", "rank"]
LinkageMethod = Literal["single", "complete", "average"]
ConsensusMethod = Literal["medoid", "average"]

PathLike = str | os.PathLike[str]
ChargeValue = int | Literal["unknown"]
ChargeBucket = tuple[ChargeValue, ...] | Literal["other"]

_STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_IO_CUSTOM_FIELDS = frozenset(
    {"input_filenames", "output_filename", "work_dir"}
)
_AVERAGE_ONLY_KEYS = frozenset(
    {"outlier_cutoff_lower", "outlier_cutoff_upper"}
)


class ConfigError(ValueError):
    """Raised when config files, layers, or save paths are invalid."""


# ----------------------------------------------------------------------
# Charge-bucket parsing and m/z bin grid helpers
# ----------------------------------------------------------------------


def charge_sort_key(value: ChargeValue) -> tuple[bool, int | str]:
    """Sort numeric precursor charges ascending before ``"unknown"``."""
    return (isinstance(value, str), value)


def _parse_charge_entry(elem: Any) -> ChargeValue:
    if elem == UNKNOWN_CHARGE:
        return UNKNOWN_CHARGE
    if (
        isinstance(elem, int)
        and not isinstance(elem, bool)
        and INT8_MIN <= elem <= INT8_MAX
    ):
        return elem
    raise ValueError(
        f"Invalid charge bucket entry {elem!r}: expected int8 charge "
        f"[{INT8_MIN}, {INT8_MAX}] or '{UNKNOWN_CHARGE}'"
    )


def parse_charge_buckets(value: Any) -> tuple[ChargeBucket, ...]:
    """Validate and canonicalize a non-empty charge-bucket list."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "Charge buckets must be a non-empty list or tuple of buckets"
        )

    buckets: list[ChargeBucket] = []
    seen_values: set[ChargeValue] = set()
    seen_other = False
    for raw_bucket in value:
        if raw_bucket == OTHER_BUCKET:
            if seen_other:
                raise ValueError(
                    "Charge bucket 'other' appears more than once"
                )
            seen_other = True
            buckets.append(OTHER_BUCKET)
            continue

        if (
            not isinstance(raw_bucket, (list, tuple, set, frozenset))
            or not raw_bucket
        ):
            raise ValueError(
                "Each charge bucket must be a non-empty sequence or 'other'"
            )
        parsed = [_parse_charge_entry(elem) for elem in raw_bucket]
        unique = set(parsed)
        if len(unique) != len(parsed):
            raise ValueError(
                f"Duplicate charge value within bucket {raw_bucket!r}"
            )
        overlap = unique & seen_values
        if overlap:
            raise ValueError(
                f"Charge value(s) {sorted(overlap, key=charge_sort_key)!r} "
                "appear in more than one bucket"
            )
        seen_values |= unique
        buckets.append(tuple(sorted(unique, key=charge_sort_key)))
    return tuple(buckets)


def mz_bin_bounds(
    min_mz: float, max_mz: float, bin_size: float
) -> tuple[int, float, float]:
    """
    Compute outward-aligned m/z bin boundaries and exact bin count.

    Guarantees ``start_dim <= min_mz``, ``end_dim > max_mz``, and
    ``n_bins >= 1`` in float64 precision.
    """
    if bin_size <= 0.0 or min_mz > max_mz:
        raise ValueError(
            f"Invalid m/z bin bounds: min_mz={min_mz}, max_mz={max_mz}, "
            f"bin_size={bin_size}"
        )
    start_idx = math.floor(min_mz / bin_size)
    if start_idx * bin_size > min_mz:
        start_idx -= 1
    end_idx = math.floor(max_mz / bin_size) + 1
    if end_idx * bin_size <= max_mz:
        end_idx += 1
    return end_idx - start_idx, start_idx * bin_size, end_idx * bin_size


# ----------------------------------------------------------------------
# Pydantic field validators and schema annotations
# ----------------------------------------------------------------------


class _SentinelMeta:
    """Annotate an optional field with its TOML/CLI string sentinel."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __get_pydantic_json_schema__(
        self,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema["anyOf"] = [
            *json_schema["anyOf"],
            {"type": "string", "const": self.sentinel},
        ]
        return json_schema


def sentinel_for_field(field_info: FieldInfo) -> str | None:
    """Return the TOML/CLI disable sentinel on ``field_info``."""
    for meta in field_info.metadata:
        if isinstance(meta, _SentinelMeta):
            return meta.sentinel
    return None


def _none_sentinel(value: Any) -> Any:
    return None if value == NONE_SENTINEL else value


def _parse_optional_charge_buckets(
    value: Any,
) -> tuple[ChargeBucket, ...] | None:
    return (
        None
        if value is None or value == AUTO_SENTINEL
        else parse_charge_buckets(value)
    )


def _parse_work_dir(value: Any) -> Path | None:
    if value is None or value == NONE_SENTINEL:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("work_dir must be a non-empty path string or Path")
    return Path(value)


def validate_file_path(value: Any, *, label: str) -> Path:
    """Validate a file or prefix path, rejecting directory paths."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty path string or Path")
    if isinstance(value, str) and value.endswith(("/", "/.", "/..")):
        raise ValueError(
            f"{label} is a directory path, expected a file path: {value!r}"
        )
    path = Path(value)
    if not path.name.strip() or path.name in (".", ".."):
        raise ValueError(
            f"{label} must include a non-empty file name, got {str(path)!r}"
        )
    return path


def _parse_output_prefix(value: Any) -> Path:
    return validate_file_path(value, label="output_filename")


def _parse_input_filenames(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "input_filenames must be a non-empty list or tuple of strings"
        )
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(
            "Each entry in input_filenames must be a non-empty string"
        )
    return tuple(value)


_NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
_NonEmptyPath = Annotated[Path, Field(json_schema_extra={"minLength": 1})]
_InputPatterns = Annotated[
    tuple[Annotated[str, Field(json_schema_extra={"minLength": 1})], ...],
    Field(min_length=1),
]
_SchemaChargeBuckets = Annotated[
    tuple[
        Annotated[
            tuple[
                Annotated[int, Field(ge=INT8_MIN, le=INT8_MAX)]
                | Literal["unknown"],
                ...,
            ],
            Field(min_length=1, json_schema_extra={"uniqueItems": True}),
        ]
        | Literal["other"],
        ...,
    ],
    Field(min_length=1),
]


# ----------------------------------------------------------------------
# Configuration models
# ----------------------------------------------------------------------


class PrecursorTolerance(BaseModel):
    """Precursor mass tolerance value and unit mode."""

    model_config = _STRICT_MODEL_CONFIG

    mass: _NonNegFloat = Field(
        default=20.0, description="Precursor mass tolerance value"
    )
    mode: PrecursorTolMode = Field(
        default="ppm",
        description='Precursor mass tolerance unit ("ppm" or "Da")',
    )


class PreprocessingConfig(BaseModel):
    """Spectrum quality-filtering and normalization settings."""

    model_config = _STRICT_MODEL_CONFIG

    min_peaks: Annotated[int, Field(ge=1)] = Field(
        default=5, description="Minimum number of peaks per spectrum"
    )
    min_mz_range: _NonNegFloat = Field(
        default=250.0, description="Minimum fragment m/z span per spectrum"
    )
    min_mz: _NonNegFloat = Field(
        default=101.0, description="Minimum peak m/z inclusive"
    )
    max_mz: _PosFloat = Field(
        default=1500.0, description="Maximum peak m/z inclusive"
    )
    remove_precursor_tol: Annotated[
        _PosFloat | None,
        BeforeValidator(_none_sentinel),
        _SentinelMeta(NONE_SENTINEL),
    ] = Field(
        default=1.5,
        description=(
            "Window (Da) around precursor m/z to remove peaks, or "
            f"'{NONE_SENTINEL}' to disable"
        ),
    )
    min_intensity: Annotated[
        float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    ] = Field(
        default=0.01,
        description="Minimum peak intensity relative to base peak in [0, 1)",
    )
    max_peaks_used: Annotated[
        Annotated[int, Field(ge=1)] | None,
        BeforeValidator(_none_sentinel),
        _SentinelMeta(NONE_SENTINEL),
    ] = Field(
        default=50,
        description=(
            f"Maximum most-intense peaks to retain, or '{NONE_SENTINEL}' "
            "for all"
        ),
    )
    scaling: ScalingMethod = Field(
        default="off", description="Peak intensity scaling method"
    )

    @model_validator(mode="after")
    def _validate_invariants(self) -> Self:
        if self.min_mz >= self.max_mz:
            raise ValueError(
                f"min_mz ({self.min_mz}) must be strictly less than "
                f"max_mz ({self.max_mz})"
            )
        if self.min_mz_range > self.max_mz - self.min_mz:
            raise ValueError(
                f"min_mz_range ({self.min_mz_range}) exceeds available "
                f"m/z span max_mz - min_mz ({self.max_mz - self.min_mz})"
            )
        if (
            self.max_peaks_used is not None
            and self.max_peaks_used < self.min_peaks
        ):
            raise ValueError(
                f"max_peaks_used ({self.max_peaks_used}) must be at least "
                f"min_peaks ({self.min_peaks})"
            )
        return self


class ClusteringConfig(BaseModel):
    """Spectrum comparison and hierarchical clustering settings."""

    model_config = _STRICT_MODEL_CONFIG

    precursor_tol: PrecursorTolerance = Field(
        default_factory=PrecursorTolerance,
        description="Precursor mass tolerance value and unit",
    )
    rt_tol: Annotated[
        _NonNegFloat | None,
        BeforeValidator(_none_sentinel),
        _SentinelMeta(NONE_SENTINEL),
    ] = Field(
        default=None,
        description=(
            f"Retention time tolerance, or '{NONE_SENTINEL}' to disable"
        ),
    )
    fragment_tol: _PosFloat = Field(
        default=0.05, description="Fragment mass tolerance in m/z"
    )
    linkage: LinkageMethod = Field(
        default="complete",
        description="Hierarchical clustering linkage criterion",
    )
    distance_threshold: Annotated[
        float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ] = Field(default=0.1, description="Cosine distance threshold in (0, 1)")
    min_matched_peaks: Annotated[int, Field(ge=0)] = Field(
        default=0, description="Minimum matched fragment peaks"
    )
    batch_size: Annotated[int, Field(ge=1)] = Field(
        default=2**15, description="Maximum spectra per precursor m/z batch"
    )
    precursor_charge_buckets: Annotated[
        _SchemaChargeBuckets | None,
        BeforeValidator(_parse_optional_charge_buckets),
        _SentinelMeta(AUTO_SENTINEL),
    ] = Field(
        default=None,
        description=(
            "Precursor charge groupings, or "
            f"'{AUTO_SENTINEL}' to cluster each observed charge separately"
        ),
    )


class ConsensusConfig(BaseModel):
    """Representative spectrum selection settings."""

    model_config = _STRICT_MODEL_CONFIG

    method: ConsensusMethod = Field(
        default="medoid", description="Representative spectrum method"
    )
    outlier_cutoff_lower: _PosFloat = Field(
        default=1.5,
        description=(
            "Lower sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average')"
        ),
    )
    outlier_cutoff_upper: _PosFloat = Field(
        default=1.5,
        description=(
            "Upper sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average')"
        ),
    )

    @model_validator(mode="after")
    def _validate_method_fields(self) -> Self:
        if self.method == "medoid" and (
            self.model_fields_set & _AVERAGE_ONLY_KEYS
        ):
            raise ValueError(
                "outlier_cutoff_lower and outlier_cutoff_upper are only "
                "valid when consensus.method = 'average'"
            )
        return self


class _BaseIOConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    work_dir: Annotated[
        _NonEmptyPath | None,
        BeforeValidator(_parse_work_dir),
        _SentinelMeta(NONE_SENTINEL),
    ] = Field(
        default=None,
        description=(
            f"Working directory for Lance staging cache, or "
            f"'{NONE_SENTINEL}' for a temporary directory"
        ),
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing results and cached staging files",
    )
    export_representatives: bool = Field(
        default=False,
        description="Export cluster representative spectra to an MGF file",
    )


class PartialIOConfig(_BaseIOConfig):
    """Input/output settings where paths may be omitted in overlays."""

    input_filenames: Annotated[
        _InputPatterns | None,
        BeforeValidator(
            lambda v: None if v is None else _parse_input_filenames(v)
        ),
    ] = Field(
        default=None,
        description="Input peak file path(s) or glob patterns",
    )
    output_filename: Annotated[
        _NonEmptyPath | None,
        BeforeValidator(
            lambda v: None if v is None else _parse_output_prefix(v)
        ),
    ] = Field(
        default=None,
        description="Output file prefix for cluster CSV and optional MGF",
    )


class IOConfig(_BaseIOConfig):
    """Complete input/output settings required to execute a run."""

    input_filenames: Annotated[
        _InputPatterns,
        BeforeValidator(_parse_input_filenames),
    ] = Field(description="Input peak file path(s) or glob patterns")
    output_filename: Annotated[
        _NonEmptyPath,
        BeforeValidator(_parse_output_prefix),
    ] = Field(
        description="Output file prefix for cluster CSV and optional MGF"
    )

    @property
    def csv_path(self) -> Path:
        """Return the output cluster-assignment CSV file path."""
        return Path(f"{self.output_filename}.csv")

    @property
    def mgf_path(self) -> Path:
        """Return the output cluster-representatives MGF file path."""
        return Path(f"{self.output_filename}.mgf")


class _BaseFalconConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)

    @model_validator(mode="after")
    def _validate_cross_section(self) -> Self:
        max_peaks = self.preprocessing.max_peaks_used
        if (
            max_peaks is not None
            and self.clustering.min_matched_peaks > max_peaks
        ):
            raise ValueError(
                f"clustering.min_matched_peaks "
                f"({self.clustering.min_matched_peaks}) cannot exceed "
                f"preprocessing.max_peaks_used ({max_peaks})"
            )
        return self


class FalconSettings(_BaseFalconConfig):
    """Validated Falcon settings allowing omitted run I/O paths."""

    io: PartialIOConfig = Field(default_factory=PartialIOConfig)


class FalconConfig(_BaseFalconConfig):
    """Complete, immutable, validated configuration for a Falcon run."""

    io: IOConfig


# ----------------------------------------------------------------------
# Hierarchical TOML loading and merging
# ----------------------------------------------------------------------


def merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two config mappings without mutating inputs.

    Drops stale average-only keys when ``override`` switches
    ``consensus.method`` to ``"medoid"``.
    """
    merged: dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, val in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(val, Mapping):
            if (
                key == "consensus"
                and val.get("method") == "medoid"
                and existing.get("method") in (None, "average")
            ):
                existing = {
                    k: v
                    for k, v in existing.items()
                    if k not in _AVERAGE_ONLY_KEYS
                }
            merged[key] = merge_config_dicts(existing, val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _load_toml_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(
            f"Configuration file not found or inaccessible: {path}: {exc}"
        ) from exc
    if resolved in stack:
        cycle = " -> ".join(
            str(p) for p in (*stack[stack.index(resolved) :], resolved)
        )
        raise ConfigError(f"Cycle detected in config 'extends': {cycle}")
    if resolved.is_dir():
        raise ConfigError(
            f"Configuration path is a directory, expected a TOML file: {path}"
        )

    try:
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(
            f"Failed to load TOML configuration {path}: {exc}"
        ) from exc

    if "extends" not in data:
        return data

    raw_extends = data.pop("extends")
    entries = [raw_extends] if isinstance(raw_extends, str) else raw_extends
    if (
        not isinstance(entries, list)
        or not entries
        or any(not isinstance(e, str) or not e.strip() for e in entries)
    ):
        raise ConfigError(
            f"Invalid 'extends' in {path}: expected a non-empty path "
            "string or list of path strings"
        )

    merged: dict[str, Any] = {}
    for entry in entries:
        parent = Path(entry)
        if not parent.is_absolute():
            parent = resolved.parent / parent
        merged = merge_config_dicts(
            merged, _load_toml_file(parent, (*stack, resolved))
        )
    return merge_config_dicts(merged, data)


def load_toml_file(path: str | Path) -> dict[str, Any]:
    """Load a TOML config file, resolving ``extends`` recursively."""
    return _load_toml_file(Path(path), ())


def _resolve_layers(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config_files, (str, bytes, os.PathLike)):
        raise ConfigError(
            "config_files must be a sequence of paths (e.g. [path]), "
            f"not a single {type(config_files).__name__}"
        )
    merged: dict[str, Any] = {}
    for cfg_file in config_files:
        merged = merge_config_dicts(merged, load_toml_file(cfg_file))
    if overrides is not None:
        merged = merge_config_dicts(merged, overrides)
    return merged


def load_settings(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> FalconSettings:
    """Load and validate Falcon settings from TOML and overrides."""
    return FalconSettings.model_validate(
        _resolve_layers(config_files=config_files, overrides=overrides)
    )


def load_config(
    config_files: Sequence[str | Path] = (),
    overrides: Mapping[str, Any] | None = None,
) -> FalconConfig:
    """Load and validate a complete runnable ``FalconConfig``."""
    merged = _resolve_layers(config_files=config_files, overrides=overrides)
    merged.setdefault("io", {})
    return FalconConfig.model_validate(merged)


# ----------------------------------------------------------------------
# TOML, metadata, JSON Schema, and saved-config serialization
# ----------------------------------------------------------------------


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (Path, str)):
        return json.dumps(str(value), ensure_ascii=False).replace(
            "\x7f", "\\u007f"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _format_section_toml(
    section: BaseModel, *, skip: frozenset[str] = frozenset()
) -> list[str]:
    lines: list[str] = []
    for name, field_info in type(section).model_fields.items():
        if name in skip:
            continue
        value = getattr(section, name)
        sentinel = sentinel_for_field(field_info)
        if sentinel is not None and value is None:
            prefix = "# " if field_info.default is None else ""
            lines.append(f"{prefix}{name} = {_toml_literal(sentinel)}")
        else:
            lines.append(f"{name} = {_toml_literal(value)}")
    return lines


def format_toml(
    cfg: FalconSettings | FalconConfig, *, work_dir: Path | None = None
) -> str:
    """Render a validated config as canonical round-trippable TOML."""
    effective_wd = cfg.io.work_dir if work_dir is None else work_dir
    cons_skip = (
        _AVERAGE_ONLY_KEYS if cfg.consensus.method == "medoid" else frozenset()
    )
    lines = [
        "[io]",
        (
            f"input_filenames = {_toml_literal(cfg.io.input_filenames)}"
            if cfg.io.input_filenames is not None
            else "# input_filenames = []"
        ),
        (
            f"output_filename = {_toml_literal(cfg.io.output_filename)}"
            if cfg.io.output_filename is not None
            else '# output_filename = ""'
        ),
        (
            f"work_dir = {_toml_literal(effective_wd)}"
            if effective_wd is not None
            else f'# work_dir = "{NONE_SENTINEL}"'
        ),
        *_format_section_toml(cfg.io, skip=_IO_CUSTOM_FIELDS),
        "",
        "[preprocessing]",
        *_format_section_toml(cfg.preprocessing),
        "",
        "[clustering]",
        *_format_section_toml(
            cfg.clustering, skip=frozenset({"precursor_tol"})
        ),
        "",
        "[clustering.precursor_tol]",
        *_format_section_toml(cfg.clustering.precursor_tol),
        "",
        "[consensus]",
        *_format_section_toml(cfg.consensus, skip=cons_skip),
    ]
    return "\n".join(lines) + "\n"


def _format_section_metadata(
    prefix: str, section: BaseModel, *, skip: frozenset[str] = frozenset()
) -> list[str]:
    lines: list[str] = []
    for name in type(section).model_fields:
        if name in skip:
            continue
        value = getattr(section, name)
        formatted = (
            f"{value.mass} {value.mode}"
            if isinstance(value, PrecursorTolerance)
            else str(value)
        )
        lines.append(f"{prefix}.{name} = {formatted}")
    return lines


def format_metadata_lines(
    cfg: FalconSettings | FalconConfig, *, work_dir: Path | None = None
) -> list[str]:
    """Return canonical metadata lines for logs and CSV headers."""
    effective_wd = cfg.io.work_dir if work_dir is None else work_dir
    cons_skip = (
        _AVERAGE_ONLY_KEYS if cfg.consensus.method == "medoid" else frozenset()
    )
    return [
        f"falcon version {__version__}",
        f"io.work_dir = {effective_wd}",
        *_format_section_metadata("io", cfg.io, skip=_IO_CUSTOM_FIELDS),
        *_format_section_metadata("preprocessing", cfg.preprocessing),
        *_format_section_metadata("clustering", cfg.clustering),
        *_format_section_metadata("consensus", cfg.consensus, skip=cons_skip),
    ]


def build_json_schema() -> dict[str, Any]:
    """Build a JSON Schema for sparse hierarchical TOML configs."""
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
    return schema


def _expand_input_candidates(patterns: Sequence[str]) -> list[Path]:
    """Expand input patterns into glob matches and literal paths."""
    candidates: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            candidates.extend(Path(match) for match in matches)
        else:
            candidates.append(Path(pattern))
    return candidates


def _resolve_path(path: Path) -> Path:
    """Resolve ``path`` strictly, falling back only when missing."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.resolve(strict=False)


def _paths_alias(
    destination: Path,
    resolved_dest: Path,
    dest_exists: bool,
    candidate: Path,
) -> bool:
    """Return ``True`` if ``destination`` aliases ``candidate``."""
    try:
        if resolved_dest == _resolve_path(candidate):
            return True
    except (OSError, RuntimeError):
        return False
    if dest_exists and candidate.exists():
        try:
            return os.path.samefile(destination, candidate)
        except OSError:
            return False
    return False


def _validate_save_destination(config: FalconConfig, dest: Path) -> None:
    """Ensure ``dest`` does not alias any configured peak input."""
    try:
        resolved_dest = _resolve_path(dest)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"Failed to resolve config output path {dest}: {exc}"
        ) from exc

    dest_exists = dest.exists()
    for candidate in _expand_input_candidates(config.io.input_filenames):
        if _paths_alias(dest, resolved_dest, dest_exists, candidate):
            raise ConfigError(
                f"Config output path {dest} aliases input file "
                f"{candidate}; choose a distinct destination"
            )


def write_saved_config(config: FalconConfig, destination: str | Path) -> None:
    """
    Write a resolved ``FalconConfig`` as TOML to ``destination``.

    Validates ``destination`` against all configured peak inputs
    (literal paths and glob matches) before atomically replacing it.

    Raises
    ------
    ConfigError
        If ``destination`` is invalid, is a directory, cannot be
        resolved, aliases any peak input file, or cannot be written.
    """
    from .ms_io import atomic_write_text

    try:
        dest = validate_file_path(destination, label="Config output path")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if dest.is_dir():
        raise ConfigError(
            f"Config output path is a directory, expected a file path: {dest}"
        )
    _validate_save_destination(config, dest)
    try:
        with atomic_write_text(dest, prefix=".config_") as f_out:
            f_out.write(format_toml(config))
    except OSError as exc:
        raise ConfigError(
            f"Failed to write configuration to {dest}: {exc}"
        ) from exc
