"""Strict hierarchical Pydantic/TOML configuration for Falcon."""

from __future__ import annotations

import copy
import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from . import __version__
from .types import (
    AUTO_SENTINEL,
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    DEFAULT_OUTLIER_CUTOFF_LOWER,
    DEFAULT_OUTLIER_CUTOFF_UPPER,
    INT8_MAX,
    INT8_MIN,
    NONE_SENTINEL,
    LinkageMethod,
    PrecursorTolMode,
    ScalingMethod,
    parse_optional_charge_buckets,
)
from .utils import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic import GetJsonSchemaHandler
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

__all__ = [
    "AverageConsensusConfig",
    "ClusteringConfig",
    "ConfigError",
    "ConsensusConfig",
    "FalconConfig",
    "FalconSettings",
    "IOConfig",
    "MedoidConsensusConfig",
    "PartialIOConfig",
    "PrecursorTolerance",
    "PreprocessingConfig",
    "build_json_schema",
    "format_metadata_lines",
    "format_toml",
    "load_config",
    "load_settings",
    "load_toml_file",
    "merge_config_dicts",
    "sentinel_for_field",
    "validate_file_path",
    "write_saved_config",
]

_T = TypeVar("_T")
_STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_IO_CUSTOM_FIELDS = frozenset(
    {"input_filenames", "output_filename", "work_dir"}
)


class ConfigError(ValueError):
    """Raised when config files, layers, or save paths are invalid."""


class _AllowSentinelSchema:
    """Pydantic annotation adding a TOML string sentinel literal."""

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


class _RelaxUnionSchema:
    """Relax a discriminated union's ``oneOf`` to ``anyOf``."""

    def __get_pydantic_json_schema__(
        self,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema["anyOf"] = json_schema.pop("oneOf")
        return json_schema


def sentinel_for_field(field_info: FieldInfo) -> str | None:
    """Return the TOML/CLI disable sentinel on ``field_info``."""
    for meta in field_info.metadata:
        if isinstance(meta, _AllowSentinelSchema):
            return meta.sentinel
    return None


def _none_sentinel(value: Any) -> Any:
    return None if value == NONE_SENTINEL else value


def _default_consensus_method(value: Any) -> Any:
    if isinstance(value, Mapping) and "method" not in value:
        return {"method": "medoid", **value}
    return value


def _optional(
    parser: Callable[[Any], _T],
    *,
    allow_none_sentinel: bool = False,
) -> Callable[[Any], _T | None]:
    def _parse(value: Any) -> _T | None:
        if value is None or (allow_none_sentinel and value == NONE_SENTINEL):
            return None
        return parser(value)

    return _parse


def _parse_path(value: Any, *, label: str = "Path") -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"Expected a path string or Path for {label}, got "
            f"{type(value).__name__}"
        )
    if not str(value).strip():
        raise ValueError(f"{label} must not be empty or whitespace")
    return value if isinstance(value, Path) else Path(value)


def validate_file_path(value: Any, *, label: str) -> Path:
    """Validate a file or prefix path, rejecting directory paths."""
    if isinstance(value, str) and value.endswith(("/", "/.", "/..")):
        raise ValueError(
            f"{label} is a directory path, expected a file path: {value!r}"
        )
    path = _parse_path(value, label=label)
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
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "Each entry in input_filenames must be a non-empty string"
            )
    return tuple(value)


_NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
_NonEmptyStr = Annotated[str, Field(json_schema_extra={"minLength": 1})]
_NonEmptyPath = Annotated[Path, Field(json_schema_extra={"minLength": 1})]
_InputPatterns = Annotated[tuple[_NonEmptyStr, ...], Field(min_length=1)]

_SchemaChargeGroup = Annotated[
    tuple[
        Annotated[int, Field(ge=INT8_MIN, le=INT8_MAX)] | Literal["unknown"],
        ...,
    ],
    Field(min_length=1, json_schema_extra={"uniqueItems": True}),
]
_SchemaChargeBuckets = Annotated[
    tuple[_SchemaChargeGroup | Literal["other"], ...], Field(min_length=1)
]

_OptionalNonNegFloat = Annotated[
    _NonNegFloat | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema(NONE_SENTINEL),
]
_OptionalPosFloat = Annotated[
    _PosFloat | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema(NONE_SENTINEL),
]
_OptionalPosInt = Annotated[
    Annotated[int, Field(ge=1)] | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema(NONE_SENTINEL),
]
_OptionalWorkDir = Annotated[
    _NonEmptyPath | None,
    BeforeValidator(_optional(_parse_path, allow_none_sentinel=True)),
    _AllowSentinelSchema(NONE_SENTINEL),
]
_OptionalChargeBuckets = Annotated[
    _SchemaChargeBuckets | None,
    BeforeValidator(parse_optional_charge_buckets),
    _AllowSentinelSchema(AUTO_SENTINEL),
]


class PrecursorTolerance(BaseModel):
    """Precursor mass tolerance value and unit mode."""

    model_config = _STRICT_MODEL_CONFIG

    mass: _NonNegFloat = Field(
        default=20.0,
        description="Precursor mass tolerance value",
    )
    mode: PrecursorTolMode = Field(
        default="ppm",
        description='Precursor mass tolerance unit ("ppm" or "Da")',
    )


class PreprocessingConfig(BaseModel):
    """Spectrum quality-filtering and normalization settings."""

    model_config = _STRICT_MODEL_CONFIG

    min_peaks: Annotated[int, Field(ge=1)] = Field(
        default=5,
        description="Minimum number of peaks per spectrum",
    )
    min_mz_range: _NonNegFloat = Field(
        default=250.0,
        description="Minimum fragment m/z span per spectrum",
    )
    min_mz: _NonNegFloat = Field(
        default=DEFAULT_MIN_MZ,
        description="Minimum peak m/z inclusive",
    )
    max_mz: _PosFloat = Field(
        default=DEFAULT_MAX_MZ,
        description="Maximum peak m/z inclusive",
    )
    remove_precursor_tol: _OptionalPosFloat = Field(
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
    max_peaks_used: _OptionalPosInt = Field(
        default=50,
        description=(
            f"Maximum most-intense peaks to retain, or '{NONE_SENTINEL}' "
            "for all"
        ),
    )
    scaling: ScalingMethod = Field(
        default="off",
        description="Peak intensity scaling method",
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
    rt_tol: _OptionalNonNegFloat = Field(
        default=None,
        description=(
            f"Retention time tolerance, or '{NONE_SENTINEL}' to disable"
        ),
    )
    fragment_tol: _PosFloat = Field(
        default=0.05,
        description="Fragment mass tolerance in m/z",
    )
    linkage: LinkageMethod = Field(
        default="complete",
        description="Hierarchical clustering linkage criterion",
    )
    distance_threshold: Annotated[
        float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ] = Field(
        default=0.1,
        description="Cosine distance threshold in (0, 1)",
    )
    min_matched_peaks: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Minimum matched fragment peaks",
    )
    batch_size: Annotated[int, Field(ge=1)] = Field(
        default=2**15,
        description="Maximum spectra per precursor m/z batch",
    )
    precursor_charge_buckets: _OptionalChargeBuckets = Field(
        default=None,
        description=(
            "Precursor charge groupings, or "
            f"'{AUTO_SENTINEL}' to cluster each observed charge separately"
        ),
    )


class MedoidConsensusConfig(BaseModel):
    """Medoid representative spectrum selection settings."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["medoid"] = Field(
        default="medoid",
        description="Representative spectrum method",
    )


class AverageConsensusConfig(BaseModel):
    """Averaged consensus spectrum with iterative sigma clipping."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["average"] = Field(
        default="average",
        description="Representative spectrum method",
    )
    outlier_cutoff_lower: _PosFloat = Field(
        default=DEFAULT_OUTLIER_CUTOFF_LOWER,
        description=(
            "Lower sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average')"
        ),
    )
    outlier_cutoff_upper: _PosFloat = Field(
        default=DEFAULT_OUTLIER_CUTOFF_UPPER,
        description=(
            "Upper sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average')"
        ),
    )


ConsensusConfig = Annotated[
    MedoidConsensusConfig | AverageConsensusConfig,
    Field(discriminator="method"),
    BeforeValidator(_default_consensus_method),
    _RelaxUnionSchema(),
]

_AVERAGE_ONLY_KEYS: frozenset[str] = frozenset(
    AverageConsensusConfig.model_fields
) - {"method"}


class _BaseIOConfig(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    work_dir: _OptionalWorkDir = Field(
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
        BeforeValidator(_optional(_parse_input_filenames)),
    ] = Field(
        default=None,
        description="Input peak file path(s) or glob patterns",
    )
    output_filename: Annotated[
        _NonEmptyPath | None,
        BeforeValidator(_optional(_parse_output_prefix)),
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
    consensus: ConsensusConfig = Field(default_factory=MedoidConsensusConfig)

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
# Hierarchical TOML loading and deep merging
# ----------------------------------------------------------------------


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    top_level: bool,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, existing in base.items():
        if key not in override:
            merged[key] = _copy_value(existing)
            continue
        val = override[key]
        if isinstance(existing, Mapping) and isinstance(val, Mapping):
            base_section = existing
            if (
                top_level
                and key == "consensus"
                and val.get("method") == "medoid"
                and existing.get("method") in (None, "average")
            ):
                base_section = {
                    k: v
                    for k, v in existing.items()
                    if k not in _AVERAGE_ONLY_KEYS
                }
            merged[key] = _merge_dicts(base_section, val, top_level=False)
        else:
            merged[key] = _copy_value(val)
    for key, val in override.items():
        if key not in base:
            merged[key] = _copy_value(val)
    return merged


def merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two config mappings without mutating inputs.

    When ``override`` switches ``consensus.method`` to ``"medoid"``,
    stale average-only keys from ``base["consensus"]`` are dropped.
    """
    return _merge_dicts(base, override, top_level=True)


def _parse_extends(raw_extends: Any, source: Path) -> list[str]:
    if isinstance(raw_extends, str):
        entries = [raw_extends]
    elif isinstance(raw_extends, list) and raw_extends:
        entries = raw_extends
    else:
        raise ConfigError(
            f"Invalid 'extends' in {source}: expected a non-empty path "
            f"string or list of path strings"
        )
    for item in entries:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"Invalid 'extends' entry {item!r} in {source}: "
                "expected non-empty path string"
            )
    return entries


def _load_toml_file(
    path: Path,
    *,
    stack: tuple[Path, ...],
    referenced_by: Path | None,
) -> dict[str, Any]:
    origin = f" (referenced by {referenced_by})" if referenced_by else ""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Configuration file not found: {path}{origin}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(
            f"Failed to resolve configuration file {path}{origin}: {exc}"
        ) from exc

    if resolved in stack:
        cycle_trace = " -> ".join(
            str(p) for p in (*stack[stack.index(resolved) :], resolved)
        )
        raise ConfigError(f"Cycle detected in config 'extends': {cycle_trace}")
    if resolved.is_dir():
        raise ConfigError(
            f"Configuration path is a directory, expected a TOML file: "
            f"{path}{origin}"
        )

    try:
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            f"Failed to read configuration file {path}{origin}: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Invalid TOML syntax in {path}{origin}: {exc}"
        ) from exc

    if "extends" not in data:
        return data

    extends_entries = _parse_extends(data.pop("extends"), path)
    next_stack = (*stack, resolved)
    merged: dict[str, Any] = {}
    for entry in extends_entries:
        parent_path = Path(entry)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent_data = _load_toml_file(
            parent_path, stack=next_stack, referenced_by=path
        )
        merged = merge_config_dicts(merged, parent_data)
    return merge_config_dicts(merged, data)


def load_toml_file(path: str | Path) -> dict[str, Any]:
    """
    Load a TOML configuration file, resolving ``extends`` recursively.

    Relative ``extends`` paths resolve against the declaring file's
    parent directory. Diamond inheritance succeeds while cycles raise
    ``ConfigError``.
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
    merged = _resolve_layers(config_files=config_files, overrides=overrides)
    return FalconSettings.model_validate(merged)


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


def _format_optional_toml(
    key: str, value: Any, sentinel: str, *, comment_when_none: bool
) -> str:
    if value is not None:
        return f"{key} = {_toml_literal(value)}"
    prefix = "# " if comment_when_none else ""
    return f"{prefix}{key} = {_toml_literal(sentinel)}"


def _format_section_toml(
    section: BaseModel, *, skip: frozenset[str] = frozenset()
) -> list[str]:
    lines: list[str] = []
    for name, field_info in type(section).model_fields.items():
        if name in skip:
            continue
        value = getattr(section, name)
        sentinel = sentinel_for_field(field_info)
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


def format_toml(
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> str:
    """Render a validated config as canonical round-trippable TOML."""
    effective_work_dir = cfg.io.work_dir if work_dir is None else work_dir
    lines: list[str] = ["[io]"]
    lines.append(
        f"input_filenames = {_toml_literal(cfg.io.input_filenames)}"
        if cfg.io.input_filenames is not None
        else "# input_filenames = []"
    )
    lines.append(
        f"output_filename = {_toml_literal(cfg.io.output_filename)}"
        if cfg.io.output_filename is not None
        else '# output_filename = ""'
    )
    lines.append(
        _format_optional_toml(
            "work_dir",
            None if effective_work_dir is None else str(effective_work_dir),
            NONE_SENTINEL,
            comment_when_none=True,
        )
    )
    lines.extend(_format_section_toml(cfg.io, skip=_IO_CUSTOM_FIELDS))
    lines.extend(
        ["", "[preprocessing]", *_format_section_toml(cfg.preprocessing)]
    )
    lines.extend(
        [
            "",
            "[clustering]",
            *_format_section_toml(
                cfg.clustering, skip=frozenset({"precursor_tol"})
            ),
            "",
            "[clustering.precursor_tol]",
            *_format_section_toml(cfg.clustering.precursor_tol),
        ]
    )
    lines.extend(["", "[consensus]", *_format_section_toml(cfg.consensus)])
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
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> list[str]:
    """Return canonical metadata lines for logs and CSV headers."""
    effective_work_dir = cfg.io.work_dir if work_dir is None else work_dir
    return [
        f"falcon version {__version__}",
        f"io.work_dir = {effective_work_dir}",
        *_format_section_metadata("io", cfg.io, skip=_IO_CUSTOM_FIELDS),
        *_format_section_metadata("preprocessing", cfg.preprocessing),
        *_format_section_metadata("clustering", cfg.clustering),
        *_format_section_metadata("consensus", cfg.consensus),
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


def write_saved_config(config: FalconConfig, destination: str | Path) -> None:
    """
    Write a resolved ``FalconConfig`` as TOML to ``destination``.

    Raises
    ------
    ConfigError
        If ``destination`` is a directory, collides with an input,
        output, or working-directory path, or cannot be written.
    """
    try:
        dest = validate_file_path(destination, label="Config output path")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if dest.is_dir():
        raise ConfigError(
            f"Config output path is a directory, expected a file path: {dest}"
        )

    resolved_dest = dest.resolve()
    protected: list[tuple[str, Path]] = [
        ("output file", config.io.csv_path),
        ("output file", config.io.mgf_path),
        *(("input file", Path(p)) for p in config.io.input_filenames),
    ]
    if config.io.work_dir is not None:
        protected.append(("working directory", config.io.work_dir))
    for kind, candidate in protected:
        if resolved_dest == candidate.resolve():
            raise ConfigError(
                f"Config output path {dest} aliases {kind} "
                f"{candidate}; choose a distinct destination"
            )

    toml_text = format_toml(config)
    try:
        with atomic_write_text(dest, prefix=".config_") as f_out:
            f_out.write(toml_text)
    except OSError as exc:
        raise ConfigError(
            f"Failed to write configuration to {dest}: {exc}"
        ) from exc
