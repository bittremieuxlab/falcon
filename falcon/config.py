"""Strict, hierarchical configuration models and validation for Falcon."""

from __future__ import annotations

import copy
import json
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from falcon import __version__

STAGING_CACHE_SCHEMA_VERSION = 1

_INT8_MIN = -128
_INT8_MAX = 127
_INT32_MAX = 2**31 - 1

ChargeValue = int | Literal["unknown"]
ChargeBucket = tuple[ChargeValue, ...] | Literal["other"]


class ConfigError(ValueError):
    """Raised when configuration files or layer structures are invalid."""


class _AllowSentinelSchema:
    """Pydantic schema annotation adding a TOML string sentinel literal."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __get_pydantic_json_schema__(
        self,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        if "anyOf" in json_schema:
            json_schema["anyOf"] = [
                *json_schema["anyOf"],
                {"const": self.sentinel},
            ]
        return json_schema


def _none_sentinel(value: Any) -> Any:
    """Normalize the TOML/CLI ``"none"`` sentinel to ``None``."""
    if value == "none":
        return None
    return value


def _parse_path(value: Any) -> Path:
    """Validate and convert a non-empty string or Path to ``Path``."""
    if isinstance(value, Path):
        if not str(value).strip():
            raise ValueError("Path must not be empty")
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Path must not be empty or whitespace")
        return Path(value)
    raise ValueError(f"Expected a path string or Path, got {type(value)}")


def _parse_optional_path(value: Any) -> Path | None:
    """Parse an optional path, treating ``"none"`` as ``None``."""
    if value is None or value == "none":
        return None
    return _parse_path(value)


def _parse_output_prefix(value: Any) -> Path:
    """Validate an output file prefix path (non-empty file name)."""
    path = _parse_path(value)
    if path.name in ("", ".", ".."):
        raise ValueError(
            "output_filename must include a file prefix name, "
            f"got {str(path)!r}"
        )
    return path


def _parse_optional_output_prefix(value: Any) -> Path | None:
    """Validate an optional output file prefix path."""
    if value is None:
        return None
    return _parse_output_prefix(value)


def _parse_input_filenames(value: Any) -> tuple[str, ...]:
    """Validate a non-empty sequence of non-blank input file patterns."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("input_filenames must be a list or tuple of strings")
    if len(value) == 0:
        raise ValueError("input_filenames must contain at least one pattern")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "Each entry in input_filenames must be a non-empty string"
            )
        result.append(item)
    return tuple(result)


def _parse_optional_input_filenames(value: Any) -> tuple[str, ...] | None:
    """Validate an optional sequence of input file patterns."""
    if value is None:
        return None
    return _parse_input_filenames(value)


def _charge_sort_key(value: ChargeValue) -> tuple[bool, int | str]:
    """Sort numeric charges ascending before the ``"unknown"`` sentinel."""
    return (isinstance(value, str), value)


def _parse_single_charge_bucket(
    raw_bucket: Any,
    seen_values: set[ChargeValue],
) -> tuple[ChargeValue, ...]:
    """Validate and canonicalize one non-``"other"`` charge bucket."""
    if not isinstance(raw_bucket, (list, tuple, set, frozenset)):
        raise ValueError(
            "Each charge bucket must be a sequence of charges or 'other'"
        )
    if len(raw_bucket) == 0:
        raise ValueError("Charge bucket must not be empty")

    bucket_seen: set[ChargeValue] = set()
    for elem in raw_bucket:
        if isinstance(elem, bool):
            raise ValueError(
                "Charge bucket entries must be integers or 'unknown', "
                "not booleans"
            )
        if elem == "unknown":
            parsed: ChargeValue = "unknown"
        elif isinstance(elem, int):
            if not (_INT8_MIN <= elem <= _INT8_MAX):
                raise ValueError(
                    f"Precursor charge {elem} is outside int8 range "
                    f"[{_INT8_MIN}, {_INT8_MAX}]"
                )
            parsed = elem
        else:
            raise ValueError(
                f"Invalid charge bucket entry {elem!r}: expected int8 "
                f"charge or 'unknown'"
            )

        if parsed in bucket_seen:
            raise ValueError(
                f"Duplicate charge value {parsed!r} within the same bucket"
            )
        if parsed in seen_values:
            raise ValueError(
                f"Charge value {parsed!r} appears in more than one bucket"
            )
        bucket_seen.add(parsed)

    seen_values |= bucket_seen
    return tuple(sorted(bucket_seen, key=_charge_sort_key))


def _parse_charge_buckets(
    value: Any,
) -> tuple[ChargeBucket, ...] | None:
    """Validate and canonicalize ``precursor_charge_buckets``."""
    if value is None or value == "auto":
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "precursor_charge_buckets must be 'auto' or a non-empty list "
            "of buckets"
        )
    if len(value) == 0:
        raise ValueError("precursor_charge_buckets must not be empty")

    buckets: list[ChargeBucket] = []
    seen_values: set[ChargeValue] = set()
    seen_other = False
    for raw_bucket in value:
        if raw_bucket == "other":
            if seen_other:
                raise ValueError(
                    "Charge bucket 'other' appears more than once"
                )
            seen_other = True
            buckets.append("other")
            continue
        buckets.append(_parse_single_charge_bucket(raw_bucket, seen_values))

    return tuple(buckets)


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

OptionalNonNegFloat = Annotated[
    NonNegFloat | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema("none"),
]
OptionalPosFloat = Annotated[
    PosFloat | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema("none"),
]
OptionalPosInt = Annotated[
    Annotated[int, Field(ge=1)] | None,
    BeforeValidator(_none_sentinel),
    _AllowSentinelSchema("none"),
]
OptionalWorkDir = Annotated[
    Path | None,
    BeforeValidator(_parse_optional_path),
    _AllowSentinelSchema("none"),
]
OptionalChargeBuckets = Annotated[
    tuple[ChargeBucket, ...] | None,
    BeforeValidator(_parse_charge_buckets),
    _AllowSentinelSchema("auto"),
]

_STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


class PrecursorTolerance(BaseModel):
    """Precursor mass tolerance value and unit mode."""

    model_config = _STRICT_MODEL_CONFIG

    mass: NonNegFloat = 20.0
    mode: Literal["ppm", "Da"] = "ppm"


class PreprocessingConfig(BaseModel):
    """Spectrum quality-filtering and normalization settings."""

    model_config = _STRICT_MODEL_CONFIG

    min_peaks: Annotated[int, Field(ge=1)] = 5
    min_mz_range: NonNegFloat = 250.0
    min_mz: NonNegFloat = 101.0
    max_mz: PosFloat = 1500.0
    remove_precursor_tol: OptionalPosFloat = 1.5
    min_intensity: Annotated[
        float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.01
    max_peaks_used: OptionalPosInt = 50
    scaling: Literal["off", "root", "log", "rank"] = "off"

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
        default_factory=PrecursorTolerance
    )
    rt_tol: OptionalNonNegFloat = None
    fragment_tol: PosFloat = 0.05
    linkage: Literal["single", "complete", "average"] = "complete"
    distance_threshold: Annotated[
        float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.1
    min_matched_peaks: Annotated[int, Field(ge=0)] = 0
    batch_size: Annotated[int, Field(ge=1)] = 2**15
    precursor_charge_buckets: OptionalChargeBuckets = None


class MedoidConsensusConfig(BaseModel):
    """Medoid representative spectrum selection settings."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["medoid"] = "medoid"


class AverageConsensusConfig(BaseModel):
    """Averaged consensus spectrum with iterative sigma clipping."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["average"] = "average"
    outlier_cutoff_lower: PosFloat = 1.5
    outlier_cutoff_upper: PosFloat = 1.5


ConsensusConfig = Annotated[
    MedoidConsensusConfig | AverageConsensusConfig,
    Field(discriminator="method"),
]


class _BaseIOConfig(BaseModel):
    """Shared optional I/O flags for partial and complete configs."""

    model_config = _STRICT_MODEL_CONFIG

    work_dir: OptionalWorkDir = None
    overwrite: bool = False
    export_representatives: bool = False


class PartialIOConfig(_BaseIOConfig):
    """Input/output settings where paths may be omitted in overlays."""

    input_filenames: Annotated[
        tuple[str, ...] | None,
        BeforeValidator(_parse_optional_input_filenames),
    ] = None
    output_filename: Annotated[
        Path | None,
        BeforeValidator(_parse_optional_output_prefix),
    ] = None


class IOConfig(_BaseIOConfig):
    """Complete input/output settings required to execute a run."""

    input_filenames: Annotated[
        tuple[str, ...],
        BeforeValidator(_parse_input_filenames),
    ]
    output_filename: Annotated[
        Path,
        BeforeValidator(_parse_output_prefix),
    ]


class _BaseFalconConfig(BaseModel):
    """Shared scientific configuration sections and cross-section checks."""

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

        min_mz = self.preprocessing.min_mz
        max_mz = self.preprocessing.max_mz
        frag_tol = self.clustering.fragment_tol
        span_ratio = (max_mz - min_mz) / frag_tol
        if not math.isfinite(span_ratio):
            raise ValueError(
                f"clustering.fragment_tol ({frag_tol}) causes non-finite "
                f"m/z bin count over [{min_mz}, {max_mz}]"
            )
        start_dim = min_mz - (min_mz % frag_tol)
        end_dim = max_mz + frag_tol - (max_mz % frag_tol)
        n_bins_raw = (end_dim - start_dim) / frag_tol
        if not math.isfinite(n_bins_raw) or math.ceil(n_bins_raw) > _INT32_MAX:
            raise ValueError(
                f"m/z bin count for [{min_mz}, {max_mz}] at fragment_tol="
                f"{frag_tol} exceeds int32 maximum ({_INT32_MAX})"
            )
        return self


class FalconSettings(_BaseFalconConfig):
    """Validated Falcon settings allowing omitted run I/O paths."""

    io: PartialIOConfig = Field(default_factory=PartialIOConfig)

    def to_toml(self, *, work_dir: Path | None = None) -> str:
        """Serialize resolved settings as deterministic TOML text."""
        return _format_toml(self, work_dir=work_dir)

    def to_metadata_lines(self, *, work_dir: Path | None = None) -> list[str]:
        """Return canonical ``section.key = value`` provenance lines."""
        return _format_metadata_lines(self, work_dir=work_dir)


class FalconConfig(_BaseFalconConfig):
    """Complete, immutable, validated configuration for a Falcon run."""

    io: IOConfig

    def to_toml(self, *, work_dir: Path | None = None) -> str:
        """Serialize resolved configuration as deterministic TOML text."""
        return _format_toml(self, work_dir=work_dir)

    def to_metadata_lines(self, *, work_dir: Path | None = None) -> list[str]:
        """Return canonical ``section.key = value`` provenance lines."""
        return _format_metadata_lines(self, work_dir=work_dir)


# ----------------------------------------------------------------------
# Hierarchical merging and TOML loading
# ----------------------------------------------------------------------


def merge_config_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two configuration mappings without mutating inputs.

    Lists and scalar values replace lower-layer entries atomically.
    When ``override`` switches ``consensus.method`` to a different
    discriminated variant than ``base``, stale variant-specific keys
    from ``base["consensus"]`` are discarded.

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
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, val in override.items():
        existing = merged.get(key)
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
    return merged


def _parse_extends_entries(raw_extends: Any, source: Path) -> list[str]:
    """Validate and normalize a TOML ``extends`` value into path strings."""
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
    resolved = path.resolve()
    if resolved in _stack:
        cycle_start = _stack.index(resolved)
        cycle_trace = " -> ".join(
            str(p) for p in (*_stack[cycle_start:], resolved)
        )
        raise ConfigError(f"Cycle detected in config 'extends': {cycle_trace}")

    origin = (
        f" (referenced by {_referenced_by})"
        if _referenced_by is not None
        else ""
    )
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
    merged: dict[str, Any] = {}
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


# ----------------------------------------------------------------------
# Deterministic TOML serialization, metadata comments, and JSON Schema
# ----------------------------------------------------------------------


def _toml_literal(value: Any) -> str:
    """Format a validated scalar or nested list as a TOML value literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)}")


def _format_toml(
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> str:
    """Render a validated configuration as canonical round-trippable TOML."""
    effective_work_dir = cfg.io.work_dir if work_dir is None else work_dir
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
    if effective_work_dir is not None:
        lines.append(f"work_dir = {_toml_literal(effective_work_dir)}")
    else:
        lines.append('# work_dir = "none"')
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
            (
                f"remove_precursor_tol = "
                f"{_toml_literal(prep.remove_precursor_tol)}"
                if prep.remove_precursor_tol is not None
                else 'remove_precursor_tol = "none"'
            ),
            f"min_intensity = {_toml_literal(prep.min_intensity)}",
            (
                f"max_peaks_used = {_toml_literal(prep.max_peaks_used)}"
                if prep.max_peaks_used is not None
                else 'max_peaks_used = "none"'
            ),
            f"scaling = {_toml_literal(prep.scaling)}",
        ]
    )

    clust = cfg.clustering
    lines.extend(
        [
            "",
            "[clustering]",
            (
                f"rt_tol = {_toml_literal(clust.rt_tol)}"
                if clust.rt_tol is not None
                else '# rt_tol = "none"'
            ),
            f"fragment_tol = {_toml_literal(clust.fragment_tol)}",
            f"linkage = {_toml_literal(clust.linkage)}",
            f"distance_threshold = {_toml_literal(clust.distance_threshold)}",
            f"min_matched_peaks = {_toml_literal(clust.min_matched_peaks)}",
            f"batch_size = {_toml_literal(clust.batch_size)}",
            (
                f"precursor_charge_buckets = "
                f"{_toml_literal(clust.precursor_charge_buckets)}"
                if clust.precursor_charge_buckets is not None
                else '# precursor_charge_buckets = "auto"'
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


def _format_metadata_lines(
    cfg: FalconSettings | FalconConfig,
    *,
    work_dir: Path | None = None,
) -> list[str]:
    """Return canonical metadata lines for startup logs and CSV headers."""
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
    Build a JSON Schema compatible with sparse hierarchical TOML configs.

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


# ----------------------------------------------------------------------
# Staging cache manifest helpers
# ----------------------------------------------------------------------


def _serialize_buckets(
    buckets: Sequence[Any] | None,
) -> list[list[int | str] | str] | None:
    """Convert charge bucket keys into deterministic JSON-safe structures."""
    if buckets is None:
        return None
    serialized: list[list[int | str] | str] = []
    for bucket in buckets:
        if bucket == "other":
            serialized.append("other")
        else:
            sorted_vals = sorted(bucket, key=_charge_sort_key)
            serialized.append(list(sorted_vals))
    return serialized


def build_staging_manifest(
    config: FalconConfig,
    resolved_inputs: Sequence[Path],
    buckets: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable staging cache provenance manifest.

    Captures every parameter that affects staged Lance datasets:
    resolved input files (canonical path, byte size, mtime_ns),
    ``preprocessing``, ``clustering.fragment_tol``, and
    ``clustering.precursor_charge_buckets``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    resolved_inputs : Sequence[Path]
        Expanded input peak file paths in deterministic staging order.
    buckets : Sequence[Any] | None, default=None
        Staged charge bucket keys written to disk, if already computed.

    Returns
    -------
    dict[str, Any]
        JSON-serializable manifest dictionary.
    """
    input_entries: list[dict[str, Any]] = []
    for input_path in resolved_inputs:
        resolved = input_path.resolve()
        st = os.stat(resolved)
        input_entries.append(
            {
                "path": str(resolved),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )

    return {
        "schema_version": STAGING_CACHE_SCHEMA_VERSION,
        "falcon_version": str(__version__),
        "inputs": input_entries,
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "fragment_tol": config.clustering.fragment_tol,
        "precursor_charge_buckets": _serialize_buckets(
            config.clustering.precursor_charge_buckets
        ),
        "buckets": _serialize_buckets(buckets),
    }


def diff_staging_manifests(
    cached: Any,
    expected: Mapping[str, Any],
) -> list[str]:
    """
    Return staging manifest section names that differ from ``expected``.

    Parameters
    ----------
    cached : Any
        Parsed JSON payload loaded from ``staging_cache.json``.
    expected : Mapping[str, Any]
        Expected staging manifest for the current run.

    Returns
    -------
    list[str]
        Names of mismatched sections (empty list when the cache is
        valid and reusable).
    """
    if not isinstance(cached, Mapping) or cached.get(
        "schema_version"
    ) != expected.get("schema_version"):
        return ["schema_version"]

    mismatches: list[str] = []
    for key in (
        "inputs",
        "preprocessing",
        "fragment_tol",
        "precursor_charge_buckets",
    ):
        if cached.get(key) != expected.get(key):
            mismatches.append(key)
    return mismatches


def deserialize_cached_buckets(raw_buckets: Any) -> list[ChargeBucket]:
    """
    Validate and deserialize staged bucket keys from ``staging_cache.json``.

    Parameters
    ----------
    raw_buckets : Any
        JSON-decoded ``buckets`` field from a staging cache manifest.

    Returns
    -------
    list[ChargeBucket]
        Validated charge bucket keys (tuples of charges/``"unknown"``
        or ``"other"``).

    Raises
    ------
    ValueError
        If ``raw_buckets`` is malformed or contains invalid bucket keys.
    """
    if not isinstance(raw_buckets, list):
        raise ValueError("Cached staging manifest has invalid 'buckets' field")
    if len(raw_buckets) == 0:
        return []
    parsed = _parse_charge_buckets(raw_buckets)
    if parsed is None:
        raise ValueError("Cached staging manifest has invalid 'buckets' field")
    return list(parsed)
