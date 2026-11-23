"""Strict, hierarchical configuration models and validation for Falcon."""

from __future__ import annotations

import math
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


class FalconConfig(_BaseFalconConfig):
    """Complete, immutable, validated configuration for a Falcon run."""

    io: IOConfig
