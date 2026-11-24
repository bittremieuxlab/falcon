"""Strict frozen Pydantic configuration models for Falcon."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from ..types import (
    AUTO_SENTINEL,
    INT32_MAX,
    NONE_SENTINEL,
    ChargeBucket,
    mz_bin_bounds,
    parse_optional_charge_buckets,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

DEFAULT_MIN_MZ: float = 101.0
DEFAULT_MAX_MZ: float = 1500.0
DEFAULT_OUTLIER_CUTOFF_LOWER: float = 1.5
DEFAULT_OUTLIER_CUTOFF_UPPER: float = 1.5

PrecursorTolMode = Literal["ppm", "Da"]
ScalingMethod = Literal["off", "root", "log", "rank"]
LinkageMethod = Literal["single", "complete", "average"]
ConsensusMethod = Literal["medoid", "average"]

__all__ = [
    "DEFAULT_MAX_MZ",
    "DEFAULT_MIN_MZ",
    "DEFAULT_OUTLIER_CUTOFF_LOWER",
    "DEFAULT_OUTLIER_CUTOFF_UPPER",
    "AverageConsensusConfig",
    "ClusteringConfig",
    "ConfigError",
    "ConsensusConfig",
    "ConsensusMethod",
    "FalconConfig",
    "FalconSettings",
    "IOConfig",
    "LinkageMethod",
    "MedoidConsensusConfig",
    "PartialIOConfig",
    "PrecursorTolMode",
    "PrecursorTolerance",
    "PreprocessingConfig",
    "ScalingMethod",
]

_T = TypeVar("_T")


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
        if "anyOf" in json_schema:
            json_schema["anyOf"] = [
                *json_schema["anyOf"],
                {"type": "string", "const": self.sentinel},
            ]
        return json_schema


def _none_sentinel(value: Any) -> Any:
    """Normalize the TOML/CLI ``"none"`` sentinel to ``None``."""
    return None if value == NONE_SENTINEL else value


def _optional(
    parser: Callable[[Any], _T],
    *,
    allow_none_sentinel: bool = False,
) -> Callable[[Any], _T | None]:
    """Wrap ``parser`` to pass ``None`` (or ``"none"``) through."""

    def _parse(value: Any) -> _T | None:
        if value is None or (allow_none_sentinel and value == NONE_SENTINEL):
            return None
        return parser(value)

    return _parse


_DIR_LIKE_SUFFIXES: tuple[str, ...] = ("/", "\\", "/.", "\\.", "/..", "\\..")


def _parse_path(value: Any) -> Path:
    """Validate and convert a non-empty string or Path to ``Path``."""
    if isinstance(value, Path):
        if not str(value).strip():
            raise ValueError("Path must not be empty or whitespace")
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Path must not be empty or whitespace")
        return Path(value)
    raise ValueError(
        f"Expected a path string or Path, got {type(value).__name__}"
    )


def validate_file_path(value: Any, *, label: str) -> Path:
    """Validate a file or prefix path (rejecting directory paths)."""
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must not be empty or whitespace")
    if isinstance(value, str) and value.endswith(_DIR_LIKE_SUFFIXES):
        raise ValueError(
            f"{label} is a directory path, expected a file path: {value!r}"
        )
    path = _parse_path(value)
    if not path.name.strip() or path.name in (".", ".."):
        raise ValueError(
            f"{label} must include a non-empty file name, got {str(path)!r}"
        )
    return path


def _parse_output_prefix(value: Any) -> Path:
    """Validate an output file prefix path (non-empty file name)."""
    return validate_file_path(value, label="output_filename")


def _parse_input_filenames(value: Any) -> tuple[str, ...]:
    """Validate a non-empty sequence of non-blank input patterns."""
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


_NonNegFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_PosFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

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
    Path | None,
    BeforeValidator(_optional(_parse_path, allow_none_sentinel=True)),
    _AllowSentinelSchema(NONE_SENTINEL),
]
_OptionalChargeBuckets = Annotated[
    tuple[ChargeBucket, ...] | None,
    BeforeValidator(parse_optional_charge_buckets),
    _AllowSentinelSchema(AUTO_SENTINEL),
]

_STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


class PrecursorTolerance(BaseModel):
    """Precursor mass tolerance value and unit mode."""

    model_config = _STRICT_MODEL_CONFIG

    mass: _NonNegFloat = 20.0
    mode: PrecursorTolMode = "ppm"


class PreprocessingConfig(BaseModel):
    """Spectrum quality-filtering and normalization settings."""

    model_config = _STRICT_MODEL_CONFIG

    min_peaks: Annotated[int, Field(ge=1)] = 5
    min_mz_range: _NonNegFloat = 250.0
    min_mz: _NonNegFloat = DEFAULT_MIN_MZ
    max_mz: _PosFloat = DEFAULT_MAX_MZ
    remove_precursor_tol: _OptionalPosFloat = 1.5
    min_intensity: Annotated[
        float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.01
    max_peaks_used: _OptionalPosInt = 50
    scaling: ScalingMethod = "off"

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
    rt_tol: _OptionalNonNegFloat = None
    fragment_tol: _PosFloat = 0.05
    linkage: LinkageMethod = "complete"
    distance_threshold: Annotated[
        float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    ] = 0.1
    min_matched_peaks: Annotated[int, Field(ge=0)] = 0
    batch_size: Annotated[int, Field(ge=1)] = 2**15
    precursor_charge_buckets: _OptionalChargeBuckets = None


class MedoidConsensusConfig(BaseModel):
    """Medoid representative spectrum selection settings."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["medoid"] = "medoid"


class AverageConsensusConfig(BaseModel):
    """Averaged consensus spectrum with iterative sigma clipping."""

    model_config = _STRICT_MODEL_CONFIG

    method: Literal["average"] = "average"
    outlier_cutoff_lower: _PosFloat = DEFAULT_OUTLIER_CUTOFF_LOWER
    outlier_cutoff_upper: _PosFloat = DEFAULT_OUTLIER_CUTOFF_UPPER


ConsensusConfig = Annotated[
    MedoidConsensusConfig | AverageConsensusConfig,
    Field(discriminator="method"),
]


class _BaseIOConfig(BaseModel):
    """Shared optional I/O flags for partial and complete configs."""

    model_config = _STRICT_MODEL_CONFIG

    work_dir: _OptionalWorkDir = None
    overwrite: bool = False
    export_representatives: bool = False


class PartialIOConfig(_BaseIOConfig):
    """Input/output settings where paths may be omitted in overlays."""

    input_filenames: Annotated[
        tuple[str, ...] | None,
        BeforeValidator(_optional(_parse_input_filenames)),
    ] = None
    output_filename: Annotated[
        Path | None,
        BeforeValidator(_optional(_parse_output_prefix)),
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

    @property
    def csv_path(self) -> Path:
        """Return the output cluster-assignment CSV file path."""
        return Path(f"{self.output_filename}.csv")

    @property
    def mgf_path(self) -> Path:
        """Return the output cluster-representatives MGF file path."""
        return Path(f"{self.output_filename}.mgf")


class _BaseFalconConfig(BaseModel):
    """Shared scientific config sections and cross-section checks."""

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
        try:
            n_bins, _, _ = mz_bin_bounds(min_mz, max_mz, frag_tol)
        except ValueError as exc:
            raise ValueError(
                f"clustering.fragment_tol ({frag_tol}): {exc}"
            ) from exc
        if n_bins > INT32_MAX:
            raise ValueError(
                f"m/z bin count ({n_bins}) for [{min_mz}, {max_mz}] at "
                f"fragment_tol={frag_tol} exceeds int32 maximum ({INT32_MAX})"
            )
        if isinstance(self.consensus, AverageConsensusConfig):
            try:
                mz_bin_bounds(min_mz, max_mz, 2.0 * frag_tol)
            except ValueError as exc:
                raise ValueError(
                    f"clustering.fragment_tol ({frag_tol}) for average "
                    f"consensus: {exc}"
                ) from exc
        return self


class FalconSettings(_BaseFalconConfig):
    """Validated Falcon settings allowing omitted run I/O paths."""

    io: PartialIOConfig = Field(default_factory=PartialIOConfig)


class FalconConfig(_BaseFalconConfig):
    """Complete, immutable, validated configuration for a Falcon run."""

    io: IOConfig
