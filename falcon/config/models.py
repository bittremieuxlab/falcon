"""Strict frozen Pydantic configuration models for Falcon."""

from __future__ import annotations

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

from ..types import (
    AUTO_SENTINEL,
    DEFAULT_MAX_MZ,
    DEFAULT_MIN_MZ,
    DEFAULT_OUTLIER_CUTOFF_LOWER,
    DEFAULT_OUTLIER_CUTOFF_UPPER,
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    NONE_SENTINEL,
    ConsensusMethod,
    LinkageMethod,
    PrecursorTolMode,
    ScalingMethod,
    mz_bin_bounds,
    parse_optional_charge_buckets,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import GetJsonSchemaHandler
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

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
    "sentinel_for_field",
    "validate_file_path",
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
    """
    Return the TOML/CLI disable sentinel configured on ``field_info``.

    Parameters
    ----------
    field_info : FieldInfo
        Pydantic field metadata object.

    Returns
    -------
    str | None
        Configured sentinel literal (e.g. ``"none"`` or ``"auto"``), or
        ``None`` if the field has no sentinel.
    """
    for meta in field_info.metadata:
        if isinstance(meta, _AllowSentinelSchema):
            return meta.sentinel
    return None


def _none_sentinel(value: Any) -> Any:
    """Normalize the TOML/CLI ``"none"`` sentinel to ``None``."""
    return None if value == NONE_SENTINEL else value


def _default_consensus_method(value: Any) -> Any:
    """Default an omitted ``consensus.method`` key to ``"medoid"``."""
    if isinstance(value, Mapping) and "method" not in value:
        return {"method": "medoid", **value}
    return value


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


def _parse_path(value: Any, *, label: str = "Path") -> Path:
    """Validate and convert a non-empty string or Path to ``Path``."""
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"Expected a path string or Path for {label}, got "
            f"{type(value).__name__}"
        )
    if not str(value).strip():
        raise ValueError(f"{label} must not be empty or whitespace")
    return value if isinstance(value, Path) else Path(value)


def validate_file_path(value: Any, *, label: str) -> Path:
    """
    Validate a file or prefix path, rejecting directory-like paths.

    Parameters
    ----------
    value : Any
        Raw path string or ``Path`` instance.
    label : str
        Human-readable label included in validation errors.

    Returns
    -------
    Path
        Validated file or prefix path.
    """
    if isinstance(value, str) and value.endswith(_DIR_LIKE_SUFFIXES):
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
_NonEmptyStr = Annotated[str, Field(json_schema_extra={"minLength": 1})]
_NonEmptyPath = Annotated[Path, Field(json_schema_extra={"minLength": 1})]
_InputPatterns = Annotated[tuple[_NonEmptyStr, ...], Field(min_length=1)]

_SchemaChargeInt = Annotated[int, Field(ge=INT8_MIN, le=INT8_MAX)]
_SchemaChargeGroup = Annotated[
    tuple[_SchemaChargeInt | Literal["unknown"], ...],
    Field(min_length=1, json_schema_extra={"uniqueItems": True}),
]
_SchemaChargeBucket = _SchemaChargeGroup | Literal["other"]
_SchemaChargeBuckets = Annotated[
    tuple[_SchemaChargeBucket, ...], Field(min_length=1)
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

_STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


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


class _BaseIOConfig(BaseModel):
    """Shared optional I/O flags for partial and complete configs."""

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
