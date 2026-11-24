"""Command-line interface, argument parsing, and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, get_args

from pydantic import ValidationError

from . import __version__
from .config import (
    AverageConsensusConfig,
    ClusteringConfig,
    ConfigError,
    ConsensusMethod,
    IOConfig,
    LinkageMethod,
    MedoidConsensusConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    ScalingMethod,
    build_json_schema,
    format_toml,
    load_config,
    load_settings,
    write_saved_config,
)
from .config.models import validate_file_path
from .types import AUTO_SENTINEL, NONE_SENTINEL
from .utils import configure_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pydantic import BaseModel

    from .config import FalconConfig, FalconSettings

__all__ = [
    "CliAction",
    "DumpSchemaAction",
    "NewlineTextHelpFormatter",
    "PrintConfigAction",
    "RunAction",
    "build_parser",
    "format_validation_error",
    "main",
    "parse_cli",
]

_T = TypeVar("_T")

_SCALING_CHOICES: tuple[str, ...] = get_args(ScalingMethod)
_LINKAGE_CHOICES: tuple[str, ...] = get_args(LinkageMethod)
_CONSENSUS_CHOICES: tuple[str, ...] = get_args(ConsensusMethod)

_CONSENSUS_UNION_TAGS = frozenset(
    {
        *_CONSENSUS_CHOICES,
        MedoidConsensusConfig.__name__,
        AverageConsensusConfig.__name__,
    }
)

_IO_DIRECT_KEYS = tuple(IOConfig.model_fields)
_PREP_KEYS = tuple(PreprocessingConfig.model_fields)
_CLUST_DIRECT_KEYS = tuple(
    k
    for k in ClusteringConfig.model_fields
    if k not in {"precursor_tol", "precursor_charge_buckets"}
)
_AVG_CUTOFF_KEYS = tuple(
    k for k in AverageConsensusConfig.model_fields if k != "method"
)


@dataclass(frozen=True, slots=True)
class DumpSchemaAction:
    """CLI action requesting JSON Schema output."""


@dataclass(frozen=True, slots=True)
class PrintConfigAction:
    """CLI action requesting resolved TOML output."""

    settings: FalconSettings


@dataclass(frozen=True, slots=True)
class RunAction:
    """CLI action executing a validated clustering run."""

    config: FalconConfig
    save_config: Path | None = None


CliAction = DumpSchemaAction | PrintConfigAction | RunAction


class NewlineTextHelpFormatter(argparse.HelpFormatter):
    """Help formatter that preserves explicit newlines in text."""

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return "\n".join(
            textwrap.fill(
                line,
                width,
                initial_indent=indent,
                subsequent_indent=indent,
                replace_whitespace=False,
            )
            for line in text.splitlines()
        )


def _add_field_arg(
    group: argparse._ArgumentGroup,
    model: type[BaseModel],
    field: str,
    help_text: str,
    *,
    flag: str | None = None,
    none_label: str = NONE_SENTINEL,
    **kwargs: Any,
) -> None:
    """Register one model-backed CLI flag with a synced default note."""
    default = model.model_fields[field].default
    default_str = none_label if default is None else str(default)
    option = flag if flag is not None else f"--{field.replace('_', '-')}"
    group.add_argument(
        option,
        help=f"{help_text} (default: {default_str}).",
        **kwargs,
    )


def _cli_int(raw: str) -> int:
    """Parse an integer CLI argument strictly."""
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {raw!r}"
        ) from exc


def _cli_float(raw: str) -> float:
    """Parse a floating-point CLI argument."""
    try:
        return float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a number, got {raw!r}"
        ) from exc


def _cli_save_path(raw: str) -> Path:
    """Validate raw ``--save-config`` path strings before ``Path()``."""
    try:
        return validate_file_path(raw, label="config output path")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _cli_optional(
    converter: Callable[[str], _T],
) -> Callable[[str], _T | str]:
    """Wrap ``converter`` to accept the ``"none"`` disable sentinel."""

    def _parse(raw: str) -> _T | str:
        return NONE_SENTINEL if raw == NONE_SENTINEL else converter(raw)

    return _parse


_cli_optional_float = _cli_optional(_cli_float)
_cli_optional_int = _cli_optional(_cli_int)


def _parse_cli_charge_buckets(
    tokens: Sequence[str],
) -> str | list[list[int | str] | str]:
    """Convert raw CLI bucket tokens into structured config values."""
    stripped = [tok.strip() for tok in tokens]
    if stripped == [AUTO_SENTINEL]:
        return AUTO_SENTINEL
    if AUTO_SENTINEL in stripped:
        raise ConfigError(
            f"Cannot combine '{AUTO_SENTINEL}' with explicit "
            "--precursor-charge-buckets"
        )

    buckets: list[list[int | str] | str] = []
    for raw in stripped:
        if raw == "other":
            buckets.append("other")
            continue
        inner = raw
        has_open = inner.startswith("[")
        has_close = inner.endswith("]")
        if has_open != has_close:
            raise ConfigError(f"Unbalanced brackets in charge bucket {raw!r}")
        if has_open:
            inner = inner[1:-1]
        if not inner.strip():
            raise ConfigError(f"Empty charge bucket: {raw!r}")

        bucket: list[int | str] = []
        for part in inner.split(","):
            part_clean = part.strip()
            if not part_clean:
                raise ConfigError(f"Empty charge entry in bucket {raw!r}")
            if part_clean == "unknown":
                bucket.append("unknown")
            else:
                try:
                    bucket.append(int(part_clean))
                except ValueError as exc:
                    raise ConfigError(
                        f"Invalid charge value {part_clean!r} in bucket "
                        f"{raw!r} (expected integer or 'unknown')"
                    ) from exc
        buckets.append(bucket)
    return buckets


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Register configuration loading and inspection flags."""
    group = parser.add_argument_group("Configuration")
    group.add_argument(
        "-c",
        "--config",
        action="append",
        dest="config_files",
        default=[],
        metavar="TOML",
        help=(
            "TOML configuration file(s), applied left-to-right. May be "
            "repeated; files may also declare 'extends'."
        ),
    )
    mode_group = group.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved configuration as TOML and exit.",
    )
    mode_group.add_argument(
        "--save-config",
        type=_cli_save_path,
        default=None,
        metavar="PATH",
        help="Write the resolved run configuration as TOML to PATH.",
    )
    mode_group.add_argument(
        "--dump-schema",
        action="store_true",
        default=False,
        help="Print the JSON Schema for TOML config files and exit.",
    )
    group.add_argument(
        "--version",
        action="version",
        version=f"falcon {__version__}",
    )


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    """Register input, output, and working-directory CLI arguments."""
    group = parser.add_argument_group("I/O")
    group.add_argument(
        "paths",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Input peak file(s) (.mzML, .mzXML, .MGF) followed by the "
            "output prefix (may be omitted when defined in [io])."
        ),
    )
    group.add_argument(
        "-i",
        "--input",
        action="extend",
        nargs="+",
        dest="input_filenames",
        metavar="FILE",
        help=(
            "Input peak file(s) or globs (alternative to positional paths; "
            "may be repeated)."
        ),
    )
    group.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        metavar="PREFIX",
        help="Output file prefix (alternative to positional paths).",
    )
    _add_field_arg(
        group,
        IOConfig,
        "work_dir",
        f"Working directory for Lance staging cache, or '{NONE_SENTINEL}' "
        "for a temporary directory",
        metavar=f"DIR|{NONE_SENTINEL}",
    )
    _add_field_arg(
        group,
        IOConfig,
        "overwrite",
        "Overwrite existing results and cached staging files",
        action=argparse.BooleanOptionalAction,
    )
    _add_field_arg(
        group,
        IOConfig,
        "export_representatives",
        "Export cluster representative spectra to an MGF file",
        action=argparse.BooleanOptionalAction,
    )


def _add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """Register spectrum preprocessing flags."""
    group = parser.add_argument_group("Preprocessing")
    _add_field_arg(
        group,
        PreprocessingConfig,
        "min_peaks",
        "Minimum number of peaks per spectrum",
        type=_cli_int,
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "min_mz_range",
        "Minimum fragment m/z span per spectrum",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "min_mz",
        "Minimum peak m/z inclusive",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "max_mz",
        "Maximum peak m/z inclusive",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "remove_precursor_tol",
        f"Window (Da) around precursor m/z to remove peaks, or "
        f"'{NONE_SENTINEL}' to disable",
        type=_cli_optional_float,
        metavar=f"DA|{NONE_SENTINEL}",
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "min_intensity",
        "Minimum peak intensity relative to base peak in [0, 1)",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "max_peaks_used",
        f"Maximum most-intense peaks to retain, or '{NONE_SENTINEL}' for all",
        type=_cli_optional_int,
        metavar=f"N|{NONE_SENTINEL}",
    )
    _add_field_arg(
        group,
        PreprocessingConfig,
        "scaling",
        "Peak intensity scaling method",
        choices=_SCALING_CHOICES,
    )


def _add_clustering_args(parser: argparse.ArgumentParser) -> None:
    """Register spectrum comparison and clustering flags."""
    tol_mass = PrecursorTolerance.model_fields["mass"].default
    tol_mode = PrecursorTolerance.model_fields["mode"].default
    group = parser.add_argument_group("Clustering")
    group.add_argument(
        "--precursor-tol",
        nargs=2,
        metavar=("MASS", "MODE"),
        help=(
            'Precursor mass tolerance and unit ("ppm" or "Da", '
            f"default: {tol_mass:g} {tol_mode})."
        ),
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "rt_tol",
        f"Retention time tolerance, or '{NONE_SENTINEL}' to disable",
        type=_cli_optional_float,
        metavar=f"TOL|{NONE_SENTINEL}",
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "fragment_tol",
        "Fragment mass tolerance in m/z",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "linkage",
        "Hierarchical clustering linkage criterion",
        choices=_LINKAGE_CHOICES,
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "distance_threshold",
        "Cosine distance threshold in (0, 1)",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "min_matched_peaks",
        "Minimum matched fragment peaks",
        type=_cli_int,
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "batch_size",
        "Maximum spectra per precursor m/z batch",
        type=_cli_int,
    )
    _add_field_arg(
        group,
        ClusteringConfig,
        "precursor_charge_buckets",
        "Charge buckets such as '[1]' '[2, 3]' unknown other, or "
        f"'{AUTO_SENTINEL}' to cluster each observed charge separately",
        none_label=AUTO_SENTINEL,
        nargs="+",
        metavar="BUCKET",
    )


def _add_consensus_args(parser: argparse.ArgumentParser) -> None:
    """Register representative spectrum / consensus flags."""
    group = parser.add_argument_group("Consensus")
    _add_field_arg(
        group,
        MedoidConsensusConfig,
        "method",
        "Representative spectrum method",
        flag="--consensus-method",
        choices=_CONSENSUS_CHOICES,
    )
    _add_field_arg(
        group,
        AverageConsensusConfig,
        "outlier_cutoff_lower",
        "Lower sigma cutoff for consensus averaging (only valid when "
        "consensus.method = 'average')",
        type=_cli_float,
    )
    _add_field_arg(
        group,
        AverageConsensusConfig,
        "outlier_cutoff_upper",
        "Upper sigma cutoff for consensus averaging (only valid when "
        "consensus.method = 'average')",
        type=_cli_float,
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the Falcon CLI ``ArgumentParser``.

    Uses ``argument_default=argparse.SUPPRESS`` so omitted CLI flags
    never overwrite values loaded from TOML config layers.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="falcon",
        argument_default=argparse.SUPPRESS,
        description=(
            "falcon: Large-scale tandem mass spectrum clustering\n"
            "===================================================\n\n"
            f"falcon version {__version__}\n\n"
            "Official code website: "
            "https://github.com/bittremieuxlab/falcon\n\n"
            "Configuration layers apply in order: built-in defaults <- "
            "TOML config file(s) via -c/--config (with optional 'extends') "
            "<- CLI flags."
        ),
        formatter_class=NewlineTextHelpFormatter,
    )
    _add_config_args(parser)
    _add_io_args(parser)
    _add_preprocessing_args(parser)
    _add_clustering_args(parser)
    _add_consensus_args(parser)
    return parser


def _format_error_loc(loc: tuple[int | str, ...]) -> str:
    """Format a Pydantic error location tuple as a clean dotted path."""
    parts: list[str] = []
    for idx, elem in enumerate(loc):
        text = str(elem)
        if (
            idx > 0
            and str(loc[idx - 1]) == "consensus"
            and text in _CONSENSUS_UNION_TAGS
        ):
            continue
        parts.append(text)
    return ".".join(parts) if parts else "config"


def format_validation_error(exc: ValidationError) -> str:
    """
    Format a Pydantic ``ValidationError`` into readable CLI diagnostics.

    Parameters
    ----------
    exc : ValidationError
        Validation error raised when constructing a config model.

    Returns
    -------
    str
        Human-readable multi-line error summary with dotted field paths.
    """
    lines = ["Invalid configuration:"]
    for err in exc.errors(include_url=False):
        path = _format_error_loc(err["loc"])
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        lines.append(f"  - {path}: {msg}")
    return "\n".join(lines)


def _extract_cli_overrides(ns: dict[str, Any]) -> dict[str, Any]:
    """Build a sparse nested override dict from parsed CLI args."""
    io_ov = {k: ns[k] for k in _IO_DIRECT_KEYS if k in ns}
    prep_ov = {k: ns[k] for k in _PREP_KEYS if k in ns}
    clust_ov = {k: ns[k] for k in _CLUST_DIRECT_KEYS if k in ns}
    cons_ov = {k: ns[k] for k in _AVG_CUTOFF_KEYS if k in ns}

    paths: list[str] = ns.get("paths", [])
    has_named_io = "input_filenames" in ns or "output_filename" in ns
    if paths and has_named_io:
        raise ConfigError(
            "Cannot mix positional input/output paths with -i/--input or "
            "-o/--output"
        )
    if len(paths) == 1:
        raise ConfigError(
            "Expected at least one input file and an output prefix, or "
            "neither when specifying them in a config file"
        )
    if len(paths) >= 2:
        io_ov["input_filenames"] = paths[:-1]
        io_ov["output_filename"] = paths[-1]

    if "precursor_tol" in ns:
        raw_mass, raw_mode = ns["precursor_tol"]
        try:
            mass_val = float(raw_mass)
        except ValueError as exc:
            raise ConfigError(
                f"--precursor-tol mass must be a number, got {raw_mass!r}"
            ) from exc
        clust_ov["precursor_tol"] = {"mass": mass_val, "mode": raw_mode}

    if "precursor_charge_buckets" in ns:
        clust_ov["precursor_charge_buckets"] = _parse_cli_charge_buckets(
            ns["precursor_charge_buckets"]
        )

    if "consensus_method" in ns:
        cons_ov["method"] = ns["consensus_method"]

    sections = {
        "io": io_ov,
        "preprocessing": prep_ov,
        "clustering": clust_ov,
        "consensus": cons_ov,
    }
    return {name: values for name, values in sections.items() if values}


def parse_cli(args: Sequence[str] | None = None) -> CliAction:
    """
    Parse CLI arguments and resolve layered TOML + CLI configuration.

    Parameters
    ----------
    args : Sequence[str] | None, default=None
        Optional CLI argument list. When ``None``, reads
        ``sys.argv[1:]``.

    Returns
    -------
    CliAction
        Typed action descriptor (``DumpSchemaAction``,
        ``PrintConfigAction``, or ``RunAction``).
    """
    parser = build_parser()
    parsed = parser.parse_args(list(args) if args is not None else None)
    ns = vars(parsed)

    if ns["dump_schema"]:
        return DumpSchemaAction()

    print_config: bool = ns["print_config"]
    save_config: Path | None = ns["save_config"]
    config_files: list[str] = ns["config_files"]

    try:
        overrides = _extract_cli_overrides(ns)
        if print_config:
            settings = load_settings(
                config_files=config_files, overrides=overrides
            )
            return PrintConfigAction(settings=settings)
        config = load_config(config_files=config_files, overrides=overrides)
        return RunAction(config=config, save_config=save_config)
    except ConfigError as exc:
        parser.error(str(exc))
    except ValidationError as exc:
        parser.error(format_validation_error(exc))


def main(args: Sequence[str] | None = None) -> int:
    """
    Run the Falcon CLI entry point.

    Parameters
    ----------
    args : Sequence[str] | None, default=None
        Optional CLI arguments. When ``None``, reads ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (``0`` on success, ``1`` on existing-output
        abort, ``2`` on post-parse configuration/serialization error;
        argument parsing errors exit with status ``2`` via
        ``SystemExit``).
    """
    action = parse_cli(args)
    if isinstance(action, DumpSchemaAction):
        print(json.dumps(build_json_schema(), indent=2))
        return 0
    if isinstance(action, PrintConfigAction):
        try:
            print(format_toml(action.settings), end="")
        except ConfigError as exc:
            print(f"falcon: error: {exc}", file=sys.stderr)
            return 2
        return 0

    if action.save_config is not None:
        try:
            write_saved_config(action.config, action.save_config)
        except ConfigError as exc:
            print(f"falcon: error: {exc}", file=sys.stderr)
            return 2

    # Defer importing heavy pipeline dependencies until an actual run.
    from .pipeline import run

    configure_logger()
    try:
        return run(action.config)
    finally:
        for handler in logging.getLogger().handlers:
            handler.flush()
