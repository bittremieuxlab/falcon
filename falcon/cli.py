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
    IOConfig,
    MedoidConsensusConfig,
    PrecursorTolerance,
    PreprocessingConfig,
    build_json_schema,
    format_toml,
    load_config,
    load_settings,
    sentinel_for_field,
    validate_file_path,
    write_saved_config,
)
from .types import (
    AUTO_SENTINEL,
    NONE_SENTINEL,
    OTHER_BUCKET,
    UNKNOWN_CHARGE,
    ConsensusMethod,
    LinkageMethod,
    PrecursorTolMode,
    ScalingMethod,
)
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

_PRECURSOR_TOL_MODES: tuple[str, ...] = get_args(PrecursorTolMode)
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
    help_text: str | None = None,
    *,
    flag: str | None = None,
    **kwargs: Any,
) -> None:
    field_info = model.model_fields[field]
    default = field_info.default
    default_str = (
        (sentinel_for_field(field_info) or NONE_SENTINEL)
        if default is None
        else str(default)
    )
    text = field_info.description if help_text is None else help_text
    option = flag if flag is not None else f"--{field.replace('_', '-')}"
    group.add_argument(
        option, help=f"{text} (default: {default_str}).", **kwargs
    )


def _cli_save_path(raw: str) -> Path:
    try:
        return validate_file_path(raw, label="Config output path")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _cli_optional(converter: Callable[[str], _T]) -> Callable[[str], _T | str]:
    def _parse(raw: str) -> _T | str:
        if raw == NONE_SENTINEL:
            return NONE_SENTINEL
        try:
            return converter(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid value: {raw!r}"
            ) from exc

    return _parse


def _parse_cli_charge_bucket(raw: str) -> list[int | str] | str:
    if raw == OTHER_BUCKET:
        return OTHER_BUCKET
    has_open, has_close = raw.startswith("["), raw.endswith("]")
    if has_open != has_close:
        raise ConfigError(f"Unbalanced brackets in charge bucket {raw!r}")
    inner = raw[1:-1] if has_open else raw
    if not inner.strip():
        raise ConfigError(f"Empty charge bucket: {raw!r}")

    bucket: list[int | str] = []
    for part in inner.split(","):
        token = part.strip()
        if not token:
            raise ConfigError(f"Empty charge entry in bucket {raw!r}")
        if token == UNKNOWN_CHARGE:
            bucket.append(UNKNOWN_CHARGE)
            continue
        try:
            bucket.append(int(token))
        except ValueError as exc:
            raise ConfigError(
                f"Invalid charge value {token!r} in bucket {raw!r} "
                f"(expected integer or '{UNKNOWN_CHARGE}')"
            ) from exc
    return bucket


def _parse_cli_charge_buckets(
    tokens: Sequence[str],
) -> str | list[list[int | str] | str]:
    stripped = [tok.strip() for tok in tokens]
    if stripped == [AUTO_SENTINEL]:
        return AUTO_SENTINEL
    if AUTO_SENTINEL in stripped:
        raise ConfigError(
            f"Cannot combine '{AUTO_SENTINEL}' with explicit "
            "--precursor-charge-buckets"
        )
    return [_parse_cli_charge_bucket(raw) for raw in stripped]


def build_parser() -> argparse.ArgumentParser:
    """
    Build the Falcon CLI ``ArgumentParser``.

    Uses ``argument_default=argparse.SUPPRESS`` so omitted CLI flags
    never overwrite values loaded from TOML config layers.
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

    cfg = parser.add_argument_group("Configuration")
    cfg.add_argument(
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
    mode = cfg.add_mutually_exclusive_group()
    mode.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved configuration as TOML and exit.",
    )
    mode.add_argument(
        "--save-config",
        type=_cli_save_path,
        default=None,
        metavar="PATH",
        help="Write the resolved run configuration as TOML to PATH.",
    )
    mode.add_argument(
        "--dump-schema",
        action="store_true",
        default=False,
        help="Print the JSON Schema for TOML config files and exit.",
    )
    cfg.add_argument(
        "--version", action="version", version=f"falcon {__version__}"
    )

    io_grp = parser.add_argument_group("I/O")
    io_grp.add_argument(
        "paths",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Input peak file(s) (.mzML, .mzXML, .MGF) followed by the "
            "output prefix (may be omitted when defined in [io])."
        ),
    )
    io_grp.add_argument(
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
    io_grp.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        metavar="PREFIX",
        help="Output file prefix (alternative to positional paths).",
    )
    _add_field_arg(
        io_grp, IOConfig, "work_dir", metavar=f"DIR|{NONE_SENTINEL}"
    )
    _add_field_arg(
        io_grp, IOConfig, "overwrite", action=argparse.BooleanOptionalAction
    )
    _add_field_arg(
        io_grp,
        IOConfig,
        "export_representatives",
        action=argparse.BooleanOptionalAction,
    )

    prep = parser.add_argument_group("Preprocessing")
    _add_field_arg(prep, PreprocessingConfig, "min_peaks", type=int)
    _add_field_arg(prep, PreprocessingConfig, "min_mz_range", type=float)
    _add_field_arg(prep, PreprocessingConfig, "min_mz", type=float)
    _add_field_arg(prep, PreprocessingConfig, "max_mz", type=float)
    _add_field_arg(
        prep,
        PreprocessingConfig,
        "remove_precursor_tol",
        type=_cli_optional(float),
        metavar=f"DA|{NONE_SENTINEL}",
    )
    _add_field_arg(prep, PreprocessingConfig, "min_intensity", type=float)
    _add_field_arg(
        prep,
        PreprocessingConfig,
        "max_peaks_used",
        type=_cli_optional(int),
        metavar=f"N|{NONE_SENTINEL}",
    )
    _add_field_arg(
        prep, PreprocessingConfig, "scaling", choices=_SCALING_CHOICES
    )

    clust = parser.add_argument_group("Clustering")
    tol_mass = PrecursorTolerance.model_fields["mass"].default
    tol_mode = PrecursorTolerance.model_fields["mode"].default
    modes_desc = " or ".join(f'"{m}"' for m in _PRECURSOR_TOL_MODES)
    clust.add_argument(
        "--precursor-tol",
        nargs=2,
        metavar=("MASS", "MODE"),
        help=(
            f"Precursor mass tolerance and unit ({modes_desc}, "
            f"default: {tol_mass:g} {tol_mode})."
        ),
    )
    _add_field_arg(
        clust,
        ClusteringConfig,
        "rt_tol",
        type=_cli_optional(float),
        metavar=f"TOL|{NONE_SENTINEL}",
    )
    _add_field_arg(clust, ClusteringConfig, "fragment_tol", type=float)
    _add_field_arg(
        clust, ClusteringConfig, "linkage", choices=_LINKAGE_CHOICES
    )
    _add_field_arg(clust, ClusteringConfig, "distance_threshold", type=float)
    _add_field_arg(clust, ClusteringConfig, "min_matched_peaks", type=int)
    _add_field_arg(clust, ClusteringConfig, "batch_size", type=int)
    _add_field_arg(
        clust,
        ClusteringConfig,
        "precursor_charge_buckets",
        "Charge buckets such as '[1]' '[2, 3]' unknown other, or "
        f"'{AUTO_SENTINEL}' to cluster each observed charge separately",
        nargs="+",
        metavar="BUCKET",
    )

    cons = parser.add_argument_group("Consensus")
    _add_field_arg(
        cons,
        MedoidConsensusConfig,
        "method",
        flag="--consensus-method",
        choices=_CONSENSUS_CHOICES,
    )
    _add_field_arg(
        cons, AverageConsensusConfig, "outlier_cutoff_lower", type=float
    )
    _add_field_arg(
        cons, AverageConsensusConfig, "outlier_cutoff_upper", type=float
    )
    return parser


def format_validation_error(exc: ValidationError) -> str:
    """Format a Pydantic ``ValidationError`` for display."""
    lines = ["Invalid configuration:"]
    for err in exc.errors(include_url=False):
        loc = err["loc"]
        parts = [
            str(elem)
            for idx, elem in enumerate(loc)
            if not (
                idx > 0
                and str(loc[idx - 1]) == "consensus"
                and str(elem) in _CONSENSUS_UNION_TAGS
            )
        ]
        path = ".".join(parts) if parts else "config"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  - {path}: {msg}")
    return "\n".join(lines)


def _extract_cli_overrides(ns: dict[str, Any]) -> dict[str, Any]:
    io_ov = {k: ns[k] for k in IOConfig.model_fields if k in ns}
    prep_ov = {k: ns[k] for k in PreprocessingConfig.model_fields if k in ns}
    clust_ov = {
        k: ns[k]
        for k in ClusteringConfig.model_fields
        if k in ns and k not in {"precursor_tol", "precursor_charge_buckets"}
    }
    cons_ov = {
        k: ns[k]
        for k in AverageConsensusConfig.model_fields
        if k in ns and k != "method"
    }

    paths: list[str] = ns.get("paths", [])
    if paths and ("input_filenames" in ns or "output_filename" in ns):
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
    """Parse CLI arguments and resolve layered TOML + CLI config."""
    parser = build_parser()
    ns = vars(parser.parse_args(list(args) if args is not None else None))

    if ns["dump_schema"]:
        return DumpSchemaAction()

    config_files: list[str] = ns["config_files"]
    try:
        overrides = _extract_cli_overrides(ns)
        if ns["print_config"]:
            return PrintConfigAction(
                settings=load_settings(
                    config_files=config_files, overrides=overrides
                )
            )
        return RunAction(
            config=load_config(config_files=config_files, overrides=overrides),
            save_config=ns["save_config"],
        )
    except ConfigError as exc:
        parser.error(str(exc))
    except ValidationError as exc:
        parser.error(format_validation_error(exc))


def main(args: Sequence[str] | None = None) -> int:
    """Run the Falcon CLI entry point and return an exit code."""
    action = parse_cli(args)
    if isinstance(action, DumpSchemaAction):
        print(json.dumps(build_json_schema(), indent=2))
        return 0
    try:
        if isinstance(action, PrintConfigAction):
            print(format_toml(action.settings), end="")
            return 0
        if action.save_config is not None:
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
