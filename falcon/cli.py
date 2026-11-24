"""Command-line interface, argument parsing, and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import TYPE_CHECKING, Any, TypeVar, get_args

from pydantic import ValidationError

from . import __version__
from .config import (
    AUTO_SENTINEL,
    NONE_SENTINEL,
    OTHER_BUCKET,
    UNKNOWN_CHARGE,
    ClusteringConfig,
    ConfigError,
    ConsensusConfig,
    ConsensusMethod,
    IOConfig,
    LinkageMethod,
    PrecursorTolerance,
    PrecursorTolMode,
    PreprocessingConfig,
    ScalingMethod,
    build_json_schema,
    format_toml,
    load_config,
    load_settings,
    sentinel_for_field,
    validate_file_path,
    write_saved_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pydantic import BaseModel


_T = TypeVar("_T")

_PRECURSOR_TOL_MODES: tuple[str, ...] = get_args(PrecursorTolMode)
_SCALING_CHOICES: tuple[str, ...] = get_args(ScalingMethod)
_LINKAGE_CHOICES: tuple[str, ...] = get_args(LinkageMethod)
_CONSENSUS_CHOICES: tuple[str, ...] = get_args(ConsensusMethod)

_DIRECT_SECTIONS: tuple[tuple[str, type[BaseModel], frozenset[str]], ...] = (
    ("io", IOConfig, frozenset()),
    ("preprocessing", PreprocessingConfig, frozenset()),
    (
        "clustering",
        ClusteringConfig,
        frozenset({"precursor_tol", "precursor_charge_buckets"}),
    ),
    ("consensus", ConsensusConfig, frozenset({"method"})),
)


def _configure_logger() -> None:
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "{asctime} {levelname} [{name}/{processName}] "
                "{module}.{funcName} : {message}",
                style="{",
            )
        )
        root.addHandler(handler)
    logging.getLogger("numba").setLevel(logging.WARNING)


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
    buckets: list[list[int | str] | str] = []
    for raw in stripped:
        if raw == OTHER_BUCKET:
            buckets.append(OTHER_BUCKET)
            continue
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
        buckets.append(bucket)
    return buckets


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
            "TOML config\nfile(s) via -c/--config (with optional 'extends') "
            "<- CLI flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        ConsensusConfig,
        "method",
        flag="--consensus-method",
        choices=_CONSENSUS_CHOICES,
    )
    _add_field_arg(cons, ConsensusConfig, "outlier_cutoff_lower", type=float)
    _add_field_arg(cons, ConsensusConfig, "outlier_cutoff_upper", type=float)
    return parser


def format_validation_error(exc: ValidationError) -> str:
    """Format a Pydantic ``ValidationError`` for display."""
    lines = ["Invalid configuration:"]
    for err in exc.errors(include_url=False):
        path = ".".join(str(elem) for elem in err["loc"]) or "config"
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  - {path}: {msg}")
    return "\n".join(lines)


def _extract_cli_overrides(ns: dict[str, Any]) -> dict[str, Any]:
    sections: dict[str, dict[str, Any]] = {
        name: {
            k: ns[k]
            for k in model.model_fields
            if k in ns and k not in excluded
        }
        for name, model, excluded in _DIRECT_SECTIONS
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
        sections["io"]["input_filenames"] = paths[:-1]
        sections["io"]["output_filename"] = paths[-1]

    if "precursor_tol" in ns:
        raw_mass, raw_mode = ns["precursor_tol"]
        try:
            mass_val = float(raw_mass)
        except ValueError as exc:
            raise ConfigError(
                f"--precursor-tol mass must be a number, got {raw_mass!r}"
            ) from exc
        sections["clustering"]["precursor_tol"] = {
            "mass": mass_val,
            "mode": raw_mode,
        }

    if "precursor_charge_buckets" in ns:
        sections["clustering"]["precursor_charge_buckets"] = (
            _parse_cli_charge_buckets(ns["precursor_charge_buckets"])
        )

    if "consensus_method" in ns:
        sections["consensus"]["method"] = ns["consensus_method"]

    return {name: values for name, values in sections.items() if values}


def main(args: Sequence[str] | None = None) -> int:
    """Run the Falcon CLI entry point and return an exit code."""
    parser = build_parser()
    ns = vars(parser.parse_args(list(args) if args is not None else None))

    if ns["dump_schema"]:
        print(json.dumps(build_json_schema(), indent=2))
        return 0

    config_files: list[str] = ns["config_files"]
    try:
        overrides = _extract_cli_overrides(ns)
        if ns["print_config"]:
            settings = load_settings(
                config_files=config_files, overrides=overrides
            )
            print(format_toml(settings), end="")
            return 0
        config = load_config(config_files=config_files, overrides=overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    except ValidationError as exc:
        parser.error(format_validation_error(exc))

    if ns["save_config"] is not None:
        try:
            write_saved_config(config, ns["save_config"])
        except ConfigError as exc:
            print(f"falcon: error: {exc}", file=sys.stderr)
            return 2

    # Defer importing heavy pipeline dependencies until an actual run.
    from .pipeline import run

    _configure_logger()
    try:
        return run(config)
    finally:
        for handler in logging.getLogger().handlers:
            handler.flush()
