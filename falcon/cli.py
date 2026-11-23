"""Command-line interface, argument parsing, and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from . import __version__
from .config import (
    AverageConsensusConfig,
    ClusteringConfig,
    ConfigError,
    FalconConfig,
    FalconSettings,
    MedoidConsensusConfig,
    PreprocessingConfig,
    build_json_schema,
    load_config,
    load_settings,
    write_saved_config,
)
from .utils import configure_logger

_T = TypeVar("_T")

_CONSENSUS_UNION_TAGS = frozenset(
    {
        "medoid",
        "average",
        MedoidConsensusConfig.__name__,
        AverageConsensusConfig.__name__,
    }
)

# Model defaults used to render accurate CLI help text without drift.
_DEFAULT_PREP = PreprocessingConfig()
_DEFAULT_CLUST = ClusteringConfig()
_DEFAULT_AVG = AverageConsensusConfig()

_IO_DIRECT_KEYS = (
    "input_filenames",
    "output_filename",
    "work_dir",
    "overwrite",
    "export_representatives",
)
_PREP_KEYS = tuple(PreprocessingConfig.model_fields)
_CLUST_DIRECT_KEYS = (
    "rt_tol",
    "fragment_tol",
    "linkage",
    "distance_threshold",
    "min_matched_peaks",
    "batch_size",
)
_AVG_CUTOFF_KEYS = ("outlier_cutoff_lower", "outlier_cutoff_upper")


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
            ).strip()
            for line in text.splitlines(keepends=True)
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


def _cli_optional(
    converter: Callable[[str], _T],
) -> Callable[[str], _T | str]:
    """Wrap ``converter`` to accept the ``"none"`` disable sentinel."""

    def _parse(raw: str) -> _T | str:
        return "none" if raw == "none" else converter(raw)

    return _parse


_cli_optional_float = _cli_optional(_cli_float)
_cli_optional_int = _cli_optional(_cli_int)


def _parse_cli_charge_buckets(
    tokens: Sequence[str],
) -> str | list[list[int | str] | str]:
    """Convert raw CLI bucket tokens into structured config values."""
    stripped = [tok.strip() for tok in tokens]
    if stripped == ["auto"]:
        return "auto"
    if "auto" in stripped:
        raise ConfigError(
            "Cannot combine 'auto' with explicit --precursor-charge-buckets"
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
        if has_open and has_close:
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
    group.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved configuration as TOML and exit.",
    )
    group.add_argument(
        "--save-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the resolved run configuration as TOML to PATH.",
    )
    group.add_argument(
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
        metavar="PATH",
        help=(
            "Input peak file(s) (.mzML, .mzXML, .MGF) followed by the "
            "output prefix (may be omitted when defined in [io])."
        ),
    )
    group.add_argument(
        "-i",
        "--input",
        nargs="+",
        dest="input_filenames",
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Input peak file(s) or globs (alternative to positional paths).",
    )
    group.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        default=argparse.SUPPRESS,
        metavar="PREFIX",
        help="Output file prefix (alternative to positional paths).",
    )
    group.add_argument(
        "--work-dir",
        default=argparse.SUPPRESS,
        metavar="DIR|none",
        help=(
            "Working directory for Lance staging cache, or 'none' for a "
            "temporary directory."
        ),
    )
    group.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Overwrite existing results and cached staging files.",
    )
    group.add_argument(
        "--export-representatives",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Export cluster representative spectra to an MGF file.",
    )


def _add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """Register spectrum preprocessing flags."""
    p = _DEFAULT_PREP
    group = parser.add_argument_group("Preprocessing")
    group.add_argument(
        "--min-peaks",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help=f"Minimum number of peaks per spectrum (default: {p.min_peaks}).",
    )
    group.add_argument(
        "--min-mz-range",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            f"Minimum fragment m/z span per spectrum "
            f"(default: {p.min_mz_range})."
        ),
    )
    group.add_argument(
        "--min-mz",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=f"Minimum peak m/z inclusive (default: {p.min_mz}).",
    )
    group.add_argument(
        "--max-mz",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=f"Maximum peak m/z inclusive (default: {p.max_mz}).",
    )
    group.add_argument(
        "--remove-precursor-tol",
        type=_cli_optional_float,
        default=argparse.SUPPRESS,
        metavar="DA|none",
        help=(
            "Window (Da) around precursor m/z to remove peaks, or 'none' "
            f"to disable (default: {p.remove_precursor_tol})."
        ),
    )
    group.add_argument(
        "--min-intensity",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Minimum peak intensity relative to base peak in [0, 1) "
            f"(default: {p.min_intensity})."
        ),
    )
    group.add_argument(
        "--max-peaks-used",
        type=_cli_optional_int,
        default=argparse.SUPPRESS,
        metavar="N|none",
        help=(
            "Maximum most-intense peaks to retain, or 'none' for all "
            f"(default: {p.max_peaks_used})."
        ),
    )
    group.add_argument(
        "--scaling",
        choices=("off", "root", "log", "rank"),
        default=argparse.SUPPRESS,
        help=f"Peak intensity scaling method (default: {p.scaling}).",
    )


def _add_clustering_args(parser: argparse.ArgumentParser) -> None:
    """Register spectrum comparison and clustering flags."""
    c = _DEFAULT_CLUST
    group = parser.add_argument_group("Clustering")
    group.add_argument(
        "--precursor-tol",
        nargs=2,
        metavar=("MASS", "MODE"),
        default=argparse.SUPPRESS,
        help=(
            'Precursor mass tolerance and unit ("ppm" or "Da", '
            f"default: {int(c.precursor_tol.mass)} {c.precursor_tol.mode})."
        ),
    )
    group.add_argument(
        "--rt-tol",
        type=_cli_optional_float,
        default=argparse.SUPPRESS,
        metavar="TOL|none",
        help="Retention time tolerance, or 'none' to disable (default: none).",
    )
    group.add_argument(
        "--fragment-tol",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=f"Fragment mass tolerance in m/z (default: {c.fragment_tol}).",
    )
    group.add_argument(
        "--linkage",
        choices=("single", "complete", "average"),
        default=argparse.SUPPRESS,
        help=(
            f"Hierarchical clustering linkage criterion "
            f"(default: {c.linkage})."
        ),
    )
    group.add_argument(
        "--distance-threshold",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            f"Cosine distance threshold in (0, 1) "
            f"(default: {c.distance_threshold})."
        ),
    )
    group.add_argument(
        "--min-matched-peaks",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help=(
            f"Minimum matched fragment peaks (default: {c.min_matched_peaks})."
        ),
    )
    group.add_argument(
        "--batch-size",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help=(
            f"Maximum spectra per precursor m/z batch "
            f"(default: {c.batch_size})."
        ),
    )
    group.add_argument(
        "--precursor-charge-buckets",
        nargs="+",
        metavar="BUCKET",
        default=argparse.SUPPRESS,
        help=(
            "Charge buckets such as '[1]' '[2, 3]' unknown other, or "
            "'auto' to cluster each observed charge separately."
        ),
    )


def _add_consensus_args(parser: argparse.ArgumentParser) -> None:
    """Register representative spectrum / consensus flags."""
    a = _DEFAULT_AVG
    group = parser.add_argument_group("Consensus")
    group.add_argument(
        "--consensus-method",
        dest="consensus_method",
        choices=("medoid", "average"),
        default=argparse.SUPPRESS,
        help="Representative spectrum method (default: medoid).",
    )
    group.add_argument(
        "--outlier-cutoff-lower",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Lower sigma cutoff for consensus averaging (only valid when "
            f"consensus.method = 'average'; default: "
            f"{a.outlier_cutoff_lower})."
        ),
    )
    group.add_argument(
        "--outlier-cutoff-upper",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Upper sigma cutoff for consensus averaging (only valid when "
            f"consensus.method = 'average'; default: "
            f"{a.outlier_cutoff_upper})."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the Falcon CLI ``ArgumentParser``.

    All configurable options default to ``argparse.SUPPRESS`` so omitted
    CLI flags never overwrite values loaded from TOML config layers.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="falcon",
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

    dump_schema: bool = ns["dump_schema"]
    print_config: bool = ns["print_config"]
    save_config: Path | None = ns["save_config"]
    config_files: list[str] = ns["config_files"]

    if dump_schema and print_config:
        parser.error("Cannot combine --dump-schema and --print-config")
    if dump_schema and save_config is not None:
        parser.error("Cannot combine --dump-schema and --save-config")
    if print_config and save_config is not None:
        parser.error(
            "Cannot combine --print-config and --save-config "
            "(redirect --print-config output to a file instead)"
        )

    if dump_schema:
        return DumpSchemaAction()

    if (
        save_config is not None
        and save_config.exists()
        and save_config.is_dir()
    ):
        parser.error(
            f"--save-config path is a directory, expected a file path: "
            f"{save_config}"
        )

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
        abort, ``2`` on configuration error).
    """
    action = parse_cli(args)
    if isinstance(action, DumpSchemaAction):
        print(json.dumps(build_json_schema(), indent=2))
        return 0
    if isinstance(action, PrintConfigAction):
        print(action.settings.to_toml(), end="")
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
        logging.shutdown()
