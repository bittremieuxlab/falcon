"""Command-line argument parsing and error formatting for Falcon."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from falcon import __version__
from falcon.config import (
    ConfigError,
    FalconConfig,
    FalconSettings,
    load_config,
    load_settings,
)

_CONSENSUS_UNION_TAGS = frozenset(
    {
        "medoid",
        "average",
        "MedoidConsensusConfig",
        "AverageConsensusConfig",
    }
)


@dataclass(frozen=True, slots=True)
class CliResult:
    """Resolved CLI action and validated configuration payload."""

    action: Literal["run", "print_config", "dump_schema"]
    config: FalconConfig | None = None
    settings: FalconSettings | None = None
    save_config: Path | None = None


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


def _cli_optional_float(raw: str) -> float | str:
    """Parse a float CLI argument or the ``"none"`` disable sentinel."""
    if raw == "none":
        return "none"
    return _cli_float(raw)


def _cli_optional_int(raw: str) -> int | str:
    """Parse an integer CLI argument or the ``"none"`` disable sentinel."""
    if raw == "none":
        return "none"
    return _cli_int(raw)


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

    # Configuration & inspection
    cfg_group = parser.add_argument_group("Configuration")
    cfg_group.add_argument(
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
    cfg_group.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved configuration as TOML and exit.",
    )
    cfg_group.add_argument(
        "--save-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the resolved run configuration as TOML to PATH.",
    )
    cfg_group.add_argument(
        "--dump-schema",
        action="store_true",
        default=False,
        help="Print the JSON Schema for TOML config files and exit.",
    )
    cfg_group.add_argument(
        "--version",
        action="version",
        version=f"falcon {__version__}",
    )

    # I/O
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help=(
            "Input peak file(s) (.mzML, .mzXML, .MGF) followed by the "
            "output prefix (may be omitted when defined in [io])."
        ),
    )
    io_group.add_argument(
        "-i",
        "--input",
        nargs="+",
        dest="input_filenames",
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Input peak file(s) or globs (alternative to positional paths).",
    )
    io_group.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        default=argparse.SUPPRESS,
        metavar="PREFIX",
        help="Output file prefix (alternative to positional paths).",
    )
    io_group.add_argument(
        "--work-dir",
        default=argparse.SUPPRESS,
        metavar="DIR|none",
        help=(
            "Working directory for Lance staging cache, or 'none' for a "
            "temporary directory."
        ),
    )
    io_group.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Overwrite existing results and cached staging files.",
    )
    io_group.add_argument(
        "--export-representatives",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Export cluster representative spectra to an MGF file.",
    )

    # Preprocessing
    prep_group = parser.add_argument_group("Preprocessing")
    prep_group.add_argument(
        "--min-peaks",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help="Minimum number of peaks per spectrum (default: 5).",
    )
    prep_group.add_argument(
        "--min-mz-range",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help="Minimum fragment m/z span per spectrum (default: 250.0).",
    )
    prep_group.add_argument(
        "--min-mz",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help="Minimum peak m/z inclusive (default: 101.0).",
    )
    prep_group.add_argument(
        "--max-mz",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help="Maximum peak m/z inclusive (default: 1500.0).",
    )
    prep_group.add_argument(
        "--remove-precursor-tol",
        type=_cli_optional_float,
        default=argparse.SUPPRESS,
        metavar="DA|none",
        help=(
            "Window (Da) around precursor m/z to remove peaks, or 'none' "
            "to disable (default: 1.5)."
        ),
    )
    prep_group.add_argument(
        "--min-intensity",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Minimum peak intensity relative to base peak in [0, 1) "
            "(default: 0.01)."
        ),
    )
    prep_group.add_argument(
        "--max-peaks-used",
        type=_cli_optional_int,
        default=argparse.SUPPRESS,
        metavar="N|none",
        help=(
            "Maximum most-intense peaks to retain, or 'none' for all "
            "(default: 50)."
        ),
    )
    prep_group.add_argument(
        "--scaling",
        choices=("off", "root", "log", "rank"),
        default=argparse.SUPPRESS,
        help="Peak intensity scaling method (default: off).",
    )

    # Clustering
    clust_group = parser.add_argument_group("Clustering")
    clust_group.add_argument(
        "--precursor-tol",
        nargs=2,
        metavar=("MASS", "MODE"),
        default=argparse.SUPPRESS,
        help=(
            'Precursor mass tolerance and unit ("ppm" or "Da", '
            "default: 20 ppm)."
        ),
    )
    clust_group.add_argument(
        "--rt-tol",
        type=_cli_optional_float,
        default=argparse.SUPPRESS,
        metavar="TOL|none",
        help="Retention time tolerance, or 'none' to disable (default: none).",
    )
    clust_group.add_argument(
        "--fragment-tol",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help="Fragment mass tolerance in m/z (default: 0.05).",
    )
    clust_group.add_argument(
        "--linkage",
        choices=("single", "complete", "average"),
        default=argparse.SUPPRESS,
        help="Hierarchical clustering linkage criterion (default: complete).",
    )
    clust_group.add_argument(
        "--distance-threshold",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help="Cosine distance threshold in (0, 1) (default: 0.1).",
    )
    clust_group.add_argument(
        "--min-matched-peaks",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help="Minimum matched fragment peaks (default: 0).",
    )
    clust_group.add_argument(
        "--batch-size",
        type=_cli_int,
        default=argparse.SUPPRESS,
        help="Maximum spectra per precursor m/z batch (default: 32768).",
    )
    clust_group.add_argument(
        "--precursor-charge-buckets",
        nargs="+",
        metavar="BUCKET",
        default=argparse.SUPPRESS,
        help=(
            "Charge buckets such as '[1]' '[2, 3]' unknown other, or "
            "'auto' to cluster each observed charge separately."
        ),
    )

    # Consensus
    cons_group = parser.add_argument_group("Consensus")
    cons_group.add_argument(
        "--consensus-method",
        dest="consensus_method",
        choices=("medoid", "average"),
        default=argparse.SUPPRESS,
        help="Representative spectrum method (default: medoid).",
    )
    cons_group.add_argument(
        "--outlier-cutoff-lower",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Lower sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average'; default: 1.5)."
        ),
    )
    cons_group.add_argument(
        "--outlier-cutoff-upper",
        type=_cli_float,
        default=argparse.SUPPRESS,
        help=(
            "Upper sigma cutoff for consensus averaging (only valid when "
            "consensus.method = 'average'; default: 1.5)."
        ),
    )
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
    """Build a sparse nested config override dict from parsed CLI args."""
    io_ov: dict[str, Any] = {}
    prep_ov: dict[str, Any] = {}
    clust_ov: dict[str, Any] = {}
    cons_ov: dict[str, Any] = {}

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

    for key in (
        "input_filenames",
        "output_filename",
        "work_dir",
        "overwrite",
        "export_representatives",
    ):
        if key in ns:
            io_ov[key] = ns[key]

    for key in (
        "min_peaks",
        "min_mz_range",
        "min_mz",
        "max_mz",
        "remove_precursor_tol",
        "min_intensity",
        "max_peaks_used",
        "scaling",
    ):
        if key in ns:
            prep_ov[key] = ns[key]

    if "precursor_tol" in ns:
        raw_mass, raw_mode = ns["precursor_tol"]
        try:
            mass_val = float(raw_mass)
        except ValueError as exc:
            raise ConfigError(
                f"--precursor-tol mass must be a number, got {raw_mass!r}"
            ) from exc
        clust_ov["precursor_tol"] = {"mass": mass_val, "mode": raw_mode}

    for key in (
        "rt_tol",
        "fragment_tol",
        "linkage",
        "distance_threshold",
        "min_matched_peaks",
        "batch_size",
    ):
        if key in ns:
            clust_ov[key] = ns[key]

    if "precursor_charge_buckets" in ns:
        clust_ov["precursor_charge_buckets"] = _parse_cli_charge_buckets(
            ns["precursor_charge_buckets"]
        )

    if "consensus_method" in ns:
        cons_ov["method"] = ns["consensus_method"]
    for key in ("outlier_cutoff_lower", "outlier_cutoff_upper"):
        if key in ns:
            cons_ov[key] = ns[key]

    overrides: dict[str, Any] = {}
    if io_ov:
        overrides["io"] = io_ov
    if prep_ov:
        overrides["preprocessing"] = prep_ov
    if clust_ov:
        overrides["clustering"] = clust_ov
    if cons_ov:
        overrides["consensus"] = cons_ov
    return overrides


def parse_cli(args: Sequence[str] | None = None) -> CliResult:
    """
    Parse CLI arguments and resolve layered TOML + CLI configuration.

    Parameters
    ----------
    args : Sequence[str] | None, default=None
        Optional CLI argument list. When ``None``, reads ``sys.argv[1:]``.

    Returns
    -------
    CliResult
        Action descriptor and validated configuration or settings object.
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
        return CliResult(action="dump_schema")

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
            return CliResult(action="print_config", settings=settings)
        config = load_config(config_files=config_files, overrides=overrides)
        return CliResult(action="run", config=config, save_config=save_config)
    except ConfigError as exc:
        parser.error(str(exc))
    except ValidationError as exc:
        parser.error(format_validation_error(exc))


def write_saved_config(config: FalconConfig, destination: Path) -> None:
    """
    Write a resolved ``FalconConfig`` as TOML to ``destination``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration to serialize.
    destination : Path
        Target TOML file path (must not be an existing directory).

    Raises
    ------
    ConfigError
        If ``destination`` is a directory or cannot be written.
    """
    if destination.exists() and destination.is_dir():
        raise ConfigError(
            f"--save-config path is a directory, expected a file path: "
            f"{destination}"
        )
    try:
        destination.write_text(config.to_toml(), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Failed to write configuration to {destination}: {exc}"
        ) from exc
