"""Command-line and INI file configuration parsing and validation."""

import argparse
import textwrap
from collections.abc import Sequence
from typing import Any

import configargparse

from falcon import __version__


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


class Config:
    """
    Command-line and INI file configuration container.

    Settings can be supplied via a ``config.ini`` file (by default in
    the current directory), an explicit ``-c``/``--config`` file, or
    command-line arguments. CLI values override file values, which
    override defaults.
    """

    def __init__(self) -> None:
        """Initialize the argument parser and default settings."""
        self._parser = configargparse.ArgParser(
            description=(
                "falcon: Large-scale tandem mass spectrum clustering\n"
                "===================================================\n\n"
                f"falcon version {__version__}\n\n"
                "Official code website: "
                "https://github.com/bittremieuxlab/falcon\n\n"
            ),
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=NewlineTextHelpFormatter,
        )

        # IO
        self._parser.add_argument(
            "input_filenames",
            nargs="+",
            help="Input peak files (supported formats: .mzML, .mzXML, .MGF).",
        )
        self._parser.add_argument("output_filename", help="Output file name.")
        self._parser.add_argument(
            "--work_dir",
            default=None,
            help="Working directory (default: temporary directory).",
        )
        self._parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing results (default: don't overwrite).",
        )
        self._parser.add_argument(
            "--export_representatives",
            action="store_true",
            help="Export cluster representatives to an MGF file "
            "(default: no export).",
        )

        # CLUSTERING
        self._parser.add_argument(
            "--precursor_tol",
            nargs=2,
            default=[20, "ppm"],
            help="Precursor tolerance mass and mode (default: 20 ppm). "
            'Mode should be either "ppm" or "Da".',
        )
        self._parser.add_argument(
            "--rt_tol",
            type=float,
            default=None,
            help="Retention time tolerance (default: no retention time "
            "filtering).",
        )
        self._parser.add_argument(
            "--fragment_tol",
            type=float,
            default=0.05,
            help="Fragment mass tolerance in m/z (default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--linkage",
            type=str,
            default="complete",
            choices=["single", "complete", "average"],
            help="Linkage criterion for hierarchical clustering "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--distance_threshold",
            type=float,
            default=0.1,
            help=(
                "The distance threshold parameter (cosine distance) for "
                "clustering (default: %(default)s). Relevant cosine distance "
                "thresholds are typically between 0.05 and 0.30."
            ),
        )
        self._parser.add_argument(
            "--min_matched_peaks",
            type=int,
            default=0,
            help=(
                "Minimum number of matched peaks to consider the spectra "
                "similar (default: %(default)s). Typically 6 for "
                "metabolomics data."
            ),
        )
        self._parser.add_argument(
            "--consensus_method",
            type=str,
            default="medoid",
            choices=["medoid", "average"],
            help="Method to use for consensus spectrum computation "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--outlier_cutoff_lower",
            type=float,
            default=1.5,
            help=(
                "Number of standard deviations below the median for outlier "
                "rejection (default: %(default)s). Only used when "
                "consensus_method='average'."
            ),
        )
        self._parser.add_argument(
            "--outlier_cutoff_upper",
            type=float,
            default=1.5,
            help=(
                "Number of standard deviations above the median for outlier "
                "rejection (default: %(default)s). Only used when "
                "consensus_method='average'."
            ),
        )
        self._parser.add_argument(
            "--batch_size",
            type=int,
            default=2**15,
            help="Batch size for clustering (default: %(default)s).",
        )
        self._parser.add_argument(
            "--precursor_charge_buckets",
            nargs="+",
            default=None,
            metavar="BUCKET",
            help=(
                "Charge buckets for precursor charges. Clustering will be "
                "performed separately within each bucket. If not specified, "
                "every distinct precursor charge is caught and "
                "clustered separately, including spectra with a missing "
                "charge. To group charges instead, pass one bucket per "
                "argument, e.g. '[1]' '[2]' '[3]' '[4]' '[unknown]' 'other' "
                "(each bucket is a list of charges; 'unknown' matches missing "
                "charges and 'other' catches any charge not in a named "
                "bucket)."
            ),
        )

        # PREPROCESSING
        self._parser.add_argument(
            "--min_peaks",
            default=5,
            type=int,
            help="Discard spectra with fewer than this number of peaks "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--min_mz_range",
            default=250.0,
            type=float,
            help="Discard spectra with a smaller mass range "
            "(default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--min_mz",
            default=101.0,
            type=float,
            help="Minimum peak m/z value (inclusive, "
            "default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--max_mz",
            default=1500.0,
            type=float,
            help="Maximum peak m/z value (inclusive, "
            "default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--remove_precursor_tol",
            default=1.5,
            type=float,
            help="Window around the precursor mass to remove peaks "
            "(default: %(default)s m/z).",
        )
        self._parser.add_argument(
            "--min_intensity",
            default=0.01,
            type=float,
            help="Remove peaks with a lower intensity relative to the base "
            "intensity (default: %(default)s).",
        )
        self._parser.add_argument(
            "--max_peaks_used",
            default=50,
            type=int,
            help="Only use the specified most intense peaks in the spectra "
            "(default: %(default)s).",
        )
        self._parser.add_argument(
            "--scaling",
            default="off",
            type=str,
            choices=["off", "root", "log", "rank"],
            help="Peak scaling method used to reduce the influence of very "
            "intense peaks (default: %(default)s).",
        )
        # Filled in 'parse', contains the specified settings.
        self._namespace: dict[str, Any] | None = None

    def parse(self, args_str: str | list[str] | None = None) -> None:
        """
        Parse and validate configuration settings.

        Parameters
        ----------
        args_str : str | list[str] | None, default=None
            CLI arguments to parse. When ``None``, arguments are read
            from ``sys.argv``; omitted options fall back to the config
            file and then built-in defaults.
        """
        self._namespace = vars(self._parser.parse_args(args_str))

        self._namespace["precursor_tol"][0] = float(
            self._namespace["precursor_tol"][0]
        )

        # Leave unset buckets as None so each observed charge (including
        # missing charges) is clustered in its own bucket.
        if self._namespace["precursor_charge_buckets"] is not None:
            self._namespace["precursor_charge_buckets"] = (
                self.parse_and_validate_charge_buckets(
                    self._namespace["precursor_charge_buckets"]
                )
            )

        self._validate()

    def _validate(self) -> None:
        """
        Validate parsed settings that argparse cannot check directly.

        Invalid values are reported via ``self._parser.error`` so CLI
        errors fail fast at startup with standard usage output.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized")
        ns = self._namespace
        mode = ns["precursor_tol"][1]
        if mode not in ("ppm", "Da"):
            self._parser.error(
                f'--precursor_tol mode must be "ppm" or "Da", got "{mode}"'
            )
        if ns["distance_threshold"] <= 0:
            self._parser.error(
                "--distance_threshold must be positive, got "
                f"{ns['distance_threshold']}"
            )
        if ns["batch_size"] < 1:
            self._parser.error(
                "--batch_size must be a positive integer, got "
                f"{ns['batch_size']}"
            )

    def parse_and_validate_charge_buckets(
        self, bucket_args: Sequence[str]
    ) -> list[set[int | str] | str]:
        """
        Parse and validate precursor charge bucket specifications.

        Parameters
        ----------
        bucket_args : Sequence[str]
            Raw charge bucket specifications from the CLI or config
            file (e.g. ``"[1]"``, ``"[2, 3]"``, ``"[unknown]"``,
            ``"other"``).

        Returns
        -------
        list[set[int | str] | str]
            Parsed buckets, each either a set of integer charges and/or
            ``"unknown"``, or the sentinel string ``"other"``.
        """
        buckets: list[set[int | str] | str] = []
        seen_values: set[int | str] = set()

        for raw in bucket_args:
            raw = raw.strip()

            if raw == "other":
                buckets.append("other")
                continue

            values = self._parse_charge_bucket(raw)

            # Reject charges assigned to more than one bucket.
            overlap = values & seen_values
            if overlap:
                sorted_overlap = sorted(
                    overlap, key=lambda x: (isinstance(x, str), x)
                )
                self._parser.error(
                    f"Charge value(s) {sorted_overlap} "
                    f"appear in more than one bucket"
                )

            seen_values |= values
            buckets.append(values)

        return buckets

    def _parse_charge_bucket(self, raw: str) -> set[int | str]:
        """
        Parse a single charge bucket specification into a set of values.

        Accepts an optionally bracketed comma-separated list of integer
        charges and/or the ``unknown`` sentinel, e.g. ``[1]``,
        ``[2, 3]``, ``[unknown]``, or ``[1, unknown]``.

        Parameters
        ----------
        raw : str
            Raw single-bucket specification.

        Returns
        -------
        set[int | str]
            Parsed integer charges and/or ``"unknown"`` in the bucket.
        """
        inner = raw
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1]
        values: set[int | str] = set()
        for part in inner.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "unknown":
                values.add("unknown")
            else:
                try:
                    values.add(int(part))
                except ValueError:
                    self._parser.error(
                        f"Invalid charge bucket value '{part}' in '{raw}' "
                        f"(expected integers and/or 'unknown')"
                    )
        if not values:
            self._parser.error(f"Empty charge bucket: '{raw}'")
        return values

    def __getattr__(self, option: str) -> Any:
        """Return a parsed configuration option by attribute name."""
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized")
        return self._namespace[option]

    def __setattr__(self, option: str, value: Any) -> None:
        """Set private attributes normally and update parsed options."""
        if option.startswith("_") or self._namespace is None:
            super().__setattr__(option, value)
        else:
            self._namespace[option] = value

    def __getitem__(self, item: str) -> Any:
        """Return a parsed configuration option by mapping key."""
        return self.__getattr__(item)


config = Config()
