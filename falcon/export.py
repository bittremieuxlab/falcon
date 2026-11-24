"""Output preparation, global label assembly, and CSV/MGF export."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING

import natsort
import pandas as pd

from . import ms_io
from .config import format_metadata_lines
from .dataset import METADATA_COLUMNS, read_canonical_metadata
from .types import offset_cluster_result
from .utils import atomic_write_text, expand_input_globs, find_path_alias

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import lance

    from .config import FalconConfig, IOConfig
    from .types import ClusterResult, ConsensusTuple

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [*METADATA_COLUMNS, "cluster"]

__all__ = [
    "export_results",
    "prepare_outputs",
    "remove_existing_outputs",
    "write_cluster_csv",
]


def _check_output_aliasing(io_cfg: IOConfig) -> None:
    """Raise ValueError if CSV or MGF aliases input or ``work_dir``."""
    input_candidates = expand_input_globs(
        io_cfg.input_filenames, keep_unmatched_literals=True
    )
    checks: list[tuple[str, Sequence[Path]]] = [
        ("input file", input_candidates)
    ]
    if io_cfg.work_dir is not None:
        checks.append(("working directory", (io_cfg.work_dir,)))
    for out_path in (io_cfg.csv_path, io_cfg.mgf_path):
        for kind, candidates in checks:
            alias = find_path_alias(out_path, candidates)
            if alias is not None:
                raise ValueError(
                    f"Output file {out_path} aliases {kind} "
                    f"{alias}; choose a distinct output prefix"
                )


def remove_existing_outputs(io_cfg: IOConfig) -> None:
    """Unlink any existing CSV or MGF output files or symlinks."""
    for out_path in (io_cfg.csv_path, io_cfg.mgf_path):
        if out_path.is_file() or out_path.is_symlink():
            out_path.unlink()


def prepare_outputs(io_cfg: IOConfig, *, unlink: bool = True) -> bool:
    """
    Ensure ``.csv`` and ``.mgf`` outputs are ready before running.

    Checks that neither output path aliases an input file, verifies the
    output parent directory exists, and validates both output targets
    for directory collisions before performing any unlinking. Evaluates
    both suffixes without short-circuiting so ``overwrite=True`` clears
    stale ``.mgf`` files even alongside an existing ``.csv``.

    Parameters
    ----------
    io_cfg : IOConfig
        Run I/O settings (``output_filename`` prefix and ``overwrite``).
    unlink : bool, default=True
        When ``True`` and ``io_cfg.overwrite`` is enabled, unlink
        existing output files immediately; set ``False`` to validate
        non-destructively before staging.

    Returns
    -------
    bool
        ``True`` if it is safe to proceed (no blocking output files),
        or ``False`` if an existing output blocks the run.

    Raises
    ------
    ValueError
        If ``csv_path`` or ``mgf_path`` aliases an input file or
        ``work_dir``.
    FileNotFoundError
        If the parent directory for output files does not exist.
    IsADirectoryError
        If ``csv_path`` or ``mgf_path`` is an existing directory.
    """
    _check_output_aliasing(io_cfg)
    parent_dir = io_cfg.csv_path.parent
    if not parent_dir.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {parent_dir}"
        )

    targets = (
        (io_cfg.csv_path, "cluster assignments"),
        (io_cfg.mgf_path, "cluster representatives"),
    )
    for path, description in targets:
        if path.is_dir():
            raise IsADirectoryError(
                f"Output path {path} ({description}) is a directory"
            )

    existing = [
        (path, description)
        for path, description in targets
        if path.is_file() or path.is_symlink()
    ]
    if not existing:
        return True

    if not io_cfg.overwrite:
        for path, description in existing:
            logger.error(
                "Output file %s (%s) already exists, aborting...",
                path,
                description,
            )
        return False

    for path, description in existing:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            description,
        )
    if unlink:
        remove_existing_outputs(io_cfg)
    return True


def write_cluster_csv(
    clusters: pd.DataFrame,
    config: FalconConfig,
    work_dir: Path,
) -> None:
    """
    Write configuration metadata header comments and cluster CSV rows.

    Parameters
    ----------
    clusters : pd.DataFrame
        Cluster assignment table with columns ``identifier``,
        ``precursor_charge``, ``precursor_mz``, ``retention_time``, and
        ``cluster``.
    config : FalconConfig
        Validated run configuration written as ``#`` comment headers.
    work_dir : Path
        Effective working directory recorded in the metadata header.
    """
    with atomic_write_text(config.io.csv_path, prefix=".csv_") as f_out:
        for line in format_metadata_lines(config, work_dir=work_dir):
            safe_line = line.replace("\r", "\\r").replace("\n", "\\n")
            f_out.write(f"# {safe_line}\n")
        f_out.write("#\n")
        clusters.to_csv(f_out, index=False, chunksize=1_000_000)


def _assemble_cluster_results(
    bucket_datasets: Sequence[lance.LanceDataset],
    cluster_results: Sequence[ClusterResult],
    *,
    export_representatives: bool,
) -> tuple[pd.DataFrame, list[ConsensusTuple]]:
    """Offset bucket labels and assemble naturally sorted metadata."""
    clusters_all: list[pd.DataFrame] = []
    current_label = 0
    representatives: list[ConsensusTuple] = []

    for dataset, result in zip(bucket_datasets, cluster_results, strict=True):
        n_rows = dataset.count_rows()
        if n_rows != len(result.labels):
            raise ValueError(
                f"Bucket dataset row count ({n_rows}) does not match "
                f"cluster label count ({len(result.labels)})"
            )
        shifted, current_label = offset_cluster_result(
            result,
            current_label,
            include_representatives=export_representatives,
        )
        if len(shifted.labels) == 0:
            continue
        if export_representatives:
            representatives.extend(shifted.representatives)

        # Re-read metadata in the canonical order used for clustering.
        metadata = read_canonical_metadata(dataset, METADATA_COLUMNS)
        metadata["cluster"] = shifted.labels
        clusters_all.append(metadata)

    if len(clusters_all) == 1:
        clusters_df = clusters_all[0]
    elif clusters_all:
        clusters_df = pd.concat(clusters_all, ignore_index=True)
    else:
        clusters_df = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    clusters_all.clear()

    if not clusters_df.empty:
        clusters_df["precursor_charge"] = clusters_df[
            "precursor_charge"
        ].astype("Int8")
        clusters_df = clusters_df.iloc[
            natsort.index_natsorted(clusters_df["identifier"])
        ]
    return clusters_df, representatives


def _write_outputs(
    clusters_df: pd.DataFrame,
    representatives: Sequence[ConsensusTuple],
    config: FalconConfig,
    work_dir: Path,
) -> None:
    """Clear stale outputs if requested and write CSV/MGF files."""
    if config.io.overwrite:
        remove_existing_outputs(config.io)

    csv_path = config.io.csv_path
    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        csv_path,
    )

    if not config.io.export_representatives:
        write_cluster_csv(clusters_df, config, work_dir)
        return

    mgf_path = config.io.mgf_path
    logger.info(
        "Export %d cluster representative spectra to output file %s",
        len(representatives),
        mgf_path,
    )

    if clusters_df.empty or not representatives:
        write_cluster_csv(clusters_df, config, work_dir)
        ms_io.write_spectra(mgf_path, representatives)
        return

    # Overlap CSV and MGF writes on background IO threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        csv_future = executor.submit(
            write_cluster_csv, clusters_df, config, work_dir
        )
        mgf_future = executor.submit(
            ms_io.write_spectra, mgf_path, representatives
        )
        csv_future.result()
        mgf_future.result()


def export_results(
    config: FalconConfig,
    work_dir: Path,
    bucket_datasets: Sequence[lance.LanceDataset],
    cluster_results: Sequence[ClusterResult],
) -> None:
    """
    Assemble global cluster assignments and write CSV/MGF outputs.

    Cluster labels from different charge buckets are offset without
    mutating caller arrays, per-bucket metadata tables are concatenated
    and naturally sorted by identifier, and ``.csv`` (plus optional
    ``.mgf``) outputs are written.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory recorded in output metadata comments.
    bucket_datasets : Sequence[lance.LanceDataset]
        Non-empty per-bucket Lance datasets.
    cluster_results : Sequence[ClusterResult]
        Positionally aligned ``ClusterResult`` records for
        ``bucket_datasets``.
    """
    clusters_df, representatives = _assemble_cluster_results(
        bucket_datasets,
        cluster_results,
        export_representatives=config.io.export_representatives,
    )
    _write_outputs(clusters_df, representatives, config, work_dir)
