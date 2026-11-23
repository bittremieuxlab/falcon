"""Output preparation, global label assembly, and CSV/MGF export."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any, cast

import natsort
import pandas as pd

from . import ms_io
from .config.serialize import format_metadata_lines
from .dataset import METADATA_COLUMNS, read_canonical_metadata
from .types import offset_cluster_result

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import lance

    from .config.models import FalconConfig, IOConfig
    from .types import ClusterResult, ConsensusTuple

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [*METADATA_COLUMNS, "cluster"]

__all__ = ["export_results", "prepare_outputs", "write_cluster_csv"]


def _check_existing_output(
    path: Path, overwrite: bool, description: str
) -> bool:
    """Return True if ``path`` is clear, unlinking when overwriting."""
    if not path.is_file():
        return True
    if overwrite:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            description,
        )
        path.unlink()
        return True
    logger.error(
        "Output file %s (%s) already exists, aborting...", path, description
    )
    return False


def prepare_outputs(io_cfg: IOConfig) -> bool:
    """
    Ensure ``.csv`` and ``.mgf`` outputs are ready before running.

    Evaluates both files without short-circuiting so ``overwrite=True``
    clears stale ``.mgf`` files even alongside an existing ``.csv``.

    Parameters
    ----------
    io_cfg : IOConfig
        Run I/O settings (``output_filename`` prefix and ``overwrite``).

    Returns
    -------
    bool
        ``True`` if it is safe to proceed (no blocking output files;
        existing outputs unlinked when ``overwrite=True``), or ``False``
        if an existing output blocks the run.
    """
    csv_ready = _check_existing_output(
        io_cfg.csv_path, io_cfg.overwrite, "cluster assignments"
    )
    mgf_ready = _check_existing_output(
        io_cfg.mgf_path, io_cfg.overwrite, "cluster representatives"
    )
    return csv_ready and mgf_ready


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
    with config.io.csv_path.open("w", encoding="utf-8") as f_out:
        for line in format_metadata_lines(config, work_dir=work_dir):
            f_out.write(f"# {line}\n")
        f_out.write("#\n")
        clusters.to_csv(f_out, index=False, chunksize=1_000_000)


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
    clusters_all: list[pd.DataFrame] = []
    current_label = 0
    representatives: list[ConsensusTuple] = []
    export_reps = config.io.export_representatives

    for dataset, result in zip(bucket_datasets, cluster_results, strict=True):
        shifted, current_label = offset_cluster_result(
            result, current_label, include_representatives=export_reps
        )
        if len(shifted.labels) == 0:
            continue
        if export_reps:
            representatives.extend(shifted.representatives)

        # Re-read metadata in the canonical order used for clustering.
        metadata = read_canonical_metadata(dataset, METADATA_COLUMNS)
        metadata["cluster"] = shifted.labels
        clusters_all.append(metadata)

    if clusters_all:
        clusters_df = pd.concat(clusters_all, ignore_index=True).sort_values(
            ["identifier"], key=cast("Any", natsort.natsort_keygen())
        )
    else:
        clusters_df = pd.DataFrame(columns=_OUTPUT_COLUMNS)

    csv_path = config.io.csv_path
    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        csv_path,
    )

    if not export_reps:
        write_cluster_csv(clusters_df, config, work_dir)
        return

    mgf_path = config.io.mgf_path
    logger.info(
        "Export %d cluster representative spectra to output file %s",
        len(representatives),
        mgf_path,
    )

    if clusters_df.empty:
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
