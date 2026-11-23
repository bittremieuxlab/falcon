"""Output validation, global label assembly, and CSV/MGF export."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import lance
import natsort
import numpy as np
import pandas as pd

from . import ms_io
from .config.models import FalconConfig, IOConfig
from .staging import CANONICAL_SORT_COLUMNS, METADATA_COLUMNS
from .types import ConsensusTuple

logger = logging.getLogger("falcon")

ClusterResult = tuple[np.ndarray, list[ConsensusTuple]]

_OUTPUT_COLUMNS = [*METADATA_COLUMNS, "cluster"]


def _check_existing_output(
    path: Path, overwrite: bool, description: str
) -> bool:
    """Check an output path and unlink it when ``overwrite`` is set."""
    if not path.is_file():
        return False
    if overwrite:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            description,
        )
        path.unlink()
        return False
    logger.error(
        "Output file %s (%s) already exists, aborting...", path, description
    )
    return True


def validate_outputs(io_cfg: IOConfig) -> bool:
    """
    Check both ``.csv`` and ``.mgf`` outputs for pre-existing files.

    Evaluates both files without short-circuiting so ``overwrite=True``
    clears stale ``.mgf`` files even alongside an existing ``.csv``.

    Parameters
    ----------
    io_cfg : IOConfig
        Run I/O settings (``output_filename`` prefix and ``overwrite``).

    Returns
    -------
    bool
        ``True`` if an existing output blocks the run without
        ``overwrite``.
    """
    csv_abort = _check_existing_output(
        io_cfg.csv_path, io_cfg.overwrite, "cluster assignments"
    )
    mgf_abort = _check_existing_output(
        io_cfg.mgf_path, io_cfg.overwrite, "cluster representatives"
    )
    return csv_abort or mgf_abort


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
        for line in config.to_metadata_lines(work_dir=work_dir):
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
        Positionally aligned ``(labels, representative_spectra)``
        results for ``bucket_datasets``.
    """
    clusters_all: list[pd.DataFrame] = []
    current_label = 0
    representatives: list[ConsensusTuple] = []
    export_reps = config.io.export_representatives

    for dataset, (clusters, rep_spectra) in zip(
        bucket_datasets, cluster_results, strict=True
    ):
        if len(clusters) == 0:
            continue
        # Offset labels without mutating the caller's array.
        offset_clusters = np.asarray(clusters, dtype=np.int32) + np.int32(
            current_label
        )
        if export_reps:
            representatives.extend(
                s._replace(
                    cluster_id=np.int32(int(s.cluster_id) + current_label)
                )
                for s in rep_spectra
            )
        current_label = int(np.amax(offset_clusters)) + 1

        # Re-read metadata in the canonical order used for clustering.
        metadata = (
            dataset.to_table(columns=list(METADATA_COLUMNS))
            .to_pandas()
            .sort_values(list(CANONICAL_SORT_COLUMNS))
        )
        metadata["cluster"] = offset_clusters
        clusters_all.append(metadata)

    if clusters_all:
        clusters_df = pd.concat(clusters_all, ignore_index=True).sort_values(
            ["identifier"], key=cast(Any, natsort.natsort_keygen())
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
    errors: list[BaseException] = []

    def _run_writer(target: Callable[..., Any], *args: Any) -> None:
        try:
            target(*args)
        except BaseException as exc:
            errors.append(exc)

    # Overlap CSV and MGF writes on background IO threads.
    csv_thread = threading.Thread(
        target=_run_writer,
        args=(write_cluster_csv, clusters_df, config, work_dir),
        daemon=True,
    )
    mgf_thread = threading.Thread(
        target=_run_writer,
        args=(ms_io.write_spectra, mgf_path, representatives),
        daemon=True,
    )
    csv_thread.start()
    mgf_thread.start()
    mgf_thread.join()
    csv_thread.join()
    if errors:
        raise errors[0]
