"""Programmatic pipeline orchestration for Falcon MS/MS clustering."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import lance

from .cluster import generate_clusters_multi
from .cluster.spectrum import build_process_spectrum
from .config.models import (
    AverageConsensusConfig,
    ConsensusConfig,
    FalconConfig,
)
from .export import ClusterResult, export_results, validate_outputs
from .staging import (
    bucket_dataset_path,
    format_bucket_label,
    load_or_stage_buckets,
    spectra_dir,
)
from .types import ChargeBucket

logger = logging.getLogger("falcon")


def _log_config(config: FalconConfig, work_dir: Path) -> None:
    """Log the Falcon version and active configuration settings."""
    metadata_lines = config.to_metadata_lines(work_dir=work_dir)
    logger.info("%s", metadata_lines[0])
    for line in metadata_lines[1:]:
        logger.debug("%s", line)


def _setup_work_dir(work_dir: Path | None) -> tuple[Path, bool]:
    """
    Create the working directory and its ``spectra`` subdirectory.

    Parameters
    ----------
    work_dir : Path | None
        Explicit working directory from ``config.io.work_dir``, or
        ``None`` to allocate a temporary directory.

    Returns
    -------
    tuple[Path, bool]
        Resolved working directory path and a boolean flag indicating
        whether it is a temporary directory to remove on exit.
    """
    if work_dir is None:
        resolved = Path(tempfile.mkdtemp())
        rm_work_dir = True
    else:
        resolved = work_dir
        rm_work_dir = False
        if resolved.is_dir():
            logger.warning(
                "Working directory %s already exists, previous "
                "results might get overwritten",
                resolved,
            )
        resolved.mkdir(parents=True, exist_ok=True)
    spectra_dir(resolved).mkdir(exist_ok=True)
    return resolved, rm_work_dir


def _warn_aggressive_outlier_cutoffs(consensus: ConsensusConfig) -> None:
    """Warn when both averaging outlier cutoffs are set below 1."""
    if (
        isinstance(consensus, AverageConsensusConfig)
        and consensus.outlier_cutoff_lower < 1.0
        and consensus.outlier_cutoff_upper < 1.0
    ):
        logger.warning(
            "Setting both outlier_cutoff_lower and outlier_cutoff_upper "
            "to values less than 1 can lead to unexpected results. It "
            "is advised to set either outlier_cutoff_lower or "
            "outlier_cutoff_upper to a value >= 1."
        )


def _run_clustering(
    config: FalconConfig,
    work_dir: Path,
    charge_buckets: Sequence[ChargeBucket],
) -> tuple[list[lance.LanceDataset], list[ClusterResult]]:
    """
    Cluster every non-empty charge bucket in a shared worker pool.

    Skips representative spectrum computation when
    ``config.io.export_representatives`` is ``False``.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory containing staged Lance datasets.
    charge_buckets : Sequence[ChargeBucket]
        Staged charge bucket keys to cluster.

    Returns
    -------
    tuple[list[lance.LanceDataset], list[ClusterResult]]
        Non-empty per-bucket Lance datasets and their positionally
        aligned ``(labels, representative_spectra)`` clustering results.
    """
    bucket_datasets: list[lance.LanceDataset] = []
    bucket_labels: list[str] = []
    for bucket in charge_buckets:
        dataset = lance.dataset(str(bucket_dataset_path(work_dir, bucket)))
        if dataset.count_rows() == 0:
            continue
        bucket_datasets.append(dataset)
        bucket_labels.append(format_bucket_label(bucket))

    cluster_results = generate_clusters_multi(
        bucket_datasets,
        config.clustering,
        config.consensus,
        min_mz=config.preprocessing.min_mz,
        max_mz=config.preprocessing.max_mz,
        bucket_labels=bucket_labels,
        compute_representatives=config.io.export_representatives,
    )
    return bucket_datasets, cluster_results


def run(config: FalconConfig) -> int:
    """
    Execute the Falcon MS/MS spectrum clustering pipeline.

    Programmatic callers should pass a validated ``FalconConfig`` here.
    Unlike :func:`falcon.cli.main`, this function does not configure or
    shut down global logging handlers.

    Parameters
    ----------
    config : FalconConfig
        Immutable, validated run configuration.

    Returns
    -------
    int
        Exit status: ``0`` on success, or ``1`` if output files already
        exist and ``config.io.overwrite`` is ``False``.
    """
    work_dir, rm_work_dir = _setup_work_dir(config.io.work_dir)
    try:
        _log_config(config, work_dir)
        if validate_outputs(config.io):
            return 1
        _warn_aggressive_outlier_cutoffs(config.consensus)

        process_spectrum = build_process_spectrum(config)
        charge_buckets = load_or_stage_buckets(
            config, work_dir, process_spectrum
        )
        bucket_datasets, cluster_results = _run_clustering(
            config, work_dir, charge_buckets
        )
        export_results(config, work_dir, bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir and work_dir.is_dir():
            shutil.rmtree(work_dir)
