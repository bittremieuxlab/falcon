"""Programmatic pipeline orchestration for Falcon MS/MS clustering."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import lance

from .cluster import generate_clusters_multi
from .config.models import AverageConsensusConfig
from .config.serialize import format_metadata_lines
from .dataset import bucket_dataset_path, format_bucket_label, spectra_dir
from .export import export_results, prepare_outputs
from .preprocessing import build_process_spectrum
from .staging import load_or_stage_buckets

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config.models import ConsensusConfig, FalconConfig
    from .types import ChargeBucket, ClusterResult

logger = logging.getLogger(__name__)

__all__ = ["run"]


def _log_config(config: FalconConfig, work_dir: Path) -> None:
    """Log the Falcon version and active configuration settings."""
    metadata_lines = format_metadata_lines(config, work_dir=work_dir)
    logger.info("%s", metadata_lines[0])
    for line in metadata_lines[1:]:
        logger.debug("%s", line)


def _setup_work_dir(
    work_dir: Path | None, *, overwrite: bool = False
) -> tuple[Path, bool]:
    """
    Create the working directory and its ``spectra`` subdirectory.

    Parameters
    ----------
    work_dir : Path | None
        Explicit working directory from ``config.io.work_dir``, or
        ``None`` to allocate a temporary directory.
    overwrite : bool, default=False
        Whether existing staged files in ``work_dir`` will be
        overwritten rather than validated and reused.

    Returns
    -------
    tuple[Path, bool]
        Resolved working directory path and a boolean flag indicating
        whether it is a temporary directory to remove on exit.
    """
    if work_dir is None:
        resolved = Path(tempfile.mkdtemp())
        try:
            spectra_dir(resolved).mkdir(exist_ok=True)
        except BaseException:
            shutil.rmtree(resolved, ignore_errors=True)
            raise
        return resolved, True

    # Absolutize explicit work_dir so Lance dataset URIs and memmap
    # paths stay valid in warm loky workers across os.chdir().
    resolved = Path(os.path.abspath(work_dir))
    if resolved.is_dir():
        if overwrite:
            logger.warning(
                "Working directory %s already exists; cached staging "
                "files will be overwritten",
                resolved,
            )
        else:
            logger.warning(
                "Working directory %s already exists; validated staging "
                "cache will be reused if available",
                resolved,
            )
    resolved.mkdir(parents=True, exist_ok=True)
    spectra_dir(resolved).mkdir(exist_ok=True)
    return resolved, False


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
        aligned ``ClusterResult`` clustering results.
    """
    bucket_datasets: list[lance.LanceDataset] = []
    bucket_labels: list[str] = []
    for bucket in charge_buckets:
        dataset = lance.dataset(str(bucket_dataset_path(work_dir, bucket)))
        if dataset.count_rows() == 0:
            continue
        bucket_datasets.append(dataset)
        bucket_labels.append(format_bucket_label(bucket))

    if not bucket_datasets:
        return [], []

    cluster_results = generate_clusters_multi(
        bucket_datasets,
        config.clustering,
        config.consensus,
        min_mz=config.preprocessing.min_mz,
        max_mz=config.preprocessing.max_mz,
        bucket_labels=bucket_labels,
        compute_representatives=config.io.export_representatives,
        temp_dir=work_dir,
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
    if not prepare_outputs(config.io, unlink=False):
        return 1
    work_dir, rm_work_dir = _setup_work_dir(
        config.io.work_dir, overwrite=config.io.overwrite
    )
    try:
        _log_config(config, work_dir)
        _warn_aggressive_outlier_cutoffs(config.consensus)

        process_spectrum = build_process_spectrum(config)
        charge_buckets = load_or_stage_buckets(
            config, work_dir, process_spectrum
        )
        bucket_datasets, cluster_results = _run_clustering(
            config, work_dir, charge_buckets
        )
        if config.io.overwrite:
            for out_path in (config.io.csv_path, config.io.mgf_path):
                if out_path.is_file() or out_path.is_symlink():
                    out_path.unlink()
        export_results(config, work_dir, bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
