"""Programmatic pipeline orchestration and CSV/MGF result export."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import lance
import natsort
import pandas as pd

from . import ms_io
from .cluster import generate_clusters_multi
from .config import AverageConsensusConfig, format_metadata_lines
from .preprocessing import build_process_spectrum
from .staging import (
    METADATA_COLUMNS,
    bucket_dataset_path,
    format_bucket_label,
    load_or_stage_buckets,
    read_canonical_metadata,
    spectra_dir,
)
from .types import offset_cluster_result
from .utils import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import FalconConfig, IOConfig
    from .types import ChargeBucket, ClusterResult, ConsensusTuple

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [*METADATA_COLUMNS, "cluster"]

__all__ = [
    "export_results",
    "prepare_outputs",
    "run",
    "write_cluster_csv",
]


def _remove_existing_outputs(io_cfg: IOConfig) -> None:
    for out_path in (io_cfg.csv_path, io_cfg.mgf_path):
        if out_path.is_file() or out_path.is_symlink():
            out_path.unlink()


def prepare_outputs(io_cfg: IOConfig) -> bool:
    """
    Validate output paths before running and report whether to proceed.

    Checks that output paths do not collide with input files or
    ``work_dir``, verifies the parent directory exists, and returns
    ``False`` when an output already exists and ``overwrite`` is off.
    """
    protected: list[tuple[str, Path]] = [
        ("input file", Path(p)) for p in io_cfg.input_filenames
    ]
    if io_cfg.work_dir is not None:
        protected.append(("working directory", io_cfg.work_dir))
    for out_path in (io_cfg.csv_path, io_cfg.mgf_path):
        resolved_out = out_path.resolve()
        for kind, candidate in protected:
            if resolved_out == candidate.resolve():
                raise ValueError(
                    f"Output file {out_path} aliases {kind} "
                    f"{candidate}; choose a distinct output prefix"
                )

    parent_dir = io_cfg.csv_path.parent
    if not parent_dir.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {parent_dir}"
        )

    targets = (
        (io_cfg.csv_path, "cluster assignments"),
        (io_cfg.mgf_path, "cluster representatives"),
    )
    for path, desc in targets:
        if path.is_dir():
            raise IsADirectoryError(
                f"Output path {path} ({desc}) is a directory"
            )

    existing = [
        (path, desc)
        for path, desc in targets
        if path.is_file() or path.is_symlink()
    ]
    if not existing:
        return True

    if not io_cfg.overwrite:
        for path, desc in existing:
            logger.error(
                "Output file %s (%s) already exists, aborting...",
                path,
                desc,
            )
        return False

    for path, desc in existing:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            desc,
        )
    return True


def write_cluster_csv(
    clusters: pd.DataFrame,
    config: FalconConfig,
    work_dir: Path,
) -> None:
    """Write configuration metadata comments and cluster CSV rows."""
    with atomic_write_text(config.io.csv_path, prefix=".csv_") as f_out:
        for line in format_metadata_lines(config, work_dir=work_dir):
            safe = line.replace("\r", "\\r").replace("\n", "\\n")
            f_out.write(f"# {safe}\n")
        f_out.write("#\n")
        clusters.to_csv(f_out, index=False, chunksize=1_000_000)


def export_results(
    config: FalconConfig,
    work_dir: Path,
    bucket_datasets: Sequence[lance.LanceDataset],
    cluster_results: Sequence[ClusterResult],
) -> None:
    """Assemble global cluster assignments and write CSV/MGF outputs."""
    clusters_all: list[pd.DataFrame] = []
    representatives: list[ConsensusTuple] = []
    current_label = 0
    export_reps = config.io.export_representatives

    for dataset, result in zip(bucket_datasets, cluster_results, strict=True):
        n_rows = dataset.count_rows()
        if n_rows != len(result.labels):
            raise ValueError(
                f"Bucket dataset row count ({n_rows}) does not match "
                f"cluster label count ({len(result.labels)})"
            )
        shifted, current_label = offset_cluster_result(
            result, current_label, include_representatives=export_reps
        )
        if len(shifted.labels) == 0:
            continue
        if export_reps:
            representatives.extend(shifted.representatives)
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

    if config.io.overwrite:
        _remove_existing_outputs(config.io)

    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        config.io.csv_path,
    )
    write_cluster_csv(clusters_df, config, work_dir)

    if export_reps:
        logger.info(
            "Export %d cluster representative spectra to output file %s",
            len(representatives),
            config.io.mgf_path,
        )
        ms_io.write_spectra(config.io.mgf_path, representatives)


def _setup_work_dir(
    work_dir: Path | None, *, overwrite: bool = False
) -> tuple[Path, bool]:
    if work_dir is None:
        resolved = Path(tempfile.mkdtemp())
        spectra_dir(resolved).mkdir(exist_ok=True)
        return resolved, True

    # Absolutize explicit work_dir so Lance dataset URIs and memmap
    # paths stay valid in warm loky workers across os.chdir().
    resolved = Path(os.path.abspath(work_dir))
    if resolved.is_dir():
        action = "overwritten" if overwrite else "reused if available"
        logger.warning(
            "Working directory %s already exists; cached staging files "
            "will be %s",
            resolved,
            action,
        )
    resolved.mkdir(parents=True, exist_ok=True)
    spectra_dir(resolved).mkdir(exist_ok=True)
    return resolved, False


def _run_clustering(
    config: FalconConfig,
    work_dir: Path,
    charge_buckets: Sequence[ChargeBucket],
) -> tuple[list[lance.LanceDataset], list[ClusterResult]]:
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

    Returns ``0`` on success, or ``1`` if output files already exist
    and ``config.io.overwrite`` is ``False``.
    """
    if not prepare_outputs(config.io):
        return 1
    work_dir, rm_work_dir = _setup_work_dir(
        config.io.work_dir, overwrite=config.io.overwrite
    )
    try:
        metadata_lines = format_metadata_lines(config, work_dir=work_dir)
        logger.info("%s", metadata_lines[0])
        for line in metadata_lines[1:]:
            logger.debug("%s", line)

        consensus = config.consensus
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

        charge_buckets = load_or_stage_buckets(
            config, work_dir, build_process_spectrum(config)
        )
        bucket_datasets, cluster_results = _run_clustering(
            config, work_dir, charge_buckets
        )
        export_results(config, work_dir, bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
