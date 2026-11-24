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
from .cluster import (
    ClusterResult,
    ConsensusTuple,
    generate_clusters_multi,
    offset_cluster_result,
)
from .config import format_metadata_lines
from .staging import (
    METADATA_COLUMNS,
    bucket_dataset_path,
    format_bucket_label,
    load_or_stage_buckets,
    read_canonical_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import FalconConfig, IOConfig

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [*METADATA_COLUMNS, "cluster"]


def prepare_outputs(io_cfg: IOConfig) -> bool:
    """
    Validate output paths before running and report whether to proceed.

    Verifies the output directory exists, rejects directory targets, and
    returns ``False`` when an output already exists without
    ``overwrite``.
    """
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
        for out_path in (config.io.csv_path, config.io.mgf_path):
            if out_path.is_file() or out_path.is_symlink():
                out_path.unlink()

    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        config.io.csv_path,
    )
    with ms_io.atomic_write_text(config.io.csv_path, prefix=".csv_") as f_out:
        for line in format_metadata_lines(config, work_dir=work_dir):
            f_out.write(f"# {line}\n")
        f_out.write("#\n")
        clusters_df.to_csv(f_out, index=False, chunksize=1_000_000)

    if export_reps:
        logger.info(
            "Export %d cluster representative spectra to output file %s",
            len(representatives),
            config.io.mgf_path,
        )
        ms_io.write_spectra(config.io.mgf_path, representatives)


def run(config: FalconConfig) -> int:
    """
    Execute the Falcon MS/MS spectrum clustering pipeline.

    Returns ``0`` on success, or ``1`` if output files already exist
    and ``config.io.overwrite`` is ``False``.
    """
    if not prepare_outputs(config.io):
        return 1

    if config.io.work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
        rm_work_dir = True
    else:
        # Absolutize explicit work_dir so Lance URIs and memmap paths
        # stay valid in warm loky workers across os.chdir().
        work_dir = Path(os.path.abspath(config.io.work_dir))
        if work_dir.is_dir():
            action = (
                "overwritten" if config.io.overwrite else "reused if available"
            )
            logger.warning(
                "Working directory %s already exists; cached staging files "
                "will be %s",
                work_dir,
                action,
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        rm_work_dir = False

    try:
        metadata_lines = format_metadata_lines(config, work_dir=work_dir)
        logger.info("%s", metadata_lines[0])
        for line in metadata_lines[1:]:
            logger.debug("%s", line)

        consensus = config.consensus
        if (
            consensus.method == "average"
            and consensus.outlier_cutoff_lower < 1.0
            and consensus.outlier_cutoff_upper < 1.0
        ):
            logger.warning(
                "Setting both outlier_cutoff_lower and outlier_cutoff_upper "
                "to values less than 1 can lead to unexpected results. It "
                "is advised to set either outlier_cutoff_lower or "
                "outlier_cutoff_upper to a value >= 1."
            )

        charge_buckets = load_or_stage_buckets(config, work_dir)
        bucket_datasets = [
            lance.dataset(str(bucket_dataset_path(work_dir, bucket)))
            for bucket in charge_buckets
        ]
        cluster_results = (
            generate_clusters_multi(
                bucket_datasets,
                config.clustering,
                config.consensus,
                min_mz=config.preprocessing.min_mz,
                max_mz=config.preprocessing.max_mz,
                bucket_labels=[format_bucket_label(b) for b in charge_buckets],
                compute_representatives=config.io.export_representatives,
                temp_dir=work_dir,
            )
            if bucket_datasets
            else []
        )
        export_results(config, work_dir, bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
