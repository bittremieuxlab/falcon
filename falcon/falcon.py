"""CLI entry point and pipeline orchestration for Falcon."""

import functools
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from typing import Any, cast

import joblib
import lance
import natsort
import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus

from . import __version__, utils
from .cluster import cluster, spectrum
from .cluster.consensus import ConsensusTuple
from .cluster.spectrum import ProcessedSpectrum
from .config import config
from .ms_io import ms_io
from .spectra_io import (
    BucketKey,
    _prepare_spectra,
    _write_cluster_info,
    bucket_key_to_str,
)

logger = logging.getLogger("falcon")

utils.set_seeds()

ClusterResult = tuple[np.ndarray, list[ConsensusTuple]]


def main(args: str | list[str] | None = None) -> int:
    """
    Run the Falcon MS/MS spectrum clustering CLI pipeline.

    Parameters
    ----------
    args : str | list[str] | None, default=None
        Optional CLI arguments. When ``None``, reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status: ``0`` on success, or ``1`` if output files
        already exist and ``--overwrite`` was not specified.
    """
    utils.configure_logger()
    config.parse(args)
    _log_config()

    rm_work_dir = _setup_work_dir()
    try:
        # Abort if outputs already exist and --overwrite was not given.
        if _validate_outputs():
            return 1
        _warn_aggressive_outlier_cutoffs()

        process_spectrum = _build_process_spectrum()
        charge_buckets = _load_or_prepare_buckets(process_spectrum)
        bucket_datasets, cluster_results = _run_clustering(charge_buckets)
        _export_results(bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir and os.path.isdir(config.work_dir):
            shutil.rmtree(config.work_dir)
        logging.shutdown()


# Config fields logged at debug level on startup. `precursor_tol` is
# logged separately because it is stored as a (value, unit) pair.
_CONFIG_LOG_FIELDS = [
    "work_dir",
    "overwrite",
    "export_representatives",
    "rt_tol",
    "fragment_tol",
    "linkage",
    "distance_threshold",
    "min_matched_peaks",
    "consensus_method",
    "outlier_cutoff_lower",
    "outlier_cutoff_upper",
    "batch_size",
    "precursor_charge_buckets",
    "min_peaks",
    "min_mz_range",
    "min_mz",
    "max_mz",
    "remove_precursor_tol",
    "min_intensity",
    "max_peaks_used",
    "scaling",
]


def _log_config() -> None:
    """Log the Falcon version and active configuration settings."""
    logger.info("falcon version %s", str(__version__))
    logger.debug("precursor_tol = %.2f %s", *config.precursor_tol)
    for field in _CONFIG_LOG_FIELDS:
        logger.debug("%s = %s", field, getattr(config, field))


def _setup_work_dir() -> bool:
    """
    Create the working directory and its ``spectra`` subdirectory.

    Returns
    -------
    bool
        ``True`` if a temporary working directory was created and
        should be removed on exit, ``False`` otherwise.
    """
    rm_work_dir = False
    if config.work_dir is None:
        config.work_dir = tempfile.mkdtemp()
        rm_work_dir = True
    elif os.path.isdir(config.work_dir):
        logger.warning(
            "Working directory %s already exists, previous "
            "results might get overwritten",
            config.work_dir,
        )
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "spectra"), exist_ok=True)
    return rm_work_dir


def _check_existing_output(suffix: str, description: str) -> bool:
    """
    Check an existing output file and remove it under ``--overwrite``.

    Parameters
    ----------
    suffix : str
        Output file extension without leading dot (``"csv"`` or
        ``"mgf"``).
    description : str
        Human-readable description used in warning and error logs.

    Returns
    -------
    bool
        ``True`` if the file already exists and must not be
        overwritten, indicating that the run should abort.
    """
    path = f"{config.output_filename}.{suffix}"
    if not os.path.isfile(path):
        return False
    if config.overwrite:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            description,
        )
        os.remove(path)
        return False
    logger.error(
        "Output file %s (%s) already exists, aborting...", path, description
    )
    return True


def _validate_outputs() -> bool:
    """
    Check both output files for pre-existing results.

    Returns
    -------
    bool
        ``True`` if an existing output blocks the run without
        ``--overwrite``.
    """
    # Evaluate both without short-circuiting so an existing .mgf file is
    # still checked/removed alongside .csv.
    csv_abort = _check_existing_output("csv", "cluster assignments")
    mgf_abort = _check_existing_output("mgf", "cluster representatives")
    return csv_abort or mgf_abort


def _warn_aggressive_outlier_cutoffs() -> None:
    """Warn when both averaging outlier cutoffs are set below 1."""
    if (
        config.consensus_method == "average"
        and config.outlier_cutoff_lower < 1
        and config.outlier_cutoff_upper < 1
    ):
        logger.warning(
            "Setting both outlier_cutoff_lower and outlier_cutoff_upper "
            "to values less than 1 can lead to unexpected results. It "
            "is advised to set either outlier_cutoff_lower or "
            "outlier_cutoff_upper to a value >= 1."
        )


def _build_process_spectrum() -> Callable[
    [sus.MsmsSpectrum], ProcessedSpectrum | None
]:
    """Build the configured single-spectrum preprocessing callable."""
    _, min_mz, max_mz = spectrum.get_dim(
        config.min_mz, config.max_mz, config.fragment_tol
    )
    return functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == "off" else config.scaling,
    )


def _load_or_prepare_buckets(
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> list[BucketKey]:
    """
    Return the charge buckets, reusing cached spectra when possible.

    Under ``--overwrite``, the intermediate ``spectra`` directory is
    cleared and spectra are re-read and re-partitioned. Otherwise, a
    cached partitioning is reused after validating that it matches the
    current ``--precursor_charge_buckets`` configuration.

    Parameters
    ----------
    process_spectrum : Callable
        Configured single-spectrum preprocessing function returning a
        ``ProcessedSpectrum`` or ``None``.

    Returns
    -------
    list[BucketKey]
        Charge bucket keys with staged Lance datasets.
    """
    if config.overwrite:
        for filename in os.listdir(os.path.join(config.work_dir, "spectra")):
            path = os.path.join(config.work_dir, "spectra", filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    charge_path = os.path.join(config.work_dir, "spectra", "charges.joblib")
    if os.path.isfile(charge_path) and not config.overwrite:
        cached = joblib.load(charge_path)
        # Validate against the configuration that produced the cache,
        # not only the resulting buckets, to catch switching between
        # explicit buckets and auto mode (None).
        if (
            not isinstance(cached, dict)
            or cached["config"] != config.precursor_charge_buckets
        ):
            raise ValueError(
                "Cached charge buckets do not match the current "
                "--precursor_charge_buckets configuration, rerun "
                "with --overwrite or a different --work_dir."
            )
        return cached["buckets"]

    # Recalculate charge buckets and recreate staged Lance datasets.
    charge_buckets = _prepare_spectra(
        process_spectrum, config.precursor_charge_buckets
    )
    joblib.dump(
        {
            "config": config.precursor_charge_buckets,
            "buckets": charge_buckets,
        },
        charge_path,
    )
    return charge_buckets


def _run_clustering(
    charge_buckets: list[BucketKey],
) -> tuple[list[lance.LanceDataset], list[ClusterResult]]:
    """
    Cluster every non-empty charge bucket in a shared worker pool.

    Clustering all non-empty charge buckets together avoids idle cores
    at per-bucket pool barriers while producing identical partitions to
    clustering each bucket independently.

    Parameters
    ----------
    charge_buckets : list[BucketKey]
        Staged charge bucket keys to cluster.

    Returns
    -------
    tuple[list[lance.LanceDataset], list[ClusterResult]]
        Non-empty per-bucket Lance datasets and their positionally
        aligned ``(labels, representative_spectra)`` clustering results.
    """
    consensus_params: dict[str, Any] = {}
    if config.consensus_method == "average":
        consensus_params["min_mz"] = config.min_mz
        consensus_params["max_mz"] = config.max_mz
        consensus_params["bin_size"] = 2 * config.fragment_tol
        consensus_params["outlier_cutoff_lower"] = config.outlier_cutoff_lower
        consensus_params["outlier_cutoff_upper"] = config.outlier_cutoff_upper

    bucket_datasets: list[lance.LanceDataset] = []
    for bucket in charge_buckets:
        dataset_path = os.path.join(
            config.work_dir,
            "spectra",
            f"spectra_charge{bucket_key_to_str(bucket)}.lance",
        )
        dataset = lance.dataset(dataset_path)
        # Skip buckets with no surviving spectra.
        if dataset.count_rows() == 0:
            continue
        bucket_datasets.append(dataset)

    cluster_results = cluster.generate_clusters_multi(
        bucket_datasets,
        config.linkage,
        config.distance_threshold,
        config.min_matched_peaks,
        config.precursor_tol[0],
        config.precursor_tol[1],
        config.rt_tol,
        config.fragment_tol,
        config.batch_size,
        config.consensus_method,
        consensus_params,
    )
    return bucket_datasets, cluster_results


def _export_results(
    bucket_datasets: list[lance.LanceDataset],
    cluster_results: list[ClusterResult],
) -> None:
    """
    Assemble global cluster assignments and write CSV/MGF outputs.

    Cluster labels from different charge buckets are offset so they do
    not overlap, per-bucket metadata tables are concatenated and
    naturally sorted by identifier, and ``.csv`` (plus optional
    ``.mgf``) outputs are written on background IO threads.

    Parameters
    ----------
    bucket_datasets : list[lance.LanceDataset]
        Non-empty per-bucket Lance datasets.
    cluster_results : list[ClusterResult]
        Positionally aligned ``(labels, representative_spectra)``
        results for ``bucket_datasets``.
    """
    clusters_all: list[pd.DataFrame] = []
    current_label = 0
    representatives: list[ConsensusTuple] = []
    for dataset, (clusters, rep_spectra) in zip(
        bucket_datasets, cluster_results, strict=True
    ):
        # Offset labels so different charge buckets never collide.
        clusters += current_label
        rep_spectra = [
            s._replace(cluster_id=s.cluster_id + current_label)
            for s in rep_spectra
        ]
        current_label = int(np.amax(clusters)) + 1
        # Re-read metadata in the canonical order used for clustering.
        metadata = (
            dataset.to_table(
                columns=[
                    "identifier",
                    "precursor_charge",
                    "precursor_mz",
                    "retention_time",
                ]
            )
            .to_pandas()
            # Match `_prepare_bucket_chunks` ordering so cluster labels
            # bind to the right spectra (`identifier` breaks ties).
            .sort_values(["precursor_mz", "retention_time", "identifier"])
        )
        metadata["cluster"] = clusters
        clusters_all.append(metadata)
        if config.export_representatives:
            representatives.extend(rep_spectra)

    # Export cluster assignments and optional representative spectra.
    if clusters_all:
        clusters_df = pd.concat(clusters_all, ignore_index=True).sort_values(
            ["identifier"], key=cast(Any, natsort.natsort_keygen())
        )
    else:
        clusters_df = pd.DataFrame(
            columns=[
                "identifier",
                "precursor_charge",
                "precursor_mz",
                "retention_time",
                "cluster",
            ]
        )
    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        f"{config.output_filename}.csv",
    )
    errors: list[BaseException] = []

    def _run_writer(target: Callable[..., Any], *args: Any) -> None:
        try:
            target(*args)
        except BaseException as exc:
            errors.append(exc)

    # Overlap CSV and MGF writes on background IO threads.
    write_csv_worker = threading.Thread(
        target=_run_writer,
        args=(_write_cluster_info, clusters_df),
        daemon=True,
    )
    write_csv_worker.start()
    if config.export_representatives:
        logger.info(
            "Export %d cluster representative spectra to output file %s",
            len(representatives),
            f"{config.output_filename}.mgf",
        )
        write_mgf_worker = threading.Thread(
            target=_run_writer,
            args=(
                ms_io.write_spectra,
                f"{config.output_filename}.mgf",
                representatives,
            ),
            daemon=True,
        )
        write_mgf_worker.start()
        write_mgf_worker.join()
    write_csv_worker.join()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    sys.exit(main())
