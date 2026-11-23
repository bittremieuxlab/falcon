"""CLI entry point and pipeline orchestration for Falcon."""

from __future__ import annotations

import functools
import json
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import lance
import natsort
import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus

from . import utils
from .cli import parse_cli, write_saved_config
from .cluster import cluster, spectrum
from .cluster.consensus import ConsensusTuple
from .cluster.spectrum import ProcessedSpectrum
from .config import (
    AverageConsensusConfig,
    ConfigError,
    ConsensusConfig,
    FalconConfig,
    IOConfig,
    build_json_schema,
    build_staging_manifest,
    deserialize_cached_buckets,
    diff_staging_manifests,
)
from .ms_io import ms_io
from .spectra_io import (
    BucketKey,
    _prepare_spectra,
    _write_cluster_info,
    bucket_key_to_str,
    resolve_input_files,
)

logger = logging.getLogger("falcon")

utils.set_seeds()

ClusterResult = tuple[np.ndarray, list[ConsensusTuple]]

_STAGING_CACHE_FILENAME = "staging_cache.json"


def run(config: FalconConfig) -> int:
    """
    Execute the Falcon MS/MS spectrum clustering pipeline.

    Programmatic callers should pass a validated ``FalconConfig`` here.
    Unlike :func:`main`, this function does not configure or shut down
    global logging handlers.

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
    _log_config(config, work_dir)
    try:
        # Abort if outputs already exist and overwrite was not enabled.
        if _validate_outputs(config.io):
            return 1
        _warn_aggressive_outlier_cutoffs(config.consensus)

        process_spectrum = _build_process_spectrum(config)
        charge_buckets = _load_or_prepare_buckets(
            config, work_dir, process_spectrum
        )
        bucket_datasets, cluster_results = _run_clustering(
            config, work_dir, charge_buckets
        )
        _export_results(config, work_dir, bucket_datasets, cluster_results)
        return 0
    finally:
        if rm_work_dir and work_dir.is_dir():
            shutil.rmtree(work_dir)


def main(args: Sequence[str] | None = None) -> int:
    """
    Run the Falcon CLI entry point.

    Parameters
    ----------
    args : Sequence[str] | None, default=None
        Optional CLI arguments. When ``None``, reads ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (``0`` on success, ``1`` on existing-output
        abort, ``2`` on configuration error).
    """
    cli_result = parse_cli(args)
    if cli_result.action == "dump_schema":
        print(json.dumps(build_json_schema(), indent=2))
        return 0
    if cli_result.action == "print_config":
        assert cli_result.settings is not None  # noqa: S101
        print(cli_result.settings.to_toml(), end="")
        return 0

    assert cli_result.config is not None  # noqa: S101
    if cli_result.save_config is not None:
        try:
            write_saved_config(cli_result.config, cli_result.save_config)
        except ConfigError as exc:
            print(f"falcon: error: {exc}", file=sys.stderr)
            return 2

    utils.configure_logger()
    try:
        return run(cli_result.config)
    finally:
        logging.shutdown()


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
    rm_work_dir = False
    if work_dir is None:
        resolved = Path(tempfile.mkdtemp())
        rm_work_dir = True
    else:
        resolved = work_dir
        if resolved.is_dir():
            logger.warning(
                "Working directory %s already exists, previous "
                "results might get overwritten",
                resolved,
            )
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "spectra").mkdir(exist_ok=True)
    return resolved, rm_work_dir


def _check_existing_output(
    io_cfg: IOConfig, suffix: str, description: str
) -> bool:
    """
    Check an existing output file and remove it under ``overwrite``.

    Parameters
    ----------
    io_cfg : IOConfig
        Run I/O settings (``output_filename`` prefix and ``overwrite``).
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
    path = Path(f"{io_cfg.output_filename}.{suffix}")
    if not path.is_file():
        return False
    if io_cfg.overwrite:
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


def _validate_outputs(io_cfg: IOConfig) -> bool:
    """
    Check both output files for pre-existing results.

    Returns
    -------
    bool
        ``True`` if an existing output blocks the run without
        ``overwrite``.
    """
    # Evaluate both without short-circuiting so an existing .mgf file is
    # still checked/removed alongside .csv.
    csv_abort = _check_existing_output(io_cfg, "csv", "cluster assignments")
    mgf_abort = _check_existing_output(
        io_cfg, "mgf", "cluster representatives"
    )
    return csv_abort or mgf_abort


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


def _build_process_spectrum(
    config: FalconConfig,
) -> Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None]:
    """Build the configured single-spectrum preprocessing callable."""
    prep = config.preprocessing
    _, min_mz, max_mz = spectrum.get_dim(
        prep.min_mz, prep.max_mz, config.clustering.fragment_tol
    )
    return functools.partial(
        spectrum.process_spectrum,
        min_peaks=prep.min_peaks,
        min_mz_range=prep.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=prep.remove_precursor_tol,
        min_intensity=prep.min_intensity,
        max_peaks_used=prep.max_peaks_used,
        scaling=None if prep.scaling == "off" else prep.scaling,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON metadata to ``path`` via rename."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=".staging_cache_", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, indent=2)
            f_out.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _has_unverified_staging_files(spectra_dir: Path) -> bool:
    """Return True if ``spectra_dir`` holds unmanifested cache artifacts."""
    for entry in spectra_dir.iterdir():
        if entry.name == "charges.joblib" or entry.name.endswith(".lance"):
            return True
    return False


def _load_or_prepare_buckets(
    config: FalconConfig,
    work_dir: Path,
    process_spectrum: Callable[[sus.MsmsSpectrum], ProcessedSpectrum | None],
) -> list[BucketKey]:
    """
    Return the charge buckets, reusing validated cached spectra if safe.

    Under ``config.io.overwrite``, the intermediate ``spectra``
    directory is cleared and spectra are re-read and re-partitioned.
    Otherwise, a cached staging manifest (``staging_cache.json``) is
    reused only when ``schema_version``, resolved inputs (paths, sizes,
    modification times), ``preprocessing``, ``fragment_tol``, and
    ``precursor_charge_buckets`` all match.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory containing the ``spectra`` folder.
    process_spectrum : Callable
        Configured single-spectrum preprocessing function returning a
        ``ProcessedSpectrum`` or ``None``.

    Returns
    -------
    list[BucketKey]
        Charge bucket keys with staged Lance datasets.
    """
    resolved_inputs = resolve_input_files(config.io.input_filenames)
    expected_manifest = build_staging_manifest(config, resolved_inputs)
    spectra_dir = work_dir / "spectra"
    cache_path = spectra_dir / _STAGING_CACHE_FILENAME

    if config.io.overwrite:
        for entry in spectra_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    elif cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Corrupted staging cache manifest at {cache_path}; "
                "rerun with --overwrite or a different --work-dir."
            ) from exc
        mismatches = diff_staging_manifests(cached, expected_manifest)
        if mismatches:
            diff_str = ", ".join(mismatches)
            raise ValueError(
                f"Cached staging data in {work_dir} does not match the "
                f"current configuration (mismatched: {diff_str}); rerun "
                "with --overwrite or a different --work-dir."
            )
        try:
            return list(deserialize_cached_buckets(cached.get("buckets")))
        except ValueError as exc:
            raise ValueError(
                f"Invalid staging cache buckets in {cache_path}: {exc}; "
                "rerun with --overwrite or a different --work-dir."
            ) from exc
    elif _has_unverified_staging_files(spectra_dir):
        raise ValueError(
            f"Working directory {work_dir} contains an unverified or legacy "
            "staging cache without staging_cache.json; rerun with "
            "--overwrite or a different --work-dir."
        )

    # Stage fresh Lance datasets and write provenance manifest atomically.
    charge_buckets = _prepare_spectra(
        resolved_inputs,
        work_dir,
        process_spectrum,
        config.clustering.precursor_charge_buckets,
    )
    manifest = build_staging_manifest(
        config, resolved_inputs, buckets=charge_buckets
    )
    _write_json_atomic(cache_path, manifest)
    return charge_buckets


def _run_clustering(
    config: FalconConfig,
    work_dir: Path,
    charge_buckets: Sequence[BucketKey],
) -> tuple[list[lance.LanceDataset], list[ClusterResult]]:
    """
    Cluster every non-empty charge bucket in a shared worker pool.

    Clustering all non-empty charge buckets together avoids idle cores
    at per-bucket pool barriers while producing identical partitions to
    clustering each bucket independently.

    Parameters
    ----------
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory containing staged Lance datasets.
    charge_buckets : Sequence[BucketKey]
        Staged charge bucket keys to cluster.

    Returns
    -------
    tuple[list[lance.LanceDataset], list[ClusterResult]]
        Non-empty per-bucket Lance datasets and their positionally
        aligned ``(labels, representative_spectra)`` clustering results.
    """
    clust = config.clustering
    consensus_params: dict[str, Any] = {}
    if isinstance(config.consensus, AverageConsensusConfig):
        consensus_params["min_mz"] = config.preprocessing.min_mz
        consensus_params["max_mz"] = config.preprocessing.max_mz
        consensus_params["bin_size"] = 2 * clust.fragment_tol
        consensus_params["outlier_cutoff_lower"] = (
            config.consensus.outlier_cutoff_lower
        )
        consensus_params["outlier_cutoff_upper"] = (
            config.consensus.outlier_cutoff_upper
        )

    bucket_datasets: list[lance.LanceDataset] = []
    for bucket in charge_buckets:
        dataset_path = (
            work_dir
            / "spectra"
            / f"spectra_charge{bucket_key_to_str(bucket)}.lance"
        )
        dataset = lance.dataset(str(dataset_path))
        # Skip buckets with no surviving spectra.
        if dataset.count_rows() == 0:
            continue
        bucket_datasets.append(dataset)

    cluster_results = cluster.generate_clusters_multi(
        bucket_datasets,
        clust.linkage,
        clust.distance_threshold,
        clust.min_matched_peaks,
        clust.precursor_tol.mass,
        clust.precursor_tol.mode,
        clust.rt_tol,
        clust.fragment_tol,
        clust.batch_size,
        config.consensus.method,
        consensus_params,
    )
    return bucket_datasets, cluster_results


def _export_results(
    config: FalconConfig,
    work_dir: Path,
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
    config : FalconConfig
        Validated run configuration.
    work_dir : Path
        Resolved working directory recorded in output metadata comments.
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
        if config.io.export_representatives:
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
    csv_path = f"{config.io.output_filename}.csv"
    mgf_path = f"{config.io.output_filename}.mgf"
    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_df),
        clusters_df["cluster"].nunique(),
        csv_path,
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
        args=(_write_cluster_info, clusters_df, config, work_dir),
        daemon=True,
    )
    write_csv_worker.start()
    if config.io.export_representatives:
        logger.info(
            "Export %d cluster representative spectra to output file %s",
            len(representatives),
            mgf_path,
        )
        write_mgf_worker = threading.Thread(
            target=_run_writer,
            args=(ms_io.write_spectra, mgf_path, representatives),
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
