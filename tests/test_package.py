"""Unit tests for falcon package metadata, exports, and __main__."""

import importlib
import importlib.metadata
import subprocess
import sys

import pytest

import falcon
import falcon.__main__
from falcon import cli, config, pipeline, staging
from falcon.staging import cache as staging_cache
from falcon.staging import ingest as staging_ingest

_HEAVY_MODULES = (
    "numpy",
    "numba",
    "scipy",
    "lance",
    "pyarrow",
    "pandas",
    "fastcluster",
    "joblib",
    "spectrum_utils",
    "pyteomics",
)
_LAZY_SUBPACKAGE_FORBIDDEN = (
    "numpy",
    "pyarrow",
    "pandas",
    "lance",
    "joblib",
    "scipy",
    "fastcluster",
    "numba",
    "spectrum_utils",
)


class TestPackageMetadataAndExports:
    def test_version_is_non_empty_string(self):
        assert isinstance(falcon.__version__, str)
        assert falcon.__version__

    def test_version_falls_back_when_package_not_found(self, monkeypatch):
        def _raise_not_found(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
        try:
            importlib.reload(falcon)
            assert falcon.__version__ == "0.0.0.dev0"
        finally:
            monkeypatch.undo()
            importlib.reload(falcon)

    @pytest.mark.parametrize(
        ("attr_name", "expected"),
        [
            ("FalconConfig", config.FalconConfig),
            ("FalconSettings", config.FalconSettings),
            ("load_config", config.load_config),
            ("load_settings", config.load_settings),
            ("run", pipeline.run),
        ],
    )
    def test_lazy_getattr_exposes_public_api(self, attr_name, expected):
        assert attr_name in falcon.__all__
        assert getattr(falcon, attr_name) is expected

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = falcon.nonexistent_attr

    def test_dir_lists_lazy_public_exports(self):
        names = dir(falcon)
        for attr in falcon.__all__:
            assert attr in names

    def test_lightweight_imports_and_cli_inspection_defer_heavy_deps(self):
        script = (
            "import sys\n"
            "import falcon, falcon.types, falcon.utils, "
            "falcon.config, falcon.cli\n"
            "assert falcon.cli.main(['--print-config']) == 0\n"
            "assert falcon.cli.main(['--dump-schema']) == 0\n"
            f"heavy = {_HEAVY_MODULES!r}\n"
            "loaded = [m for m in heavy if m in sys.modules]\n"
            "assert not loaded, f'Heavy modules loaded eagerly: {loaded}'\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_lazy_subpackages_defer_heavy_runtime_deps(self):
        script = (
            "import sys\n"
            "import falcon, falcon.types, falcon.utils, falcon.dataset, "
            "falcon.ms_io, falcon.ms_io.reader_utils, "
            "falcon.cluster, falcon.staging, falcon.staging.cache\n"
            f"forbidden = {_LAZY_SUBPACKAGE_FORBIDDEN!r}\n"
            "loaded = [m for m in forbidden if m in sys.modules]\n"
            "assert not loaded, f'Forbidden modules loaded: {loaded}'\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    def test_consensus_module_does_not_import_pydantic(self):
        script = (
            "import sys\n"
            "import falcon.cluster.consensus\n"
            "assert 'pydantic' not in sys.modules, "
            "'falcon.cluster.consensus eagerly imported pydantic'\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


class TestStagingPackageLazyExports:
    @pytest.mark.parametrize(
        ("attr_name", "expected"),
        [
            ("STAGING_CACHE_FILENAME", staging_cache.STAGING_CACHE_FILENAME),
            (
                "STAGING_CACHE_SCHEMA_VERSION",
                staging_cache.STAGING_CACHE_SCHEMA_VERSION,
            ),
            ("build_staging_manifest", staging_cache.build_staging_manifest),
            (
                "deserialize_cached_buckets",
                staging_cache.deserialize_cached_buckets,
            ),
            ("diff_staging_manifests", staging_cache.diff_staging_manifests),
            ("load_or_stage_buckets", staging_cache.load_or_stage_buckets),
            (
                "serialize_cached_buckets",
                staging_cache.serialize_cached_buckets,
            ),
            ("ChargeRouter", staging_ingest.ChargeRouter),
            ("resolve_input_files", staging_ingest.resolve_input_files),
            ("stage_spectra", staging_ingest.stage_spectra),
        ],
    )
    def test_lazy_getattr_exposes_staging_api(self, attr_name, expected):
        assert attr_name in staging.__all__
        assert getattr(staging, attr_name) is expected

    def test_dir_lists_staging_exports_and_unknown_attr_raises(self):
        names = dir(staging)
        for attr in staging.__all__:
            assert attr in names
        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = staging.nonexistent_attr


class TestMainModule:
    def test_main_module_exposes_cli_main(self):
        assert falcon.__main__.main is cli.main
