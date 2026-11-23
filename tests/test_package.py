"""Unit tests for falcon package metadata, exports, and __main__."""

import importlib
import importlib.metadata

import pytest

import falcon
import falcon.__main__
from falcon import cli, config, pipeline


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


class TestMainModule:
    def test_main_module_exposes_cli_main(self):
        assert falcon.__main__.main is cli.main
