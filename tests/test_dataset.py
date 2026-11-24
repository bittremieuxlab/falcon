"""Unit tests for falcon.dataset schema, bucket naming, and metadata."""

import pyarrow as pa
import pytest

import falcon.dataset as dataset_mod
from falcon.dataset import (
    BUCKET_PREFIX,
    BUCKET_SUFFIX,
    CANONICAL_SORT_COLUMNS,
    INTERVAL_COLUMNS,
    METADATA_COLUMNS,
    SORTED_INTERVAL_COLUMNS,
    SPECTRA_SUBDIR,
    SPECTRUM_SCHEMA,
    bucket_dataset_name,
    bucket_dataset_path,
    bucket_key_to_str,
    bucket_label_from_uri,
    format_bucket_label,
    is_bucket_dataset_name,
    read_canonical_metadata,
    spectra_dir,
)


class TestDatasetSchemaAndConstants:
    def test_column_groups_and_schema_contract(self):
        assert SPECTRA_SUBDIR == "spectra"
        assert BUCKET_PREFIX == "spectra_charge_"
        assert BUCKET_SUFFIX == ".lance"
        assert CANONICAL_SORT_COLUMNS == (
            "precursor_mz",
            "retention_time",
            "identifier",
        )
        assert METADATA_COLUMNS == (
            "identifier",
            "precursor_charge",
            "precursor_mz",
            "retention_time",
        )
        assert INTERVAL_COLUMNS == (
            "identifier",
            "precursor_mz",
            "precursor_charge",
            "retention_time",
            "mz",
            "intensity",
        )
        expected_schema = pa.schema(
            [
                pa.field("identifier", pa.string()),
                pa.field("precursor_mz", pa.float32()),
                pa.field("precursor_charge", pa.int8()),
                pa.field("mz", pa.list_(pa.float32())),
                pa.field("intensity", pa.list_(pa.float32())),
                pa.field("retention_time", pa.float32()),
            ]
        )
        assert SPECTRUM_SCHEMA.equals(expected_schema)
        schema_names = set(SPECTRUM_SCHEMA.names)
        assert set(CANONICAL_SORT_COLUMNS) <= schema_names
        assert set(METADATA_COLUMNS) <= schema_names
        assert set(INTERVAL_COLUMNS) == schema_names
        assert "SPECTRUM_SCHEMA" in dir(dataset_mod)
        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = dataset_mod.nonexistent_attr


class TestBucketNamingHelpers:
    def test_spectra_dir_and_dataset_path(self, tmp_path):
        assert spectra_dir(tmp_path) == tmp_path / "spectra"
        assert (
            bucket_dataset_path(tmp_path, (1, 2))
            == tmp_path / "spectra" / "spectra_charge_1_2.lance"
        )

    @pytest.mark.parametrize(
        ("bucket", "expected"),
        [
            ((1,), "1"),
            ((1, 2), "1_2"),
            ((-2, -1), "-2_-1"),
            (("unknown",), "unknown"),
            ((3, "unknown"), "3_unknown"),
            ("other", "other"),
        ],
    )
    def test_bucket_key_to_str(self, bucket, expected):
        assert bucket_key_to_str(bucket) == expected

    @pytest.mark.parametrize("bad_bucket", [[1, 2], 2, "1", None])
    def test_bucket_key_to_str_rejects_invalid_type(self, bad_bucket):
        with pytest.raises(TypeError, match="Unsupported bucket key type"):
            bucket_key_to_str(bad_bucket)

    @pytest.mark.parametrize(
        ("bucket", "expected"),
        [
            ((1, 2), "spectra_charge_1_2.lance"),
            (("unknown",), "spectra_charge_unknown.lance"),
            ("other", "spectra_charge_other.lance"),
        ],
    )
    def test_bucket_dataset_name(self, bucket, expected):
        assert bucket_dataset_name(bucket) == expected

    @pytest.mark.parametrize(
        ("bucket", "expected"),
        [
            ((2,), "2"),
            ((1, 2), "1, 2"),
            ((3, "unknown"), "3, unknown"),
            (("unknown",), "unknown"),
            ("other", "other"),
        ],
    )
    def test_format_bucket_label(self, bucket, expected):
        assert format_bucket_label(bucket) == expected

    @pytest.mark.parametrize("bad_bucket", [[1, 2], "unknown", 2])
    def test_format_bucket_label_rejects_invalid_types(self, bad_bucket):
        with pytest.raises(TypeError, match="Unsupported bucket key type"):
            format_bucket_label(bad_bucket)

    @pytest.mark.parametrize(
        "helper",
        [bucket_key_to_str, bucket_dataset_name, format_bucket_label],
    )
    def test_bucket_helpers_reject_empty_tuple(self, helper):
        with pytest.raises(ValueError, match="must not be empty"):
            helper(())

    def test_is_bucket_dataset_name_and_sorted_interval_columns(self):
        assert is_bucket_dataset_name("spectra_charge_2.lance")
        assert is_bucket_dataset_name("spectra_charge_other.lance")
        assert not is_bucket_dataset_name("spectra_charge_.lance")
        assert not is_bucket_dataset_name("custom.lance")
        assert not is_bucket_dataset_name("spectra_charge_2.parquet")
        assert "identifier" not in SORTED_INTERVAL_COLUMNS
        assert set(SORTED_INTERVAL_COLUMNS) == set(INTERVAL_COLUMNS) - {
            "identifier"
        }

    @pytest.mark.parametrize(
        ("uri", "fallback", "expected"),
        [
            ("/data/work/spectra/spectra_charge_1_2.lance", "fb", "1, 2"),
            (
                "/data/work/spectra/spectra_charge_unknown.lance/",
                "fb",
                "unknown",
            ),
            (
                "/data/work/spectra/spectra_charge_3_unknown.lance",
                "fb",
                "3, unknown",
            ),
            ("/data/work/spectra/spectra_charge_other.lance", "fb", "other"),
            ("/data/work/spectra/spectra_charge_.lance", "fb", "fb"),
            (
                "/data/work/custom_table.lance",
                "fallback_label",
                "fallback_label",
            ),
            ("/data/work/spectra_charge_2.parquet", "fb", "fb"),
        ],
    )
    def test_bucket_label_from_uri(self, uri, fallback, expected):
        assert bucket_label_from_uri(uri, fallback) == expected


class TestReadCanonicalMetadata:
    def test_sorts_by_precursor_mz_rt_and_identifier_including_nan_rt_ties(
        self, lance_dataset, spectrum_row
    ):
        nan = float("nan")
        rows = [
            spectrum_row(
                "z_high_mz", 600.0, [100.0], [1.0], retention_time=1.0
            ),
            spectrum_row("nan_b", 500.0, [100.0], [1.0], retention_time=nan),
            spectrum_row("rt20", 500.0, [100.0], [1.0], retention_time=20.0),
            spectrum_row("tie_b", 500.0, [100.0], [1.0], retention_time=10.0),
            spectrum_row("nan_a", 500.0, [100.0], [1.0], retention_time=nan),
            spectrum_row("tie_a", 500.0, [100.0], [1.0], retention_time=10.0),
            spectrum_row("low_mz", 400.0, [100.0], [1.0], retention_time=99.0),
        ]
        ds = lance_dataset(rows)
        expected_order = [
            "low_mz",
            "tie_a",
            "tie_b",
            "rt20",
            "nan_a",
            "nan_b",
            "z_high_mz",
        ]

        df = read_canonical_metadata(
            ds, METADATA_COLUMNS, with_row_index=False
        )
        assert list(df.columns) == list(METADATA_COLUMNS)
        assert "index" not in df.columns
        assert df["identifier"].tolist() == expected_order

        df_idx = read_canonical_metadata(
            ds, METADATA_COLUMNS, with_row_index=True
        )
        assert "index" in df_idx.columns
        assert df_idx["identifier"].tolist() == expected_order
        taken = ds.take(
            df_idx["index"].tolist(), columns=["identifier"]
        ).to_pydict()["identifier"]
        assert taken == expected_order

    def test_rejects_missing_canonical_sort_columns(self, lance_dataset):
        ds = lance_dataset([])
        with pytest.raises(ValueError, match="CANONICAL_SORT_COLUMNS"):
            read_canonical_metadata(ds, ["identifier", "precursor_mz"])
