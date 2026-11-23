# Falcon Hierarchical Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Falcon's `configargparse`-based mutable global configuration with a strict, immutable, hierarchically composable Pydantic v2 + TOML configuration system, clean CLI, programmatic `run(config)` API, and safe JSON staging-cache provenance.

**Architecture:** Configuration models, strict validation, hierarchical TOML loading (`extends` and deep-merge), deterministic TOML/JSON-Schema export, and staging-cache manifests live in `falcon/config.py` with zero CLI coupling. Command-line parsing, kebab-case overrides, and human-readable validation error formatting live in `falcon/cli.py`. Orchestration in `falcon/falcon.py` and I/O in `falcon/spectra_io.py` accept an immutable `FalconConfig` explicitly, eliminating the global `config` singleton.

**Tech Stack:** Python `>=3.11` (`tomllib`, `argparse`, `json`, `pathlib`), `pydantic>=2.10`, `uv`, `ruff`, `ty`, `pytest` (`--cov-fail-under=95` with branch coverage).

**Spec:** `docs/superpowers/specs/2026-11-23-configuration-design.md`

## Global Constraints

- Python `>=3.11`; replace `configargparse>=1.4` with `pydantic>=2.10` in `pyproject.toml` and update `uv.lock` (net-zero runtime dependency change).
- Strict Ruff linting and formatting: 79-character code lines, 72-character comments/docstrings, NumPy-style docstrings for public APIs, full type annotations.
- `uv run ty check falcon` must pass with `error-on-warning = true`.
- `uv run pytest --cov=falcon --cov-report=term-missing --cov-fail-under=95` (branch coverage) must pass.
- No backwards compatibility required for `configargparse`, `config.ini`, underscore CLI flags, or global `falcon.config.config`.
- Preserve existing numerical and output contracts in `falcon/cluster/` and `falcon/ms_io/` (Arrow `pa.int8()` charge storage, `get_dim` rounding, CSV `#` comments parseable via `pandas.read_csv(..., comment="#")`, MGF export format).
- Write Markdown with one sentence per line, American English, and the Oxford comma.
- Preserve unrelated pre-existing uncommitted work in `src/` (do not `git add -A` or revert unrelated files).

## Review Focus

1. Non-UTF-8 bytes, invalid TOML syntax, or a directory path passed to `-c`, `extends`, or `--save-config` must raise `ConfigError` (exit code `2` on CLI) naming the offending path rather than crashing with an unhandled OS or decoder exception.
2. Output prefixes containing dots (such as `run_v1.2`) must produce `run_v1.2.csv` and `run_v1.2.mgf` without stripping `.2`, and negative charges in `--precursor-charge-buckets` (such as `[-2, -1]`) must parse cleanly without `argparse` misinterpreting them as flags.
3. Indirect `extends` cycles (`a.toml -> sub/b.toml -> ../a.toml`) must raise `ConfigError` with the cycle chain, while valid diamond inheritance (`child -> [left, right]`, both extending `shared.toml`) must succeed without a false cycle error.
4. Switching `consensus.method` from `average` (in a lower layer with custom cutoffs) to `medoid` (in a higher TOML layer or `--consensus-method medoid`) must drop stale `outlier_cutoff_*` keys cleanly, whereas specifying `outlier_cutoff_*` alongside `medoid` in the same layer or against the default `medoid` must fail validation.
5. Modified input file contents (same path, changed size or mtime) or legacy/corrupted files in `<work_dir>/spectra/` (`charges.joblib`, orphaned `.lance`, truncated `staging_cache.json`) without `--overwrite` must fail fast with an actionable `ValueError` instead of reusing stale spectra or appending onto old Lance datasets.

## File Structure

- Modify: `pyproject.toml`, `uv.lock` — swap `configargparse` for `pydantic>=2.10`.
- Rewrite: `falcon/config.py` — strict Pydantic models, validation, hierarchical TOML loading, deep-merge, TOML/metadata/JSON-Schema serialization, staging-cache manifest builder.
- Create: `falcon/cli.py` — `argparse` parser, CLI override extraction, error formatting, `--version`/`--print-config`/`--save-config`/`--dump-schema` handling.
- Modify: `falcon/spectra_io.py` — remove global `config` import; add `resolve_input_files`; pass `work_dir`, `input_filenames`, and `config` explicitly.
- Modify: `falcon/falcon.py` — remove global `config`; add programmatic `run(config: FalconConfig) -> int`; keep CLI `main(args)` owning CLI logging; upgrade staging cache to `spectra/staging_cache.json`.
- Modify: `falcon/__init__.py` — export `FalconConfig`, `FalconSettings`, `load_config`, `load_settings`, `run`.
- Create: `configs/metabolomics.toml` — repository-level example sparse TOML override file.
- Rewrite/Create Tests: `tests/test_config.py`, `tests/test_cli.py`, `tests/conftest.py`, `tests/test_spectra_io.py`, `tests/test_falcon.py`, `tests/integration/test_cli_end_to_end.py`.
- Modify Docs: `README.md`, `AGENTS.md`.

---

### Task 1: Dependencies and Strict Pydantic Configuration Models

**Files:**
- Modify: `pyproject.toml:26-41`
- Modify: `uv.lock`
- Rewrite: `falcon/config.py` (models, sentinels, charge-bucket validation, cross-field validators)
- Rewrite: `tests/test_config.py` (schema, type strictness, bounds, invariants)

**Interfaces:**
- Produces in `falcon/config.py`:
  - `class ConfigError(ValueError)`
  - `ChargeBucket = tuple[int | Literal["unknown"], ...] | Literal["other"]`
  - `class PrecursorTolerance(BaseModel)` — `mass: float = 20.0`, `mode: Literal["ppm", "Da"] = "ppm"`
  - `class PreprocessingConfig(BaseModel)` — `min_peaks: int = 5`, `min_mz_range: float = 250.0`, `min_mz: float = 101.0`, `max_mz: float = 1500.0`, `remove_precursor_tol: float | None = 1.5`, `min_intensity: float = 0.01`, `max_peaks_used: int | None = 50`, `scaling: Literal["off", "root", "log", "rank"] = "off"`
  - `class ClusteringConfig(BaseModel)` — `precursor_tol: PrecursorTolerance`, `rt_tol: float | None = None`, `fragment_tol: float = 0.05`, `linkage: Literal["single", "complete", "average"] = "complete"`, `distance_threshold: float = 0.1`, `min_matched_peaks: int = 0`, `batch_size: int = 32768`, `precursor_charge_buckets: tuple[ChargeBucket, ...] | None = None`
  - `class MedoidConsensusConfig(BaseModel)` — `method: Literal["medoid"] = "medoid"`
  - `class AverageConsensusConfig(BaseModel)` — `method: Literal["average"] = "average"`, `outlier_cutoff_lower: float = 1.5`, `outlier_cutoff_upper: float = 1.5`
  - `ConsensusConfig = Annotated[MedoidConsensusConfig | AverageConsensusConfig, Field(discriminator="method")]`
  - `class PartialIOConfig(BaseModel)` — `input_filenames: tuple[str, ...] | None = None`, `output_filename: Path | None = None`, `work_dir: Path | None = None`, `overwrite: bool = False`, `export_representatives: bool = False`
  - `class IOConfig(PartialIOConfig)` — narrows `input_filenames: tuple[str, ...]` (required, non-empty) and `output_filename: Path` (required, non-empty `.name`)
  - `class FalconSettings(BaseModel)` — `io: PartialIOConfig`, `preprocessing: PreprocessingConfig`, `clustering: ClusteringConfig`, `consensus: ConsensusConfig`
  - `class FalconConfig(FalconSettings)` — narrows `io: IOConfig`

- [ ] **Step 1: Swap `configargparse` for `pydantic>=2.10` in `pyproject.toml` and sync**

Update `dependencies` in `pyproject.toml` (remove `"configargparse>=1.4"`, add `"pydantic>=2.10"`), then run `uv sync` inside `src/`.
Verify `uv run python -c "import pydantic; print(pydantic.__version__)"` succeeds with version `>=2.10`.

- [ ] **Step 2: Write failing unit tests for models, strictness, bounds, sentinels, and invariants in `tests/test_config.py`**

Create `tests/test_config.py` covering:
```python
def test_default_settings_and_immutability():
    cfg = FalconConfig(
        io=IOConfig(input_filenames=("in.mgf",), output_filename=Path("out"))
    )
    assert cfg.preprocessing.min_peaks == 5
    assert cfg.preprocessing.min_mz == 101.0
    assert cfg.preprocessing.max_mz == 1500.0
    assert cfg.preprocessing.min_mz_range == 250.0
    assert cfg.preprocessing.remove_precursor_tol == 1.5
    assert cfg.preprocessing.min_intensity == 0.01
    assert cfg.preprocessing.max_peaks_used == 50
    assert cfg.preprocessing.scaling == "off"
    assert cfg.clustering.precursor_tol == PrecursorTolerance(
        mass=20.0, mode="ppm"
    )
    assert cfg.clustering.rt_tol is None
    assert cfg.clustering.fragment_tol == 0.05
    assert cfg.clustering.linkage == "complete"
    assert cfg.clustering.distance_threshold == 0.1
    assert cfg.clustering.min_matched_peaks == 0
    assert cfg.clustering.batch_size == 32768
    assert cfg.clustering.precursor_charge_buckets is None
    assert cfg.consensus == MedoidConsensusConfig()
    assert hash(cfg) == hash(cfg)
    with pytest.raises(ValidationError):
        cfg.clustering.batch_size = 10  # type: ignore[misc]
```
Also add parametrized tests for:
- Strict typing: rejecting `bool` for `int`/`float`, `float` for `int`, numeric `str` for `int`/`float`, non-`bool` (`1`, `"true"`) for `bool`, unknown keys (`extra="forbid"`), while accepting `int` for `float` (e.g., `min_mz=0`).
- Finiteness & bounds: `nan`/`inf`/`-inf` on all float fields; `distance_threshold` at `0.0`, `-0.1`, `1.0`, `1.2`; `fragment_tol <= 0`; `precursor_tol.mass < 0` and case-sensitive mode `"PPM"`/`"da"`; `rt_tol < 0` (accepting `0.0`); `min_peaks < 1`; `min_mz < 0`; `max_mz <= 0`; `min_mz_range < 0`; `remove_precursor_tol <= 0`; `min_intensity < 0` and `>= 1.0`; `max_peaks_used < 1`; `min_matched_peaks < 0`; `batch_size < 1`; `outlier_cutoff_lower <= 0` and `outlier_cutoff_upper <= 0`.
- Nullable TOML sentinels: `"none"` normalizing to `None` for `work_dir`, `rt_tol`, `remove_precursor_tol`, `max_peaks_used`, and `"auto"` normalizing to `None` for `precursor_charge_buckets`; rejecting wrong sentinel strings (e.g., `"off"` or `"none"` for `precursor_charge_buckets`).
- `IOConfig` validation: rejecting empty `input_filenames`, blank strings in `input_filenames`, empty `output_filename`, directory-only `output_filename` (`Path(".")`, `Path("/")`), and preserving dotted prefix `Path("run_v1.2")`.
- Charge buckets: accepting lists/tuples/sets canonicalized to sorted tuples with `"unknown"` last; rejecting empty outer list `[]`, empty inner bucket `[]`, duplicate within same bucket `[2, 2]`, overlap across buckets `[[1, 2], [2, 3]]`, duplicate `"unknown"`, duplicate `"other"`, `"other"` inside a list `[["other"]]`, `bool` inside bucket `[[True]]`, out-of-int8 charges `[[-129]]` and `[[128]]`, bare string `"12"`.
- Cross-field and cross-section validators: `min_mz >= max_mz`, `min_mz_range > max_mz - min_mz`, `max_peaks_used < min_peaks`, `min_matched_peaks > max_peaks_used`, subnormal `fragment_tol` overflowing `(max_mz - min_mz) / fragment_tol` or `int32` bin count, and `MedoidConsensusConfig` rejecting `outlier_cutoff_lower`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v` (from `src/`)
Expected: FAIL (`ImportError` for new model names).

- [ ] **Step 4: Implement models and validators in `falcon/config.py`**

Implement all models with `ConfigDict(frozen=True, extra="forbid", strict=True)`, `BeforeValidator` helpers for `ConfigPath`, nullable `"none"`/`"auto"` sentinels (with `WithJsonSchema` so JSON Schema includes the sentinel literals), charge-bucket validation and canonical sorting `(isinstance(x, str), x)` with explicit `not isinstance(x, bool)` and `-128 <= x <= 127` bounds, discriminated union `ConsensusConfig`, and `@model_validator(mode="after")` cross-field/cross-section checks (checking `math.isfinite((max_mz - min_mz) / fragment_tol)` and `n_bins <= 2**31 - 1` for `bin_size = 2 * fragment_tol`).

- [ ] **Step 5: Run tests and typecheck to verify Task 1 passes**

Run: `uv run pytest tests/test_config.py -v && uv run ty check falcon/config.py`
Expected: All Task 1 tests PASS; `ty` reports zero errors/warnings.

---

### Task 2: Hierarchical TOML Loader, `extends`, Deep-Merge, TOML Export, JSON Schema, and Staging Manifest

**Files:**
- Modify: `falcon/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: Task 1 models (`ConfigError`, `FalconSettings`, `FalconConfig`, `ChargeBucket`, `AverageConsensusConfig`, `MedoidConsensusConfig`).
- Produces in `falcon/config.py`:
  - `merge_config_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]`
  - `load_toml_file(path: Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]`
  - `load_settings(config_files: Sequence[str | Path] = (), overrides: Mapping[str, Any] | None = None) -> FalconSettings`
  - `load_config(config_files: Sequence[str | Path] = (), overrides: Mapping[str, Any] | None = None) -> FalconConfig`
  - `FalconSettings.to_toml(self, *, work_dir: Path | None = None) -> str`
  - `FalconSettings.to_metadata_lines(self, *, work_dir: Path | None = None) -> list[str]`
  - `build_json_schema() -> dict[str, Any]`
  - `STAGING_CACHE_SCHEMA_VERSION: int = 1`
  - `build_staging_manifest(config: FalconConfig, resolved_inputs: Sequence[Path], buckets: Sequence[ChargeBucket] | None = None) -> dict[str, Any]`
  - `diff_staging_manifests(cached: Any, expected: Mapping[str, Any]) -> list[str]`
  - `deserialize_cached_buckets(raw_buckets: Any) -> list[ChargeBucket]`

- [ ] **Step 1: Write failing unit tests in `tests/test_config.py` for layering, `extends`, export, schema, and staging manifests**

Add test classes covering:
- `merge_config_dicts`: pure non-mutating deep-merge of nested dicts (`preprocessing`, `clustering.precursor_tol`), atomic replacement of lists (`input_filenames`, `precursor_charge_buckets`), discriminated-union rule for `consensus` (switching `average -> medoid` drops stale `outlier_cutoff_*`; partial override of `outlier_cutoff_upper` without `method` keeps `average` and `outlier_cutoff_lower`; same-layer `medoid` + `outlier_cutoff_lower` still fails validation).
- `load_settings` / `load_config` and `extends`:
  - Sparse override TOML only specifying changed fields (e.g. metabolomics settings), leaving defaults intact.
  - Repeatable `config_files=[base, child]` left-to-right precedence, plus `overrides` highest precedence.
  - `extends = "base.toml"` and `extends = ["a.toml", "b.toml"]` resolving relative to declaring file's directory (even when CWD differs).
  - Review Focus #3: indirect cycle `a.toml -> sub/b.toml -> ../a.toml` raises `ConfigError` naming the cycle; valid diamond DAG (`child -> [left, right]`, both extending `shared.toml`) succeeds cleanly.
  - Review Focus #1: non-UTF-8 bytes, malformed TOML syntax, missing file, directory path passed as config file, and invalid `extends` types (`123`, `[]`, `[""]`, `[1]`) all raise `ConfigError` naming the file.
  - Unsetting inherited parent values via `"none"` (for `rt_tol`, `remove_precursor_tol`, `max_peaks_used`, `work_dir`) and `"auto"` (for `precursor_charge_buckets`).
  - No auto-loading of `./config.ini` or `./falcon.toml` when present in CWD.
- `to_toml()` round-trip: serialize default `FalconSettings`, customized `FalconConfig` with `AverageConsensusConfig`, explicit charge buckets, and disabled `None` fields (verify commented disabled fields, verify `tomllib.loads` parses cleanly, verify `load_config` reproduces an equal `FalconConfig`).
- `to_metadata_lines(work_dir=...)`: includes `falcon version`, resolved `work_dir`, all `io`/`preprocessing`/`clustering`/`consensus` keys, and omits `outlier_cutoff_*` under `medoid` while including them under `average`.
- `build_json_schema()`: valid JSON-serializable schema exposing `extends` (`string | array[string]`), optional `io.input_filenames`/`io.output_filename`, and nullable TOML sentinels (`"none"`, `"auto"`).
- `build_staging_manifest` / `diff_staging_manifests` / `deserialize_cached_buckets`:
  - Changing `distance_threshold`, `linkage`, `rt_tol`, `min_matched_peaks`, `batch_size`, `consensus`, or `output_filename` yields empty diff `[]`.
  - Changing `preprocessing`, `fragment_tol`, `precursor_charge_buckets`, input file list, or input file content (size/mtime — Review Focus #5) returns the exact mismatched section names.
  - Corrupted/non-dict/wrong `schema_version` manifest returns a mismatch indicator; `deserialize_cached_buckets` converts JSON lists back to `tuple` while preserving `"other"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (`ImportError` for loader/export/manifest helpers).

- [ ] **Step 3: Implement hierarchical loader, TOML emitter, JSON Schema, and staging manifest helpers in `falcon/config.py`**

Implement:
- `merge_config_dicts` using `copy.deepcopy` so inputs are never mutated, applying the `consensus.method` discriminator replacement check only when both layers have `"consensus"` dicts and `"method"` changes.
- `load_toml_file` checking `path.is_dir()`, reading UTF-8 bytes with `tomllib.loads`, validating and recursively resolving `extends` using an immutable active-path stack `_stack` (so cycles fail with full path chain while diamond DAGs succeed), catching `OSError`, `UnicodeDecodeError`, and `tomllib.TOMLDecodeError` into `ConfigError`.
- Deterministic TOML serializer in `to_toml` escaping strings via `json.dumps` (valid TOML basic strings), formatting bools/ints/floats/lists cleanly, commenting `None` fields with their TOML sentinel, and printing `[io]`, `[preprocessing]`, `[clustering]`, `[clustering.precursor_tol]`, `[consensus]`.
- `to_metadata_lines` producing canonical `section.key = value` lines from the validated model.
- `build_json_schema` augmenting `FalconSettings.model_json_schema()` with top-level optional `extends` property.
- `build_staging_manifest`, `diff_staging_manifests`, and `deserialize_cached_buckets` using `os.stat` for `(path, size, mtime_ns)`.

- [ ] **Step 4: Run tests and typecheck to verify Task 2 passes**

Run: `uv run pytest tests/test_config.py -v && uv run ty check falcon/config.py`
Expected: All `test_config.py` tests PASS; `ty` clean.

---

### Task 3: CLI Parser, Override Extraction, and Error Formatting (`falcon/cli.py`)

**Files:**
- Create: `falcon/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes from `falcon/config.py`: `ConfigError`, `FalconConfig`, `FalconSettings`, `build_json_schema`, `load_config`, `load_settings`.
- Produces in `falcon/cli.py`:
  - `class CliResult(NamedTuple)` (or frozen `@dataclass`): `action: Literal["run", "print_config", "dump_schema"]`, `config: FalconConfig | None`, `settings: FalconSettings | None`, `save_config: Path | None`
  - `build_parser() -> argparse.ArgumentParser`
  - `format_validation_error(exc: ValidationError) -> str`
  - `parse_cli(args: Sequence[str] | None = None) -> CliResult`
  - `write_saved_config(config: FalconConfig, destination: Path) -> None`

- [ ] **Step 1: Write failing unit tests in `tests/test_cli.py`**

Cover:
- Positional `paths`: `["a.mgf", "b.mzml", "out"]` sets `io.input_filenames=("a.mgf", "b.mzml")` and `io.output_filename=Path("out")`; single positional `["only_one.mgf"]` exits with code `2`; 0 positionals with `-c full_run.toml` succeeds.
- Named `-i`/`--input` and `-o`/`--output`: overriding only `-o out_v2` on top of `-c run.toml`; mixing positional `paths` with `-i` or `-o` exits with code `2`.
- `SUPPRESS` layering: omitted CLI flags do not override TOML file values; explicit CLI flags override TOML values; `--no-overwrite` and `--no-export-representatives` turn off `true` set in a TOML file.
- Scientific flags: kebab-case flags for all preprocessing, clustering, and consensus options; `--precursor-tol 10 ppm` sets both mass and mode; `--rt-tol none`, `--remove-precursor-tol none`, `--max-peaks-used none`, `--work-dir none` unset parent TOML values; `--precursor-charge-buckets auto` resets parent buckets to `None`.
- Review Focus #2: `--precursor-charge-buckets "[-2, -1]" "[1]" unknown other` parses cleanly into `((-2, -1), (1,), ("unknown",), "other")`; empty token `[2,,3]` exits with code `2`; dotted output prefix `run_v1.2` preserved as `Path("run_v1.2")`; negative `--distance-threshold -0.1` exits with code `2` with formatted message mentioning `clustering.distance_threshold`.
- Review Focus #4: base TOML with `consensus.method = "average"` and cutoffs + `--consensus-method medoid` succeeds; default `medoid` + `--outlier-cutoff-lower 2.0` exits with code `2` with readable stderr error.
- Inspection/utility flags: `--version` prints version and exits `0`; `--print-config` works without positional inputs/output and returns `action="print_config"` with resolved `FalconSettings`; `--dump-schema` returns `action="dump_schema"`; `--save-config path` writes round-trippable TOML and rejects existing directory paths (Review Focus #1); combining `--print-config` with `--dump-schema` or `--save-config` exits with code `2`.
- Review Focus #1: nonexistent `-c`, directory `-c`, or malformed TOML exits with code `2` and prints clean `ConfigError` message to stderr (no traceback).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'falcon.cli'`).

- [ ] **Step 3: Implement `falcon/cli.py`**

Implement:
- `build_parser()` with grouped help sections (`I/O`, `Configuration`, `Preprocessing`, `Clustering`, `Consensus`), `--version` via `action="version"`, `default=argparse.SUPPRESS` on every option, `argparse.BooleanOptionalAction` for `--overwrite` and `--export-representatives`, strict typed parsers for ints/floats/`none` sentinels, and charge-bucket token parsing (supporting `auto`, `other`, and comma-separated ints/`unknown` with optional outer `[...]` while rejecting empty tokens between commas).
  Note on negative bracketed tokens like `[-2, -1]`: bracketed tokens do not start with `-`, so `argparse` accepts them directly; unbracketed negative numbers passed to `--distance-threshold -0.1` are handled natively by `argparse` when numeric.
- `format_validation_error(exc: ValidationError) -> str`: formats each Pydantic error with its dotted `loc` path (excluding internal union branch class names like `MedoidConsensusConfig` from paths where redundant) and concise message.
- `parse_cli(args)`: handles `--dump-schema` immediately, parses options into sparse nested override dict, resolves via `load_settings` (for `--print-config`) or `load_config` (for normal run), validates `--save-config` path is not an existing directory, and converts `ConfigError` / `ValidationError` into `parser.error(...)` (exit code `2`).
- `write_saved_config(config, destination)`: rejects directory destinations with `ConfigError` and writes `config.to_toml()` as UTF-8.

- [ ] **Step 4: Run tests and typecheck to verify Task 3 passes**

Run: `uv run pytest tests/test_cli.py -v && uv run ty check falcon/cli.py`
Expected: All `test_cli.py` tests PASS; `ty` clean.

---

### Task 4: Pipeline & I/O Refactor (`falcon/spectra_io.py`, `falcon/falcon.py`, `falcon/__init__.py`) and Test Suite Migration

**Files:**
- Modify: `falcon/spectra_io.py`
- Modify: `falcon/falcon.py`
- Modify: `falcon/__init__.py`
- Modify: `tests/conftest.py` (remove `reset_global_config`)
- Modify: `tests/test_spectra_io.py`
- Modify: `tests/test_falcon.py`
- Modify: `tests/integration/test_cli_end_to_end.py`

**Interfaces:**
- Consumes from `falcon/config.py` and `falcon/cli.py`:
  - `AverageConsensusConfig`, `ChargeBucket`, `FalconConfig`, `FalconSettings`, `build_json_schema`, `build_staging_manifest`, `deserialize_cached_buckets`, `diff_staging_manifests`, `load_config`, `load_settings`, `parse_cli`, `write_saved_config`
- Produces:
  - In `falcon/spectra_io.py`:
    - `resolve_input_files(patterns: Sequence[str]) -> list[Path]` (expands each glob sorted lexicographically, preserving duplicates, raising `FileNotFoundError` if empty)
    - `_prepare_spectra(input_filenames: Sequence[str | os.PathLike[str]], work_dir: str | os.PathLike[str], process_spectrum: ..., charge_buckets: Sequence[BucketKey] | None) -> list[BucketKey]`
    - `_create_lance_dataset(charge_bucket: BucketKey, schema: pa.Schema, work_dir: str | os.PathLike[str]) -> lance.LanceDataset`
    - `_write_spectra_lance(..., work_dir: str | os.PathLike[str]) -> None`
    - `_write_cluster_info(clusters: pd.DataFrame, config: FalconConfig, work_dir: Path) -> None`
  - In `falcon/falcon.py`:
    - `run(config: FalconConfig) -> int` (does not call `configure_logger()` or `logging.shutdown()`)
    - `main(args: Sequence[str] | None = None) -> int` (calls `parse_cli`, handles `--dump-schema`/`--print-config`/`--save-config`, configures and shuts down CLI logging, delegates to `run(config)`)
    - Refactored helpers accepting explicit `config` / `work_dir`: `_log_config`, `_setup_work_dir`, `_validate_outputs`, `_warn_aggressive_outlier_cutoffs`, `_build_process_spectrum`, `_load_or_prepare_buckets`, `_run_clustering`, `_export_results`.
  - In `falcon/__init__.py`:
    - Exports `__version__`, `FalconConfig`, `FalconSettings`, `load_config`, `load_settings`, `run`.

- [ ] **Step 1: Update unit and integration tests first to target the new explicit-config API and staging cache**

Update:
- `tests/conftest.py`: remove `_global_config` import and `reset_global_config` fixture; add a small helper factory `make_falcon_config(input_filenames, output_filename, **overrides)` if useful across tests.
- `tests/test_spectra_io.py`: remove global config usage; pass `work_dir`, `input_filenames`, and `FalconConfig` explicitly; test `resolve_input_files` deterministic sorting and unmatched glob `FileNotFoundError`; verify `_write_cluster_info` writes `# falcon version ...` and `section.key = value` header comments parseable by `pandas.read_csv(..., comment="#")` and preserves dotted prefix `run_v1.2.csv` (Review Focus #2).
- `tests/test_falcon.py`: update all helper tests (`TestSetupWorkDir`, `TestValidateOutputs`, `TestWarnAggressiveOutlierCutoffs`, `TestBuildProcessSpectrum`, `TestLoadOrPrepareBuckets`, `TestRunClusteringAndExportResults`) to pass `FalconConfig` and `work_dir` explicitly; add tests verifying:
  - `_setup_work_dir` does not mutate `config` and returns `(Path, True/False)`.
  - `staging_cache.json` reuse when clustering/consensus/output settings change.
  - Review Focus #5: cache mismatch raises `ValueError` naming mismatched section on `preprocessing` change, `fragment_tol` change, `precursor_charge_buckets` change, input file content/size/mtime change, corrupted `staging_cache.json`, or legacy `charges.joblib` / orphaned `.lance` files without `--overwrite`.
  - `run(config)` executes programmatically without shutting down logging handlers.
- `tests/integration/test_cli_end_to_end.py`: update CLI invocations to new kebab-case flags and hierarchical TOML (`-c base.toml -c child.toml` and `extends`); add end-to-end tests for `--print-config`, `--save-config` (replaying saved config via `falcon -c saved.toml`), `--dump-schema`, dotted output prefix `run_v1.2`, cache reuse across `--distance-threshold` sweep + cache rejection on `--min-mz` change, programmatic `falcon.run(cfg)`, and temp work-dir cleanup (verified via `monkeypatch` on `tempfile.mkdtemp` instead of inspecting mutated global `config.work_dir`).

- [ ] **Step 2: Run updated tests to verify they fail before refactoring production modules**

Run: `uv run pytest tests/test_spectra_io.py tests/test_falcon.py tests/integration/test_cli_end_to_end.py -v`
Expected: FAIL (signature mismatches / old global `config` references).

- [ ] **Step 3: Refactor `falcon/spectra_io.py`, `falcon/falcon.py`, and `falcon/__init__.py`**

Implement:
- `falcon/spectra_io.py`: remove `from .config import config`; add `resolve_input_files`; thread explicit `input_filenames`, `work_dir`, and `config` through `_prepare_spectra`, `_create_lance_dataset`, `_write_spectra_lance`, `_write_to_dataset`, and `_write_cluster_info` (using `Path(f"{config.io.output_filename}.csv")` and `config.to_metadata_lines(work_dir=work_dir)`).
- `falcon/falcon.py`:
  - Remove `from .config import config`.
  - Implement `run(config: FalconConfig) -> int` and update `main(args)` to call `parse_cli(args)`, handle `--dump-schema` (print formatted JSON, return `0`) and `--print-config` (print `settings.to_toml()`, return `0`), optionally call `write_saved_config`, then configure logger, run `run(config)`, and call `logging.shutdown()` in `finally`.
  - In `_load_or_prepare_buckets(config, work_dir, process_spectrum)`: expand inputs via `resolve_input_files(config.io.input_filenames)`, compute expected manifest via `build_staging_manifest`, handle `--overwrite` clearing `<work_dir>/spectra/`, validate existing `spectra/staging_cache.json` via `diff_staging_manifests` (or fail fast if legacy `charges.joblib` / orphaned `.lance` exists without a manifest), and write `staging_cache.json` via atomic write (temp file + `os.replace`) after staging!
  - In `_validate_outputs` and `_export_results`: always construct output paths with `Path(f"{config.io.output_filename}.{suffix}")` (never `.with_suffix`).
  - In `_run_clustering`: narrow `isinstance(config.consensus, AverageConsensusConfig)` cleanly for `consensus_params`, pass `config.clustering.precursor_tol.mass` and `config.clustering.precursor_tol.mode`, etc.
- `falcon/__init__.py`: export `__version__`, `FalconConfig`, `FalconSettings`, `load_config`, `load_settings`, `run`.

- [ ] **Step 4: Run full test suite and typecheck**

Run: `uv run pytest -v && uv run ty check falcon`
Expected: ALL tests across the entire repository PASS; `ty check falcon` exits `0`.

---

### Task 5: Reference Metabolomics Config, Documentation, and Full Verification

**Files:**
- Create: `configs/metabolomics.toml`
- Modify: `tests/test_config.py` (add smoke test loading `configs/metabolomics.toml`)
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `pyproject.toml` (review `ANN401` comment accuracy)

**Interfaces:**
- Consumes: `falcon/config.py`, `falcon/cli.py`, `falcon/falcon.py`, `falcon/__init__.py`.

- [ ] **Step 1: Add `configs/metabolomics.toml` and smoke test**

Create `configs/metabolomics.toml`:
```toml
[preprocessing]
min_mz = 0.0
max_mz = 1000.0
min_mz_range = 150.0

[clustering]
min_matched_peaks = 6
```
Add a test in `tests/test_config.py` loading `configs/metabolomics.toml` via `load_settings` and asserting the overridden fields (`min_mz == 0.0`, `max_mz == 1000.0`, `min_mz_range == 150.0`, `min_matched_peaks == 6`) alongside unchanged defaults (`min_peaks == 5`, `fragment_tol == 0.05`, `consensus == MedoidConsensusConfig()`).

- [ ] **Step 2: Update `README.md`, `AGENTS.md`, and `pyproject.toml` comment**

- `README.md`: document hierarchical TOML configuration, `extends` and repeatable `-c`/`--config`, `configs/metabolomics.toml` sparse override example, kebab-case CLI flags, `--print-config`/`--save-config`/`--dump-schema`, nullable `"none"`/`"auto"` sentinels, and programmatic `falcon.run(config)` usage.
- `AGENTS.md`: update file table (`falcon/config.py`, `falcon/cli.py`, `tests/test_cli.py`), execution and configuration section (no global singleton, TOML layering, no CWD auto-loading, kebab-case flags, `staging_cache.json` provenance), and validation/discrepancies bullets. Follow one-sentence-per-line Markdown convention.
- `pyproject.toml`: check whether `ANN401` is still needed for `**kwargs`/`Any` boundaries in other modules (`falcon/cluster/`, `falcon/config.py` deep-merge) and keep the comment accurate.

- [ ] **Step 3: Run complete local verification gate**

Run from `src/`:
```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check falcon
uv run pytest --cov=falcon --cov-report=term-missing --cov-fail-under=95
uv run pip-audit --local --skip-editable
uv build
```
Plus wheel smoke test:
```bash
wheel="$(ls dist/*.whl | tail -n 1)"
uv run --isolated --no-project --with "$wheel" falcon --help
uv run --isolated --no-project --with "$wheel" falcon --print-config
uv run --isolated --no-project --with "$wheel" falcon --dump-schema
```
Expected: every check exits `0`, branch coverage `>= 95%`, zero lint/type/audit warnings, built wheel CLI commands work cleanly.
