# Falcon Hierarchical Configuration Design

## Intent and Why

Falcon's current configuration system wraps `configargparse` around an untyped, mutable process-wide `config` singleton backed by `dict[str, Any]` and INI files.
It validates only a handful of settings (`precursor_tol`, `distance_threshold > 0`, `batch_size >= 1`, and charge buckets), silently auto-loads `config.ini` from the current working directory, mutates `config.work_dir` at runtime, duplicates the field list across logging and CSV headers, and validates `--work_dir` caches only against charge buckets rather than full preprocessing settings.

As Falcon v2 serves both proteomics and untargeted metabolomics workloads, configuration handling must become robust, strictly validated, and safe by default.
Backwards compatibility with existing CLI flags, `config.ini`, and the global `config` singleton is explicitly not required.

## Goals, Non-Goals, and Anti-Goals

### Goals

- Enforce strict static and runtime types (rejecting implicit coercion such as booleans as integers, floats as integers, strings as numbers, and unknown keys) across Python, config files, and CLI inputs.
- Enforce domain bounds, finiteness (`allow_inf_nan=False`), and cross-field scientific invariants before any I/O or clustering begins.
- Support hierarchical, sparse configuration layering where standard defaults live at the base and users override only changed settings via composable config files (`extends` and repeatable `-c`/`--config`) and CLI flags.
- Eliminate mutable process-wide configuration state in favor of an immutable, hashable `FalconConfig` passed explicitly through the pipeline, enabling safe programmatic execution and strict `ty` type checking.
- Strengthen run safety and reproducibility: no silent CWD config auto-loading, safe JSON staging-cache manifests that check all preprocessing-affecting parameters and input metadata, deterministic TOML export (`--print-config` / `--save-config`), and JSON Schema export (`--dump-schema`).

### Non-Goals

- Hardcoded named domain profiles (such as `--profile proteomics` or `--profile metabolomics`) in CLI parsing or Python enums.
- Backwards compatibility with `configargparse`, `config.ini`, underscore CLI flags, or the global `falcon.config.config` singleton.
- Runtime template/expression engines or `${...}` variable interpolation across config values.
- Environment-variable configuration layers (`FALCON_*`), which are omitted to keep scientific run provenance explicit in files and CLI invocations.

### Anti-Goals

- Accepting silent no-op configurations (such as setting `outlier_cutoff_lower` when `consensus.method = "medoid"`, or specifying `min_matched_peaks > max_peaks_used`).
- Allowing YAML 1.1 implicit typing traps (such as unquoted `off` parsing as `False`).
- Silently reusing stale Lance staging caches in `--work-dir` when preprocessing settings, `fragment_tol`, charge buckets, or input files have changed.
- Leaving `config` mutable or untyped (`Any`) in core orchestration modules.

## Constraints

- Python `>= 3.11` (configured in `pyproject.toml`), so `tomllib` is available in the standard library.
- Strict Ruff linting/formatting (79-character code lines, 72-character comments/docstrings, NumPy docstrings, full annotations) and `ty check falcon` with `error-on-warning = true`.
- CI enforces `pytest --cov=falcon --cov-fail-under=95` with branch coverage across Python 3.11–3.13 on Linux and Python 3.12 on macOS, plus `pip-audit` and wheel smoke testing (`falcon --help`).
- Preserve existing numerical and output contracts in `falcon/cluster/` and `falcon/ms_io/` (Arrow `pa.int8()` charge storage, `get_dim` rounding, CSV `#` comment header format readable via `pandas.read_csv(path, comment="#")`, and MGF output format).

## Approaches Considered

### 1. Pydantic v2 + layered TOML (`tomllib`) + stdlib `argparse` *(Selected)*

Define immutable, strict nested Pydantic v2 models (`PreprocessingConfig`, `ClusteringConfig`, `ConsensusConfig`, `IOConfig`, `FalconConfig`), resolve hierarchical TOML files (`extends` + repeatable `-c`) via stdlib `tomllib` with an explicit deep-merge, and parse CLI overrides via stdlib `argparse` using `SUPPRESS` defaults.
Replaces `configargparse` with `pydantic` (net-zero dependency count change).

- **Trade-offs:** Best-in-class type, finiteness, range, discriminated-union, and cross-field validation; native `ty` typing; built-in JSON Schema export; no YAML typing traps; minimal dependencies.
  Lacks OmegaConf-style `${...}` interpolation, which is unnecessary here.

### 2. OmegaConf (YAML) + post-merge validation

Define `@dataclass` structured configs in `struct` mode and merge YAML layers and dotlist CLI overrides with `OmegaConf.merge`.

- **Trade-offs:** Strong hierarchical merging and interpolation, and maintained under `hydra-ecosystem` (v2.4 supports Python 3.13).
  However, OmegaConf does not enforce value ranges, finiteness, or cross-field invariants without converting to Pydantic or custom post-merge validators anyway; `DictConfig` weakens `ty` typing; adds `PyYAML` + `antlr4-python3-runtime`; and PyYAML's YAML 1.1 parser turns unquoted `off` into `False`.

### 3. Stdlib-only (`dataclasses` + `tomllib` + `argparse`)

Use frozen dataclasses with manual `__post_init__` checks, `tomllib`, and `argparse`.

- **Trade-offs:** Zero third-party config dependencies, but requires hand-writing strict type enforcement, unknown-key checks, nested dict deserialization, multi-error reporting, and JSON Schema generation.

**Selection:** The user chose Approach 1, prioritizing robustness, safety, and efficiency.

## Detailed Design

### Module Layout and Dependencies

- **`pyproject.toml`:** Remove `configargparse>=1.4` and add `pydantic>=2.10` to runtime dependencies.
  Tighten typing where possible; retain `ANN401` ignore only if still required by non-config `**kwargs` boundaries.
- **`falcon/config.py`:** Pure configuration schema, validation, hierarchical TOML loading, deep-merging, TOML serialization, JSON Schema generation, and staging-cache manifest/fingerprinting.
  No `argparse` or `sys.exit` coupling.
- **`falcon/cli.py`:** Command-line argument parsing (`argparse`), CLI-to-config-dict translation, human-readable validation error formatting, and CLI-only actions (`--version`, `--print-config`, `--save-config`, `--dump-schema`).
- **`falcon/falcon.py`:** Adds public `run(config: FalconConfig) -> int` and keeps `main(args: Sequence[str] | None = None) -> int` as the CLI entry point calling `falcon.cli` and `run`.
  `main` owns CLI logging setup (`configure_logger`) and teardown (`logging.shutdown`); `run` leaves global logging untouched so programmatic callers and tests are not disrupted.
  Eliminates all references to a global `config` object.
- **`falcon/spectra_io.py`:** Accepts explicit `config` / `work_dir` / `input_filenames` arguments instead of importing global `config`.
- **`falcon/__init__.py`:** Re-exports `FalconConfig`, `FalconSettings`, `load_config`, `load_settings`, and `run` alongside `__version__` for clean programmatic use.
- **`configs/metabolomics.toml`:** Repository-level example sparse TOML override file illustrating metabolomics settings from the v2 manuscript (referenced in `README.md`, not installed as package data or hardcoded into the CLI).

### Configuration Schema (`falcon/config.py`)

All models use `ConfigDict(frozen=True, extra="forbid", strict=True)` so instances are immutable, reject unknown keys, and forbid silent type coercion (for example, `bool` is rejected for `int`/`float`, `float` is rejected for `int`, and strings are rejected for numbers/booleans).
Because Pydantic v2 `strict=True` permits `int` values for `float` fields (PEP 484 numeric tower), writing `min_mz = 0` in TOML works cleanly while `min_mz = true` or `min_mz = "0"` fails.

#### Shared Types and Nullable TOML Sentinels

TOML has no `null` literal.
To allow a child TOML file or CLI flag to explicitly reset a nullable field inherited from a parent config back to `None`, nullable fields accept both Python `None` and an explicit string sentinel via a dedicated validator (which also updates the JSON Schema so TOML editors accept the sentinel):

- Nullable numeric/path fields (`work_dir`, `rt_tol`, `remove_precursor_tol`, `max_peaks_used`): `"none"` normalizes to `None`.
- Nullable charge buckets (`precursor_charge_buckets`): `"auto"` normalizes to `None` (avoiding confusion with "empty / drop all charges").
- After validation, Python attributes always hold `None` or the typed value—sentinels never leak into application code.
- `ConfigPath`: accepts `Path` or non-empty, non-whitespace `str` and converts to `Path`, rejecting booleans, numbers, and empty strings.
  For `work_dir`, `"none"` normalizes to `None` (a literal local directory named `none` can be passed as `./none`).

#### `PartialIOConfig` and `IOConfig` (`[io]`)

| Field | Python Type (`IOConfig`) | Default | Constraints |
| --- | --- | --- | --- |
| `input_filenames` | `tuple[str, ...]` | *(required in `IOConfig`; `None` in `PartialIOConfig`)* | Non-empty sequence (`min_length=1`); each element is a non-empty, non-whitespace `str`. |
| `output_filename` | `Path` | *(required in `IOConfig`; `None` in `PartialIOConfig`)* | Non-empty `Path` with non-empty `.name` (rejects bare `.` or `/`). Treated as an output prefix: CSV/MGF paths are built as `Path(f"{output_filename}.{suffix}")` rather than `with_suffix` so prefixes containing dots (such as `run_v1.2`) are preserved intact. |
| `work_dir` | `Path \| None` | `None` | When not `None`, valid non-empty `Path`. |
| `overwrite` | `bool` | `False` | Strict `bool`. |
| `export_representatives` | `bool` | `False` | Strict `bool`. |

*Model hierarchy (design call):*

- `FalconSettings(BaseModel)` holds `io: PartialIOConfig = Field(default_factory=PartialIOConfig)`, `preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)`, `clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)`, and `consensus: ConsensusConfig = Field(default_factory=MedoidConsensusConfig)`, along with all cross-section validators, `to_toml()`, and `to_metadata_lines()`.
  `load_settings(config_files=..., overrides=...) -> FalconSettings` validates reusable or partial configurations (used by `--print-config` and usable for validating overlay files like `metabolomics.toml`).
- `FalconConfig(FalconSettings)` narrows `io: IOConfig` so `input_filenames` and `output_filename` are guaranteed non-`None` at both static (`ty`) and runtime boundaries.
  `load_config(config_files=..., overrides=...) -> FalconConfig` validates complete runnable configurations for `run(config)`.
  Both classes share the exact same sub-models and cross-section validators, avoiding validation drift.

#### `PreprocessingConfig` (`[preprocessing]`)

Defaults preserve Falcon's bottom-up proteomics baseline:

| Field | Python Type | Default | Constraints |
| --- | --- | --- | --- |
| `min_peaks` | `int` | `5` | `ge=1` |
| `min_mz_range` | `float` | `250.0` | `ge=0.0`, finite |
| `min_mz` | `float` | `101.0` | `ge=0.0`, finite |
| `max_mz` | `float` | `1500.0` | `gt=0.0`, finite |
| `remove_precursor_tol` | `float \| None` | `1.5` | `gt=0.0`, finite when not `None` (`"none"` disables precursor peak removal) |
| `min_intensity` | `float` | `0.01` | `ge=0.0, lt=1.0`, finite |
| `max_peaks_used` | `int \| None` | `50` | `ge=1` when not `None` (`"none"` keeps all peaks) |
| `scaling` | `Literal["off", "root", "log", "rank"]` | `"off"` | Strict literal |

Cross-field model validators on `PreprocessingConfig`:

1. `min_mz < max_mz`.
2. `min_mz_range <= max_mz - min_mz` (otherwise no spectrum can survive `set_mz_range` + `_check_spectrum_valid`).
3. If `max_peaks_used is not None`: `max_peaks_used >= min_peaks` (otherwise every spectrum fails quality gates after intensity filtering).

#### `PrecursorTolerance` (`[clustering.precursor_tol]`)

| Field | Python Type | Default | Constraints |
| --- | --- | --- | --- |
| `mass` | `float` | `20.0` | `ge=0.0`, finite |
| `mode` | `Literal["ppm", "Da"]` | `"ppm"` | Case-sensitive strict literal |

#### `ClusteringConfig` (`[clustering]`)

`ChargeBucket = tuple[int | Literal["unknown"], ...] | Literal["other"]`

| Field | Python Type | Default | Constraints |
| --- | --- | --- | --- |
| `precursor_tol` | `PrecursorTolerance` | `PrecursorTolerance(mass=20.0, mode="ppm")` | Nested validated model |
| `rt_tol` | `float \| None` | `None` | `ge=0.0`, finite when not `None` |
| `fragment_tol` | `float` | `0.05` | `gt=0.0`, finite |
| `linkage` | `Literal["single", "complete", "average"]` | `"complete"` | Strict literal |
| `distance_threshold` | `float` | `0.1` | `gt=0.0, lt=1.0`, finite (prevents `1.0`, which merges orthogonal/unmatched pairs via inclusive `fcluster`) |
| `min_matched_peaks` | `int` | `0` | `ge=0` |
| `batch_size` | `int` | `32768` | `ge=1` |
| `precursor_charge_buckets` | `tuple[ChargeBucket, ...] \| None` | `None` | See charge-bucket rules below; `None` (or `"auto"`) clusters each observed charge separately |

Charge-bucket validation rules (enforced uniformly for Python, TOML, and CLI):

- When not `None`, outer sequence must be non-empty (`len >= 1`), rejecting `[]` (which would silently drop all spectra).
- Each element is either the literal `"other"` or a non-empty sequence/set of elements, each either a strict `int` (explicitly rejecting `bool`) in Arrow `int8` range `[-128, 127]` or the literal `"unknown"`.
- `"other"` may appear at most once and never inside a bucket list.
- No charge value or `"unknown"` may appear twice within the same bucket or across different buckets (rejects copy-paste duplicates and overlaps).
- Each non-`"other"` bucket is canonicalized to a sorted tuple with integer charges ascending followed by `"unknown"` (matching `spectra_io.py` naming and guaranteeing deterministic hashing/serialization).

#### `ConsensusConfig` (`[consensus]`)

Discriminated union on `method`:

- `MedoidConsensusConfig`: `method: Literal["medoid"] = "medoid"`.
- `AverageConsensusConfig`:
  - `method: Literal["average"] = "average"`.
  - `outlier_cutoff_lower: float = 1.5` (`gt=0.0`, finite).
  - `outlier_cutoff_upper: float = 1.5` (`gt=0.0`, finite).

Runtime warning preserved in `falcon.py`: when `isinstance(config.consensus, AverageConsensusConfig)` and both cutoffs `< 1.0`, emit a warning as before.

#### Cross-Section Validators (`FalconSettings` / `FalconConfig`)

1. If `preprocessing.max_peaks_used is not None` and `clustering.min_matched_peaks > preprocessing.max_peaks_used`: fail validation (pairs can never match more peaks than retained).
2. `math.isfinite((preprocessing.max_mz - preprocessing.min_mz) / clustering.fragment_tol)` and resulting `get_dim` bin count for averaging (`bin_size = 2 * fragment_tol`) fits safely within int32 bounds (`<= 2**31 - 1`), preventing subnormal-float division overflow or Numba `int32` index overflow.

### Hierarchical Layering and Merge Semantics

Configuration layers merge in deterministic order from lowest to highest precedence:

1. **Built-in model defaults.**
2. **Explicit config files** in order (`-c file1.toml -c file2.toml` or `load_config(config_files=[...])`).
   No config file is ever auto-loaded from the current working directory.
3. **CLI overrides** (or explicit Python `overrides` dict in `load_config`).

#### `extends` Inheritance in TOML Files

Any TOML file may specify a top-level `extends` key:

- `extends = "base.toml"` or `extends = ["base.toml", "instrument.toml"]`.
- Must be a non-empty `str` or non-empty `list[str]` of non-empty strings (strictly validated; invalid types raise `ConfigError`).
- Relative paths resolve against the declaring file's parent directory (`file_path.parent`), not CWD.
- Parents are loaded and merged left-to-right, then the declaring file's own keys (minus `extends`) merge on top.
- Cycle detection tracks the active stack of resolved canonical paths (`path.resolve()`) and raises `ConfigError` reporting the exact cycle (`a.toml -> b.toml -> a.toml`).
- Missing files or TOML syntax errors raise `ConfigError` naming the file (and parent file when reached via `extends`).

#### Deep-Merge Rules

`merge_config_dicts(base, override)` returns a fresh deep-copied dict without mutating inputs:

- **Discriminated union handling (`consensus`):** When merging the `consensus` mapping, if `override` contains `"method"` and `base` contains a different `"method"`, `override`'s `consensus` dict **replaces** `base`'s `consensus` dict.
  This ensures switching from `average` (in a parent file) to `medoid` (in a child file or via `--consensus-method medoid`) drops stale `outlier_cutoff_*` keys from lower layers, while setting `method = "medoid"` alongside `outlier_cutoff_lower` in the *same* layer (or passing `--outlier-cutoff-lower` against default `medoid`) still fails validation.
- **Nested tables:** When both `base[k]` and `override[k]` are dicts (such as `preprocessing`, `clustering`, `clustering.precursor_tol`, `io`), merge recursively so partial overrides keep sibling keys intact.
- **Lists and scalars:** Replace atomically (lists like `input_filenames` or `precursor_charge_buckets` are never concatenated across layers).

### CLI Design (`falcon/cli.py`)

Uses stdlib `argparse` with `default=argparse.SUPPRESS` on every configurable option so omitted CLI flags never enter the override dict and never clobber config-file layers.

#### Positional and I/O Arguments

- `paths` (`nargs="*"`):
  - `0` arguments: rely on `[io]` in config files (valid for `falcon -c run.toml`, `--print-config`, `--dump-schema`).
  - `1` argument: fail fast with a clear usage error (ambiguous input vs. output).
  - `>= 2` arguments: `paths[:-1]` → `io.input_filenames`, `paths[-1]` → `io.output_filename`.
- Explicit named alternatives: `-i` / `--input` (`nargs="+"`) and `-o` / `--output`, allowing users to override only inputs or only output prefix on top of a `run.toml` (for example, `falcon -c run.toml -o run_v2`).
  Mixing positional `paths` with `-i` or `-o` is rejected as ambiguous.
- `--work-dir PATH|none`
- `--overwrite` / `--no-overwrite` (`argparse.BooleanOptionalAction`)
- `--export-representatives` / `--no-export-representatives` (`argparse.BooleanOptionalAction`)

#### Configuration and Inspection Flags

- `-c` / `--config PATH` (`action="append"`): Repeatable TOML config file(s), applied left-to-right.
- `--print-config`: Resolve and validate via `load_settings` (inputs/output optional), print canonical resolved TOML to stdout, and exit `0` without running clustering.
- `--save-config PATH`: During a normal run, validate `FalconConfig` and write the resolved TOML to `PATH` before starting pipeline execution (rejected if `PATH` is an existing directory).
  (Users who want to materialize a merged config without executing a run can redirect `--print-config`.)
- `--dump-schema`: Print JSON Schema for TOML config files to stdout as formatted JSON and exit `0`.
  Built from `FalconSettings.model_json_schema()` with an optional top-level `extends` property (`string | array[string]`) and nullable TOML sentinels (`"none"` / `"auto"`) so editor tooling (such as Taplo / VS Code) validates sparse hierarchical override files without false errors.
- `--version`: Print `falcon <version>` and exit `0`.

#### Scientific Option Flags (Kebab-Case, Grouped by Section in `--help`)

- **Preprocessing:** `--min-peaks`, `--min-mz-range`, `--min-mz`, `--max-mz`, `--remove-precursor-tol` (float or `none`), `--min-intensity`, `--max-peaks-used` (int or `none`), `--scaling` (`off`, `root`, `log`, `rank`).
- **Clustering:** `--precursor-tol MASS MODE` (2 tokens; sets both mass and unit explicitly to prevent silent unit mismatches), `--rt-tol` (float or `none`), `--fragment-tol`, `--linkage` (`single`, `complete`, `average`), `--distance-threshold`, `--min-matched-peaks`, `--batch-size`, `--precursor-charge-buckets` (`nargs="+"`, accepting `auto`, `other`, or comma-separated charge specs with optional brackets like `1`, `2,3`, `[2, 3]`, `unknown`; rejects empty tokens between commas like `2,,3`).
- **Consensus:** `--consensus-method` (`medoid`, `average`), `--outlier-cutoff-lower`, `--outlier-cutoff-upper`.

#### CLI Error Handling

`ConfigError` and Pydantic `ValidationError` are formatted cleanly to `stderr` (listing each invalid dotted config path and reason) and exit with status `2`, rather than printing raw tracebacks.

### Serialization, Provenance, and Staging Cache Safety

#### Deterministic TOML Serialization (`to_toml`)

Implemented in `falcon/config.py` without extra dependencies (~50 lines for Falcon's bounded schema):

- Emits clean, deterministic TOML sections (`[io]`, `[preprocessing]`, `[clustering]`, `[clustering.precursor_tol]`, `[consensus]`) that round-trip cleanly through `tomllib.loads` and `load_config`.
- Comments disabled/unset optional fields (for example, `# rt_tol = "none"` or `# precursor_charge_buckets = "auto"`) so exported configs are self-documenting and directly replayable.

#### Logging and CSV Header Comments (Single Source of Truth)

`FalconConfig.to_metadata_lines(work_dir: Path) -> list[str]` generates canonical `section.key = value` lines directly from the validated model (using the resolved `work_dir` when a temporary directory was created).
Both `_log_config` in `falcon/falcon.py` and `_write_cluster_info` in `falcon/spectra_io.py` consume this single helper, guaranteeing debug logs and CSV `#` headers stay permanently synchronized with the schema.

#### Safe Staging Cache Validation (`spectra/staging_cache.json`)

Replaces `spectra/charges.joblib` with a safe JSON manifest written alongside the Lance datasets in `<work_dir>/spectra/staging_cache.json`:

```json
{
  "schema_version": 1,
  "falcon_version": "...",
  "inputs": [
    {"path": "/abs/path/a.mgf", "size": 1234, "mtime_ns": 1700000000000000000}
  ],
  "preprocessing": { "...": "..." },
  "fragment_tol": 0.05,
  "precursor_charge_buckets": null,
  "buckets": [[2], [3]]
}
```

- Input file patterns are expanded once in `_load_or_prepare_buckets` (sorting `glob.glob(pattern)` lexicographically per pattern for deterministic order and raising `FileNotFoundError` when no files match) and passed directly to `_prepare_spectra` so cache validation and file reading operate on the exact same resolved file list.
- Each resolved input entry in `inputs` records `path` (`str(Path(fn).resolve())`), `size`, and `mtime_ns` via `os.stat`, preserving repeated matches if supplied so changes to the input list are caught.
- When `staging_cache.json` exists and `not config.io.overwrite`:
  - Compare `schema_version`, `inputs`, `preprocessing`, `fragment_tol`, and `precursor_charge_buckets`.
  - On mismatch (or corrupted/invalid JSON cache file), raise `ValueError` naming the specific mismatched sections (for example, `preprocessing`, `inputs`) and instructing the user to rerun with `--overwrite` or a different `--work-dir`.
  - When all staging-affecting fields match, deserialize `buckets` back to `list[ChargeBucket]` (converting JSON lists to tuples and keeping `"other"` as `"other"`) and reuse the staged Lance datasets even if clustering (`distance_threshold`, `linkage`, `rt_tol`, `min_matched_peaks`, `batch_size`), consensus, or output parameters changed.
- When `staging_cache.json` is absent and `not config.io.overwrite`:
  - If `<work_dir>/spectra/` already contains `.lance` datasets or a legacy `charges.joblib`, raise `ValueError` explaining that the working directory holds an unverified or legacy staging cache and advising `--overwrite` or a fresh `--work-dir` (preventing silent appends onto stale datasets).
- Under `--overwrite`, clear `<work_dir>/spectra/` as before, restage, and write the fresh `staging_cache.json`.

### Reference Metabolomics Override Config

Ship `configs/metabolomics.toml` as an example sparse override file reflecting the v2 manuscript's metabolomics configuration:

```toml
[preprocessing]
min_mz = 0.0
max_mz = 1000.0
min_mz_range = 150.0

[clustering]
min_matched_peaks = 6
```

Users can pass `-c configs/metabolomics.toml`, extend it via `extends = "..."`, or layer additional `-c` files and CLI flags on top.

## What Is Left to the Builder

- Exact helper decomposition and internal function names inside `falcon/config.py` and `falcon/cli.py`.
- Exact wording of CLI help strings and formatted validation messages (as long as dotted field paths and invalid values/reasons are clearly shown).
- Updating `tests/` (`test_config.py`, new `test_cli.py`, `test_falcon.py`, `test_spectra_io.py`, `test_cli_end_to_end.py`, `conftest.py` removing `reset_global_config`) to maintain `>= 95%` branch coverage with thorough boundary, cross-field, layering, `extends`, cache-invalidation, and round-trip tests.
- Updating `README.md` and `AGENTS.md` to document the new configuration model, CLI flags, TOML format, hierarchical layering, and cache provenance.
