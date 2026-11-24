# Contributing to Falcon

Thank you for contributing to Falcon (`falcon-ms`).
This guide covers local setup, quality checks, testing, and releasing.

## Prerequisites

- Python 3.11+ (Linux or macOS)
- [`uv`](https://docs.astral.sh/uv/) for environment, dependency, and build management

## Development Setup

Sync the project and all development dependency groups (`lint`, `test`, `audit`):

```bash
uv sync
```

Optionally install pre-commit hooks with [`prek`](https://github.com/j178/prek) (or `pre-commit`):

```bash
prek install
prek run --all-files
```

## Common Commands

All commands run inside the managed virtual environment via `uv run` (matching targets are also provided in `Makefile`):

```bash
uv run ruff format --check .                                                      # Check formatting
uv run ruff check .                                                               # Lint code
uv run ruff check --fix . && uv run ruff format .                                 # Auto-fix and format
uv run ty check falcon                                                            # Type check
uv run pytest                                                                     # Run test suite
uv run pytest --cov=falcon --cov-report=term-missing                             # Run tests with coverage
uv run pip-audit --local --skip-editable                                          # Audit dependencies
uv run zizmor --persona=regular --min-severity=medium --min-confidence=medium .github/workflows/  # Audit workflows
uv build                                                                          # Build sdist and wheel
```

## Coding Standards

- Follow the repository Ruff and `ty` configuration in `pyproject.toml`.
- Keep Python code within 79 characters per line and comments/docstrings within 72 characters.
- Use NumPy-style docstrings for public APIs.
- Manage dependencies through `uv add` / `uv remove` so `uv.lock` stays in sync.
- Preserve numerical correctness, spectrum ordering, and output formats; add or update `pytest` tests for behavioral changes.
- Update `README.md` or `AGENTS.md` when changing user-facing options, dependencies, architecture, or workflows.

## Releasing

Versions are derived from Git tags via `setuptools-scm` (for example, `v0.2.0`).
Publishing to PyPI is automated by `.github/workflows/publish.yml` when a GitHub Release is published.

Before publishing the first Trusted Publishing release, configure a PyPI Trusted Publisher for `falcon-ms`:

- **Owner:** `bittremieuxlab`
- **Repository:** `falcon`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

Also create the matching `pypi` environment in GitHub repository settings (with optional deployment branch/tag protection rules).
