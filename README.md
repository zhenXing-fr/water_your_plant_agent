# water_your_plant_agent

Agent project for watering my own garden.

## Tooling setup (uv + pre-commit)

This project is configured with:
- Runtime dependency: `pydantic`
- Dev dependencies: `ruff`, `pytest`, `pre-commit`
- Pre-commit hooks for linting/formatting and basic checks

### 1. Install uv

On macOS:

```bash
brew install uv
```

Or with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual env + install dependencies

From the repository root:

```bash
uv sync --group dev
```

This creates `.venv` and installs both runtime + dev dependencies.

### 3. Run tooling once manually

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

### 4. Install pre-commit hooks

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### 5. Run hooks on all files (first-time check)

```bash
uv run pre-commit run --all-files
```

## Daily usage

- Run tests: `uv run pytest`
- Run lint: `uv run ruff check .`
- Run format: `uv run ruff format .`

## Files added

- `pyproject.toml`: project metadata + dependency/tool configuration
- `.pre-commit-config.yaml`: git hook definitions
- `.gitignore`: common Python/uv ignores
- `tests/test_smoke.py`: minimal pytest smoke test
