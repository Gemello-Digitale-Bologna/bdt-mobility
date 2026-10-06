# bdt-mobility

Data extraction and simulation toolkit for predicting the direct and indirect impacts of the different possible
policies for regulating the traffic in a Zone of Interest (ZOI) of the city of Bologna.

This package has been developed in the scope of the Civic Digital Twin / Bologna Digital Twin project.

See [docs/glossary.md](docs/glossary.md) for definitions of domain and codebase terms.

## Environment Setup

We use [uv](https://astral.sh/uv) for managing the development environment.

To get started, run:

```bash
git clone https://github.com/Gemello-Digitale-Bologna/bdt-mobility
cd bdt-mobility
uv venv
source .venv/bin/activate
uv sync --group dev
```

## Running the extraction pipeline

Run the full pipeline locally:

```bash
uv run python run_extraction.py
```

See `src/bdt_mobility/modules/transport/extraction/README.md` for running individual steps.

Alternatively, if the pipeline has already been executed on the platform, download all outputs
(raw inputs + step results) to the local data directory:

```bash
uv run python src/bdt_mobility/extraction/platform/download_data.py
```

For the moment, dynamic step outputs (seed-indexed files) are not downloaded automatically.

## Running the simulation

```bash
uv run python -m bdt_mobility.simulation.dashboard
```

The ethics module is disabled by default in the model and dashboard; enable it with
`"enable_ethics": True` in `config_params` (`src/bdt_mobility/model/config_vars.py`).
The ethics extraction steps always run as part of the extraction pipeline.

## Optional checks

### Tests

There are three categories of tests:

- **unit** — fast, no data required
- **simulation** — regression tests for the simulation model, require real data
- **integration** — end-to-end pipeline tests, require real data

Run all tests:

```bash
uv run pytest
```

Run a specific category:

```bash
uv run pytest -m unit
uv run pytest -m simulation
uv run pytest -m integration
```

Tests that require data fail explicitly when data is absent.

## Before merging to main

- [ ] Regenerate the platform dependency snapshot (strip the local editable entry). Sync the
  environment first, so that packages installed by hand and not in `uv.lock` are left out:
  ```bash
  uv sync --group dev
  uv pip freeze | grep -v "^-e file" > src/bdt_mobility/extraction/platform/req.txt
  ```
- [ ] Update `TAG_GITHUB` in `src/bdt_mobility/extraction/config.py`
  to the target branch or commit SHA.
- [ ] Verify the platform handler fingerprint is current:
  ```bash
  uv run python scripts/generate_platform.py
  ```
  The command should report the same fingerprint as embedded in `workflow_handlers.py`
  (i.e. produce no diff). If it differs, commit the regenerated files.

### Linting and formatting (ruff)

Check only — no files modified:

```bash
uv run ruff check .
uv run ruff format --check .
```

Check and auto-fix:

```bash
uv run ruff check --fix .
uv run ruff format .
```
