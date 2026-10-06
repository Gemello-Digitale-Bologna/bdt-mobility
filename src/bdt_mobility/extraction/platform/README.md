# Extraction — Platform Execution

DigitalHub assets to register, build, and run the BDT mobility data-extraction
pipeline on the platform, plus a local entry point for the same step sequence.

The pipeline runs transport extraction steps (`step_00`–`step_08`), the ethics
fragilities step, and the `od_routing` sub-pipeline (`step_01`–`step_08`). Each
step module exposes a `main()`; the ordered list is assembled from per-module
`STEPS` lists via `extraction.pipeline.PIPELINE` and exposed in `utils.py` as
`MODULES`.

## Files

| File | Purpose |
| --- | --- |
| `outputs.py` | Type definitions (`PlatformOutput`, `DynamicPlatformOutputs`) used in each step's `MAIN_OUTPUTS` declaration and read by the generator. |
| `utils.py` | `MODULES` (ordered step list), `exec_main()` (dynamic import + `main()`), `parse_labels()`. |
| `workflow_handlers.py` | **Auto-generated.** One handler per step; executes the module and logs that step's `MAIN_OUTPUTS` back to the project as artifacts/tables. Calls `_verify()` on import to abort if stale. |
| `workflow.py` | **Auto-generated.** Hera DAG (`data-ingestion-pipeline`) wiring the handlers in order. |
| `ingest_data.py` | **Auto-generated.** Registers per-step DigitalHub functions and triggers the workflow run. |
| `download_data.py` | Downloads raw inputs (via step_00) and all static `MAIN_OUTPUTS` from DigitalHub to local PathRegistry paths. |
| `req.txt` | Dependency snapshot passed to DigitalHub at registration time. |

The three auto-generated files are produced by `scripts/generate_platform.py`.
Re-run it whenever `PIPELINE` order changes or any step's `MAIN_OUTPUTS` changes:

```
uv run python scripts/generate_platform.py
```

The generator embeds a SHA-256 fingerprint of `(module_name, MAIN_OUTPUTS)` for every
step. `workflow_handlers.py` calls `_verify()` at import time — if the fingerprint
does not match the live `MAIN_OUTPUTS` declarations, the import aborts with a
`RuntimeError` pointing to the generator command.

## Execution modes

| Mode | Entry point | Use case |
| --- | --- | --- |
| Local | `run_extraction.py` | Run the full pipeline locally without DigitalHub orchestration. |
| Platform | `ingest_data.py` | Register per-step functions and run the Hera DAG on the platform. |
| Download | `download_data.py` | Pull all data to the local data directory: raw inputs (step_00) then all static `MAIN_OUTPUTS`. Dynamic outputs (seed-indexed files) must be downloaded manually from the DigitalHub UI. |

## Notes

- `code_src` in `ingest_data.py` must point at the branch or commit that is
  deployed to the platform; update it before running the registration script.
- `req.txt` is a pinned freeze of the resolved environment (`uv pip freeze`),
  more specific than the loose ranges in `pyproject.toml`; regenerate it before
  any deployment to keep the platform container in sync with the local environment.
- Some OD-routing steps return dynamic lists of table names; those handlers log
  each generated table individually.
- `MODULES` is derived from `extraction.pipeline.PIPELINE` — add steps there, not here.
