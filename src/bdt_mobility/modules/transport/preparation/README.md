# Transport — Data Preparation

Infrequent data-preparation steps for the transport module (spira, gate, PT
cost/fare). The package is installed (editable) via `uv sync`, so no
`PYTHONPATH` manipulation is needed — run the steps as modules.

Run all steps:
```bash
uv run python run_preparation.py
```

Run individual steps:
```bash
uv run python -m bdt_mobility.modules.transport.preparation.cost_fare.step_01_prepare_pts_cost_fare

uv run python -m bdt_mobility.modules.transport.preparation.gates.step_01_convert_gates
uv run python -m bdt_mobility.modules.transport.preparation.gates.step_02_merge_gate_files

uv run python -m bdt_mobility.modules.transport.preparation.spira.step_01_convert_spira
```
