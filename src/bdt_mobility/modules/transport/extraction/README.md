# Transport — Data Extraction

These are the transport-module extraction steps. The package is installed
(editable) via `uv sync`, so no `PYTHONPATH` manipulation is needed — steps run
as modules from anywhere in the repo.

## Run the whole pipeline

From the repository root, `run_extraction.py` runs every extraction step in
order and stops on the first failure:
```bash
uv run python run_extraction.py
```

## Run individual steps

Each step is a module with a `main()` entry point. Run with `python -m`:
```bash
uv run python -m bdt_mobility.extraction.step_00_get_input_data
uv run python -m bdt_mobility.modules.transport.extraction.step_01_process_road_network
uv run python -m bdt_mobility.modules.transport.extraction.step_02_agg_in_out_flows_from_od
uv run python -m bdt_mobility.modules.transport.extraction.step_03_inflow_traffic
uv run python -m bdt_mobility.modules.transport.extraction.step_04_spira_outflow_estimation
uv run python -m bdt_mobility.modules.transport.extraction.step_05_spira_traffic_estimation
uv run python -m bdt_mobility.modules.transport.extraction.step_06_traffic_model
uv run python -m bdt_mobility.modules.transport.extraction.step_07_get_tpm_metrics
uv run python -m bdt_mobility.modules.transport.extraction.step_08_emissions_preprocess
```

Then the `od_routing` sub-pipeline:
```bash
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_01_preprocessing_roads
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_02_preprocessing_platoons
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_03_simulation
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_04_estimate_zonal_flows
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_05_estimate_edge_metrics
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_06_estimate_trip_metrics
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_07_estimate_non_linear_fit_parameters
uv run python -m bdt_mobility.modules.transport.extraction.od_routing.step_08_estimate_emissions
```

> The fragilities step belongs to the ethics module:
> `bdt_mobility.modules.ethics.extraction.step_01_fragilities_preprocess`.
> `run_extraction.py` already includes it in the full sequence.
