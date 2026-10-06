from __future__ import annotations

import numpy as np
from pathlib import Path

from civic_digital_twins.dt_model import Index, Model, inputs, outputs, expose, config, define
from civic_digital_twins.dt_model.engine.frontend import graph

from .cases import GenderKey, resolve_case_slice
from .core.vulnerability import compute_vulnerability_by_area
from .core.builders import compute_indicator_deltas, compute_mpi_by_area, compute_mpi_deltas
from .loading.loaders import CaseLoader
from .projection import ProjectionMap, project_index_map

# NOTE
# right now the ticket costs is commented out since the digital twin library can not take None as input (only indexes)
# solution: create a runtime builder for the inputs, so to filter out any "None" input and keep the model generalized

# NOTE:
# EthicalModel currently assumes the dashboard will not change the min-max of the sliders for "Parametri comportamentali"
# this will be fixed in the future, right now that param can not be passed to MobilityModel since its created after
# the MobilityModel instantion
SECONDS_PER_HOUR = 3600.0
MAX_TIME_SHIFT_HOURS = 6.0
MAX_AVAILABILITY_HOURS = MAX_TIME_SHIFT_HOURS


@define("Ethical")
class EthicalModel(Model):

    @inputs
    class Inputs:
        ztl_start: Index
        ztl_end: Index
        avail_postpone: Index | None = None
        avail_anticipate: Index | None = None
        median_postpone: Index | None = None
        median_anticipate: Index | None = None
        ticket_costs: Index | None = None
        euro_class_split: Index | None = None

    @outputs
    class Outputs:
        baseline_mpi: dict[int, Index]
        modified_mpi: dict[int, Index]
        delta_mpi: dict[int, Index]

    @expose
    class Expose:
        # can be expanded
        vulnerability_by_area: dict[int, Index]

    @config
    class Config:
        case_id: str = "elderly"
        gender: GenderKey = "None"
        year: int = 2023
        base_dir: Path | None = None
        projection: ProjectionMap | None = None # projection map at construction time
        k_scale: float = 1.0


    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> tuple[Outputs, Expose]:

        # filenames, MPI names, policy class
        slice_spec = resolve_case_slice(config.case_id, config.gender)
        case_spec = slice_spec.case_spec
        self.slice_spec = slice_spec
        self.slice_id = slice_spec.slice_id
        self.case_spec = case_spec
        self._base_dir = config.base_dir
        self.projection = config.projection # model know if owns a projection
        self.output_geography = "zone" if config.projection is not None else "area_statistica" # model set the projection
        # actual data (as Index objects)
        eth_inputs = CaseLoader(slice_spec, base_dir=config.base_dir).load()
        
        vulnerability_by_area = compute_vulnerability_by_area(eth_inputs.indicators, eth_inputs.vulnerability_weights)
        baseline_mpi_data = compute_mpi_by_area(eth_inputs.indicators, eth_inputs.mpi_weights, mpi_name=slice_spec.baseline_mpi_name, prefix="")
        
        # sanity check
        policy_params = {key: value for key, value in {
            "ztl_start": _seconds_to_hours(inputs.ztl_start, name="I_E_ztl_start_hours"),
            "ztl_end": _seconds_to_hours(inputs.ztl_end, name="I_E_ztl_end_hours"),
            "avail_postpone": _normalize_availability_hours(inputs.avail_postpone, name="I_E_avail_postpone_norm"),
            "avail_anticipate": _normalize_availability_hours(inputs.avail_anticipate, name="I_E_avail_anticipate_norm"),
            "median_postpone": inputs.median_postpone,
            "median_anticipate": inputs.median_anticipate,
        }.items() if value is not None}
        if inputs.ticket_costs is not None:
            policy_params["ticket_costs"] = inputs.ticket_costs
        if inputs.euro_class_split is not None:
            policy_params["euro_class_split"] = inputs.euro_class_split
        case_spec.policy.validate(policy_params)

        # modified indicators
        modified_by_area = case_spec.policy.apply(
            eth_inputs.indicators,
            policy_params,
            eth_inputs.sensibility_weights,
            accept_zone_by_area=eth_inputs.accept_zone_by_area,
            vulnerability_by_area=vulnerability_by_area,
            exempt_share_by_area=eth_inputs.exempt_share_by_area,
            k_scale=config.k_scale,
        )

        # modified MPI
        modified_mpi_data = compute_mpi_by_area(
            modified_by_area,
            eth_inputs.mpi_weights,
            mpi_name=slice_spec.modified_mpi_name,
        )

        # NOTE: indicator deltas are not consumed right now. Still, availability is one comment away.
        # delta_indicators_by_area = compute_indicator_deltas(eth_inputs.indicators, modified_by_area)
        delta_mpi_data = compute_mpi_deltas(
            baseline_mpi_data,
            modified_mpi_data,
            baseline_mpi_name=slice_spec.baseline_mpi_name,
            modified_mpi_name=slice_spec.modified_mpi_name,
            delta_mpi_name=slice_spec.delta_mpi_name,
        )

        # dicts for a specific year
        baseline_mpi_selected = _project_mpi(baseline_mpi_data, slice_spec.baseline_mpi_name, config.year)
        modified_mpi_selected = _project_mpi(modified_mpi_data, slice_spec.modified_mpi_name, config.year)
        delta_mpi_selected = _project_mpi(delta_mpi_data, slice_spec.delta_mpi_name, config.year)

        # now project onto the desired output
        baseline_mpi_out = _project_output_map(baseline_mpi_selected, projection=config.projection, index_name=slice_spec.baseline_mpi_name)
        modified_mpi_out = _project_output_map(modified_mpi_selected, projection=config.projection, index_name=slice_spec.modified_mpi_name)
        delta_mpi_out = _project_output_map(delta_mpi_selected, projection=config.projection, index_name=slice_spec.delta_mpi_name)

        # exposed
        vulnerability_out: dict[int, Index] = {int(area): year_map[config.year]
            for area, year_map in vulnerability_by_area.items()
            if config.year in year_map
        }

        # baseline_ind_out: dict[int, dict[str, Index]] = {
        #     int(area): year_map[year]
        #     for area, year_map in eth_inputs.indicators.items()
        #     if year in year_map
        # }

        # modified_ind_out: dict[int, dict[str, Index]] = {
        #     int(area): year_map[year]
        #     for area, year_map in modified_by_area.items()
        #     if year in year_map
        # }

        # output handles
        self.I_baseline_mpi = baseline_mpi_out
        self.I_modified_mpi = modified_mpi_out
        self.I_delta_mpi = delta_mpi_out
        self.I_vulnerability_by_area = vulnerability_out
        # not exposed yet
        # self.I_baseline_indicators_by_area = baseline_ind_out
        # self.I_modified_indicators_by_area = modified_ind_out

        return (
            EthicalModel.Outputs(
                baseline_mpi=baseline_mpi_out,
                modified_mpi=modified_mpi_out,
                delta_mpi=delta_mpi_out,
            ),
            EthicalModel.Expose(
                vulnerability_by_area=vulnerability_out,
            )
        )


    # ------------------------------------ Model Methods ----------------------------------------------------

    def extract_output_maps(self, subs) -> dict[str, dict[int, float]]:
        return {
            "baseline_mpi": _extract_float_map(self.outputs.baseline_mpi, subs),
            "modified_mpi": _extract_float_map(self.outputs.modified_mpi, subs),
            "delta_mpi": _extract_float_map(self.outputs.delta_mpi, subs),
        }


# bring Index to hours
def _seconds_to_hours(value: Index | None, *, name: str) -> Index | None:
    if value is None:
        return None
    return Index(name, graph.ensure_node(value) / graph.constant(SECONDS_PER_HOUR))

# normalize Index objects (add nodes to the graph)
def _normalize_availability_hours(value: Index | None, *, name: str) -> Index | None:
    if value is None:
        return None
    normalized = graph.ensure_node(value) / graph.constant(MAX_AVAILABILITY_HOURS)
    clipped = graph.where(normalized < graph.constant(0.0),graph.constant(0.0),
        graph.where(normalized > graph.constant(1.0),graph.constant(1.0),normalized,),)
    return Index(name, clipped)

# return selected data window
def _project_mpi(mpi_by_area: dict, mpi_name: str, year: int) -> dict[int, Index]:
    return {
        int(area): year_map[year][mpi_name]
        for area, year_map in mpi_by_area.items()
        if year in year_map and mpi_name in year_map[year]
    }

# return the selected data in the requested output geography
def _project_output_map(index_map: dict[int, Index], *, projection: ProjectionMap | None, index_name: str) -> dict[int, Index]:
    if projection is None:
        return index_map
    return project_index_map(index_map, projection, index_name=index_name)

# numerical outputs for numerical consumer (e.g. dashboard)
def _extract_float_map(index_map: dict[int, Index], subs) -> dict[int, float]:
    if hasattr(subs, "expected_value"):
        return {
            area: float(np.asarray(subs.expected_value(idx)).reshape(-1)[0])
            for area, idx in index_map.items()
        }
    return {
        area: float(np.asarray(subs[idx]).reshape(-1)[0])
        for area, idx in index_map.items()
        if idx in subs
    }
