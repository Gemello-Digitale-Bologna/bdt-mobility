"""Modular BDT mobility model — v0.8.0 dataclass I/O API.

This module uses the
:class:`~civic_digital_twins.dt_model.Model` dataclass I/O contract and
:class:`~civic_digital_twins.dt_model.ModelVariant` for the four variant axes:

* ``DECISION_STRATEGY``  → :class:`ModelVariant` at :class:`BehaviorModel`
* ``MODAL_SHIFT_OPTION`` → :class:`ModelVariant` at :class:`ModalShiftModel`
  (owned inside :class:`BehaviorModel`)
* ``TIME_SHIFT_STRATEGY``→ :class:`ModelVariant` at :class:`ShiftFormula`
  (owned inside :class:`TimeShiftModel` and :class:`AreaModifiedFlowsModel`)
* ``POLICY``             → :class:`ModelVariant` at :class:`ModifiedFlowsModel`
  and :class:`RevenueModel`; also a plain ``str`` constructor arg to
  :class:`BehaviorModel` for the ``I_fraction_p_anticipating`` formula difference

``I_B_starting_modified_factor`` is intentionally omitted: under
``cordon-based`` policy ``modified_starting`` is simply ``ts_starting``
(identity), which is the degenerate case of the factor being zero.

Hierarchy
---------
::

    MobilityModel
    ├── [A] BaseFlowsModel
    ├── [B] BaseZonalFlowsModel
    ├── [C] PolicyWindowModel
    ├── [V.2] BehaviorModel                  ← ModelVariant(decision_strategy)
    │         └── [V.1] ModalShiftModel      ← ModelVariant(modal_shift_option)
    ├── [D] TimeShiftModel
    │         └── [V.3] ShiftFormula         ← ModelVariant(time_shift_strategy)
    ├── [V.4] ModifiedFlowsModel             ← ModelVariant(policy)
    │         └── [V.3] ShiftFormula (area)  ← ModelVariant(time_shift_strategy)
    ├── [V.5] RevenueModel                   ← ModelVariant(policy)
    ├── [E] TrafficModel
    ├── [F] ModifiedZonalFlowsModel
    ├── [G] EmissionsModel
    └── [H] EthicalModel
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from civic_digital_twins.dt_model import (
    DistributionEnsemble,
    Index, TimeseriesIndex, DistributionIndex, ConstTimeseriesIndex,
    Model, Scenario, ModelVariant, Evaluation, EvaluationResult,
    define, inputs, outputs, expose, config,
    NumpyBackend,
)
from bdt_mobility.modules.ethics.model import EthicalModelBundle, build_projection_map_from_zone_ids

from bdt_mobility.utils import data_reader
from bdt_mobility.utils import constants
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils.config import logging, logging_level

from bdt_mobility.utils.paths import PathRegistry

from bdt_mobility.modules.transport.model import (
    ts_solve, ts_b_choose, ts_anticipate, ts_postpone,
    zone_flows_df_to_mat,
    BaseFlowsModel, BaseZonalFlowsModel,
    PolicyWindowModel,
    ParallelBehaviorModel, SequentialBehaviorModel,
    TimeShiftModel,
    CordonModifiedFlowsModel, AreaModifiedFlowsModel,
    AreaRevenueModel, CordonRevenueModel,
    TrafficModel, ModifiedZonalFlowsModel, EmissionsModel,
)

from bdt_mobility.model.config_vars import behavioral_params, policy_params, config_params
from bdt_mobility.model.axis import (
    EclIndex,
    ZonePtIndex,
    ZoneInTimeseriesIndex,
)


def _ethical_expose_fields(bundle: EthicalModelBundle, case_id: str) -> dict[str, dict[int, Index]]:
    fields = {}
    for gender, gender_key in (("None", "none"), ("male", "male"), ("female", "female")):
        fields[f"ethical_{case_id}_{gender_key}_baseline_mpi"] = bundle.outputs.baseline_mpi[gender]
        fields[f"ethical_{case_id}_{gender_key}_modified_mpi"] = bundle.outputs.modified_mpi[gender]
        fields[f"ethical_{case_id}_{gender_key}_delta_mpi"] = bundle.outputs.delta_mpi[gender]
        fields[f"ethical_{case_id}_{gender_key}_vbya"] = bundle.expose.vulnerability_by_area[gender]
    return fields

# ---------------------------------------------------------------------------
# Root — MobilityModel
# ---------------------------------------------------------------------------


@define("MobilityModel")
class MobilityModel(Model):
    """Root model for the ZOI mobility scenario.

    Composes all sub-models and exposes the final KPI outputs.

    Parameters
    ----------
    policy :
        ``"cordon-based"`` (default) or ``"area-based"``.
    decision_strategy :
        ``"parallel"`` (default) or ``"sequential"``.
    time_shift_strategy :
        ``"flexible"`` (default) or ``"fixed"``.
    modal_shift_option :
        ``"no"``, ``"lpt"`` (default), ``"active"``, or ``"lpt+active"``.

    Sub-models and parameters are accessible as named attributes::

        m = MobilityModel()
        m.base_flows.outputs.ts_inflow
        m.behavior.outputs.fraction_rigid
        m.time_shift.outputs.number_time_shifted
        m.modified_flows.outputs.modified_flows
        m.zone_downscaling.outputs.modified_traffic_zone
        m.emissions.outputs.total_modified_emissions

    Policy parameters (``I_P_*``) and behavioural parameters (``I_B_*``) are
    aliased directly on the model instance::

        m.I_P_start_time
        m.I_P_end_time
        m.I_P_cost          # list[Index], one per euro class (dashboard view)
        m.I_P_fraction_exempted
        m.I_P_pt_frequency_modification
        m.I_P_pt_capillarity_modification
        m.I_P_pt_cost_modification
        m.I_P_pt_time_modification
        m.I_B_p50_cost
        m.I_B_p50_anticipating
        m.I_B_p50_postponing
        m.I_B_p50_anticipation
        m.I_B_p50_postponement
        m.I_B_pt_capillarity
        m.I_B_pt_frequency
        m.I_B_pt_cost
        m.I_B_pt_time
    """

    @inputs
    class Inputs:
        """Parameters and abstract indices as inputs"""

        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: EclIndex
        i_p_fraction_exempted: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index

        i_b_p50_cost: DistributionIndex
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        euro_class_emission: EclIndex
        euro_class_split: EclIndex
        pt_zone_proportions: ZonePtIndex
        pt_capillarity: ZonePtIndex
        pt_frequency: ZonePtIndex
        pt_avg_cost: ZonePtIndex
        pt_avg_time_diff: ZonePtIndex


    @outputs
    class Outputs:
        """KPI outputs of :class:`MobilityModel`."""

        total_base_inflow: Index
        total_modified_inflow: Index
        total_traffic: Index
        total_modified_traffic: Index
        total_mode_shifted: Index
        total_time_shifted: Index
        total_lost: Index
        total_paying: Index
        modified_avg_cost: Index
        total_paid: Index
        total_emissions: Index
        total_modified_emissions: Index

    @expose
    class Expose:
        """All sub-model indexes, collected so the engine reaches every node."""

        modified_inflow: TimeseriesIndex
        modified_inflow_zone: ZoneInTimeseriesIndex
        traffic: TimeseriesIndex
        traffic_zone: ZoneInTimeseriesIndex
        modified_traffic: TimeseriesIndex
        modified_traffic_zone: ZoneInTimeseriesIndex
        emissions: TimeseriesIndex
        emissions_zone: ZoneInTimeseriesIndex
        modified_emissions:  TimeseriesIndex
        modified_emissions_zone: ZoneInTimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        ts_inflow_zone_from_inside: ZoneInTimeseriesIndex
        ts_inflow_zone_from_outside: ZoneInTimeseriesIndex
        ts_inflow_zone: ZoneInTimeseriesIndex
        ts_traffic_zone: ZoneInTimeseriesIndex
        ethical_elderly_none_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_none_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_none_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_none_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_male_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_male_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_male_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_male_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_female_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_female_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_female_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_elderly_female_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_minors_none_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_none_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_none_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_none_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_minors_male_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_male_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_male_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_male_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_minors_female_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_female_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_female_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_minors_female_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_income_none_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_none_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_none_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_none_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_income_male_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_male_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_male_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_male_vbya: dict[int, Index] = field(default_factory=dict)
        ethical_income_female_baseline_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_female_modified_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_female_delta_mpi: dict[int, Index] = field(default_factory=dict)
        ethical_income_female_vbya: dict[int, Index] = field(default_factory=dict)

    @config
    class Config:
        policy: str
        decision_strategy: str
        time_shift_strategy: str
        modal_shift_option: str
        enable_ethics: bool

        all_zones: list[int]
        in_zones: list[int]
        out_zones: list[int]
        pt_zones: list[int]

        zone_ifi: np.ndarray
        zone_ifo: np.ndarray
        zone_t: np.ndarray
        zone_od_starting: pd.DataFrame
        pt_metrics_df: pd.DataFrame
        ethical_projection: dict | None

    @classmethod
    def default_config(
        cls,
        config_params: dict,
    ) -> MobilityModel.Config:

        # Spatial data
        gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)

        aree_gdf_inside_raw, aree_gdf_outside_raw = data_reader.AOI_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od", df_around=gdf_zoi, return_type="both"
        )

        aree_gdf_inside_raw["id_zone"] = aree_gdf_inside_raw["id_zone"].astype(int)
        aree_gdf_inside_raw["type_zoi"] = "inside"
        aree_gdf_outside_raw["id_zone"] = aree_gdf_outside_raw["id_zone"].astype(int)
        aree_gdf_outside_raw["type_zoi"] = "outside"
        aree_gdf_inside_raw = aree_gdf_inside_raw.to_crs(constants.CRS_LATLONG)
        aree_gdf_outside_raw = aree_gdf_outside_raw.to_crs(constants.CRS_LATLONG)
        aree_gdf = pd.concat([aree_gdf_inside_raw, aree_gdf_outside_raw], axis=0)
        aree_gdf = aree_gdf.reset_index(drop=True).reset_index().sort_values(by="index")
        aree_gdf_inside = aree_gdf[aree_gdf["type_zoi"] == "inside"]

        in_zones = np.array(aree_gdf.loc[aree_gdf["type_zoi"] == "inside", "index"].unique())
        out_zones = np.array(aree_gdf.loc[aree_gdf["type_zoi"] == "outside", "index"].unique())
        all_zones = np.concatenate([in_zones, out_zones])

        pt_zones = all_zones if config_params["policy"] == "area-based" else out_zones

        # Zonal vehicular TS flows
        zone_io_df = pd.read_parquet(PathRegistry.INFLOW_PER_ZONE_SIM_PROCESSED)
        assert set(zone_io_df["index"].unique()) == set(in_zones), \
            f"Mismatch found in indices: {set(zone_io_df['index'].unique()) ^ set(in_zones)}"
        zone_ifo = zone_flows_df_to_mat(zone_io_df, zones=in_zones, flow_col="inflow_from_outside_mean")
        zone_ifi = zone_flows_df_to_mat(zone_io_df, zones=in_zones, flow_col="inflow_from_inside_mean")

        zone_traffic_df = pd.read_parquet(PathRegistry.TRAFFIC_PER_ZONE_SIM_PROCESSED)
        assert set(zone_traffic_df["index"].unique()) == set(in_zones), \
            f"Mismatch found in indices: {set(zone_traffic_df['index'].unique()) ^ set(in_zones)}"
        zone_t = zone_flows_df_to_mat(zone_traffic_df, zones=in_zones, flow_col="traffic_in_zone_mean")

        # Zonal total outflows for zone proportions
        zone_od_starting = pd.read_parquet(PathRegistry.TOTAL_STARTS_FROM_OD)
        zone_od_starting = pd.merge(
            zone_od_starting,
            aree_gdf[["index", "id_zone", "type_zoi"]],
            left_on=["from", "type_zoi_from"],
            right_on=["id_zone", "type_zoi"],
            how="right",
        ).sort_values("index")
        assert set(zone_od_starting["index"].unique()) == set(all_zones), \
            f"Mismatch found in indices: {set(zone_od_starting['index'].unique()) ^ set(all_zones)}"

        pt_metrics_df = pd.read_parquet(PathRegistry.PT_METRICS_PER_ZONE)
        pt_metrics_df = pd.merge(
            pt_metrics_df,
            aree_gdf[["index", "type_zoi", "id_zone"]],
            left_on=["zone_id", "type_zoi"],
            right_on=["id_zone", "type_zoi"],
            how="left",
        ).sort_values("index")
        pt_metrics_df = pt_metrics_df.loc[pt_metrics_df["index"].isin(pt_zones)]
        assert set(pt_metrics_df["index"].unique()) == set(pt_zones), \
            f"Mismatch found in indices: {set(pt_metrics_df['index'].unique()) ^ set(pt_zones)}"

        ethical_projection = None
        if config_params["enable_ethics"]:
            crosswalk_df = pd.read_parquet(PathRegistry.MAP_AS_ZONEPUMS)
            crosswalk_df = crosswalk_df[crosswalk_df["id_zone"].isin(aree_gdf_inside["id_zone"].unique())].copy()
            ethical_projection = build_projection_map_from_zone_ids(crosswalk_df, aree_gdf_inside)

        return MobilityModel.Config(
            **config_params,
            all_zones=all_zones,
            in_zones=in_zones,
            out_zones=out_zones,
            pt_zones=pt_zones,
            zone_ifo=zone_ifo,
            zone_ifi=zone_ifi,
            zone_t=zone_t,
            zone_od_starting=zone_od_starting,
            pt_metrics_df=pt_metrics_df,
            ethical_projection=ethical_projection
        )

    @classmethod
    def default_inputs(
        cls,
        config: MobilityModel.Config,
        policy_params: dict,
        behavioral_params: dict,
    ) -> MobilityModel.Inputs:
        """Return the reference-scenario inputs as an :class:`~.Inputs` instance."""
    
        # Vehicular TS flows
        vehicle_inflow = pd.read_parquet(PathRegistry.INFLOW_ESTIMATED)["Weekday"].to_numpy()
        ts_inflow = TimeseriesIndex("base inflow", vehicle_inflow)
        vehicle_starting = pd.read_parquet(PathRegistry.STARTING_ESTIMATED)["Weekday"].to_numpy()
        ts_starting = TimeseriesIndex("base starting", vehicle_starting)

        # Euro class split
        euro_counts = pd.read_parquet(PathRegistry.EURO_CLASS_SPLIT_PARQUET).sort_values(by="euro_level")
        assert euro_counts["euro_level"].nunique() == 7, f"Expected 7 unique values in `euro_counts`, but got {euro_counts['euro_level'].nunique()}"
        euro_class_split = EclIndex(
            "split euro levels",
            np.array(euro_counts["prop"])
        )
        
        # Euro class emissions
        euro_emissions = pd.read_parquet(PathRegistry.EURO_CLASS_EMISSIONS_PARQUET).sort_values(by="euro_level")
        euro_emissions = euro_emissions.loc[euro_emissions["euro_level"].isin(euro_counts["euro_level"].unique())]
        assert euro_emissions["euro_level"].nunique() == 7, f"Expected 7 unique values in `euro_emissions`, but got {euro_emissions['euro_level'].nunique()}"
        euro_class_emission = EclIndex(
            "emissions euro levels",
            np.array(euro_emissions["emission"])
        )

        # Pt zone proportions
        _total = sum(config.zone_od_starting[config.zone_od_starting['index'].isin(config.pt_zones)]['flow'].values)
        pt_zone_proportions = ZonePtIndex(
            "proportion of pt zone in total inflow",
            np.array( config.zone_od_starting[config.zone_od_starting['index'].isin(config.pt_zones)]['flow'].values / _total)
        )

        # PT metrics — select the right zone set per policy
        pt_capillarity = ZonePtIndex(
            "capillarity of pt zones",
            np.array(config.pt_metrics_df["capillarity"])
        ) 
        pt_frequency = ZonePtIndex(
            "frequency of pt zones",
            np.array(config.pt_metrics_df["frequency"])
        ) 
        pt_avg_cost = ZonePtIndex(
            "average cost of pt zones",
            np.array(config.pt_metrics_df["cost_euro"])
        )
        pt_avg_time_diff = ZonePtIndex(
            "average time difference of pt zones",
            np.array(config.pt_metrics_df["duration_seconds_diff_mean"])/60
        )

        return MobilityModel.Inputs(
            **policy_params,
            **behavioral_params,
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            euro_class_split=euro_class_split,
            euro_class_emission=euro_class_emission,
            pt_zone_proportions=pt_zone_proportions,
            pt_frequency=pt_frequency,
            pt_capillarity=pt_capillarity,
            pt_avg_cost=pt_avg_cost,
            pt_avg_time_diff=pt_avg_time_diff,
        )

    def compute(
        self,
        inputs: Inputs,
        config: Config,
    ) -> tuple[Outputs, Expose]:

        # ---------------------------------------------------------------
        # Validate selectors
        # ---------------------------------------------------------------
        if config.policy not in ("cordon-based", "area-based"):
            raise ValueError(f"policy must be 'cordon-based' or 'area-based', got {config.policy!r}")
        if config.decision_strategy not in ("parallel", "sequential"):
            raise ValueError(f"decision_strategy must be 'parallel' or 'sequential', got {config.decision_strategy!r}")
        if config.time_shift_strategy not in ("flexible", "fixed"):
            raise ValueError(f"time_shift_strategy must be 'flexible' or 'fixed', got {config.time_shift_strategy!r}")
        if config.modal_shift_option not in ("no", "lpt", "active", "lpt+active"):
            raise ValueError(
                f"modal_shift_option must be 'no', 'lpt', 'active', or 'lpt+active', got {config.modal_shift_option!r}"
            )

        # ---------------------------------------------------------------
        # Shared timeseries index
        # ---------------------------------------------------------------
        ts = ConstTimeseriesIndex(
            "time range",
            np.array([
                (t - pd.Timestamp("00:00:00")).total_seconds()
                for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
            ]),
        )

        # ---------------------------------------------------------------
        # [A] BaseFlowsModel
        # ---------------------------------------------------------------
        base_flows = BaseFlowsModel(
            inputs=BaseFlowsModel.Inputs(
                ts=ts,
                ts_inflow=inputs.ts_inflow,
            )
        )

        # ---------------------------------------------------------------
        # [B] BaseZonalFlowsModel
        # ---------------------------------------------------------------
        base_zonal_flows = BaseZonalFlowsModel(
            inputs=BaseZonalFlowsModel.Inputs(ts=ts),
            config=BaseZonalFlowsModel.Config(
                zones_ifi_mat=config.zone_ifi,
                zones_ifo_mat=config.zone_ifo,
                zones_t_mat=config.zone_t,
            )
        )

        # ---------------------------------------------------------------
        # [C] PolicyWindowModel
        # ---------------------------------------------------------------
        policy_window = PolicyWindowModel(
            PolicyWindowModel.Inputs(
                ts=ts,
                i_p_start_time=inputs.i_p_start_time,
                i_p_end_time=inputs.i_p_end_time,
            )
        )

        # ---------------------------------------------------------------
        # [V,2] BehaviorModel — ModelVariant on decision_strategy
        # ---------------------------------------------------------------
        _behavior_common = dict(
            ts=ts,
            delta_from_start=policy_window.outputs.delta_from_start,
            delta_to_end=policy_window.outputs.delta_to_end,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            i_p_fraction_exempted=inputs.i_p_fraction_exempted,
            i_p_cost=inputs.i_p_cost,
            i_b_p50_cost=inputs.i_b_p50_cost,
            i_b_p50_anticipating=inputs.i_b_p50_anticipating,
            i_b_p50_postponing=inputs.i_b_p50_postponing,
            i_b_pt_capillarity=inputs.i_b_pt_capillarity,
            i_b_pt_frequency=inputs.i_b_pt_frequency,
            i_b_pt_cost=inputs.i_b_pt_cost,
            i_b_pt_time=inputs.i_b_pt_time,
            i_p_pt_capillarity_modification=inputs.i_p_pt_capillarity_modification,
            i_p_pt_frequency_modification=inputs.i_p_pt_frequency_modification,
            i_p_pt_cost_modification=inputs.i_p_pt_cost_modification,
            i_p_pt_time_modification=inputs.i_p_pt_time_modification,
            euro_class_split=inputs.euro_class_split,
            pt_capillarity=inputs.pt_capillarity,
            pt_frequency=inputs.pt_frequency,
            pt_avg_cost=inputs.pt_avg_cost,
            pt_avg_time_diff=inputs.pt_avg_time_diff,
            pt_zone_proportions=inputs.pt_zone_proportions,
        )
        _behavior_config_common = dict(
            policy=config.policy,
            modal_shift_option=config.modal_shift_option,
        )
        behavior: ModelVariant = ModelVariant(
            "BehaviorModel",
            variants={
                "parallel": ParallelBehaviorModel(
                    inputs=ParallelBehaviorModel.Inputs(**_behavior_common),
                    config=ParallelBehaviorModel.Config(**_behavior_config_common)
                ),
                "sequential": SequentialBehaviorModel(
                    inputs=SequentialBehaviorModel.Inputs(**_behavior_common),
                    config=SequentialBehaviorModel.Config(**_behavior_config_common)
                ),
            },
            selector=config.decision_strategy,
        )

        # ---------------------------------------------------------------
        # [D] TimeShiftModel
        # ---------------------------------------------------------------
        time_shift = TimeShiftModel(
            inputs=TimeShiftModel.Inputs(
                ts_inflow=inputs.ts_inflow,
                fraction_anticipating=behavior.outputs.fraction_anticipating,
                fraction_postponing=behavior.outputs.fraction_postponing,
                fraction_mode_shifted=behavior.outputs.fraction_mode_shifted,
                fraction_lost=behavior.outputs.fraction_lost,
                delta_from_start=policy_window.outputs.delta_from_start,
                delta_to_end=policy_window.outputs.delta_to_end,
                delta_before_start=policy_window.outputs.delta_before_start,
                delta_after_end=policy_window.outputs.delta_after_end,
                i_b_p50_anticipating=inputs.i_b_p50_anticipating,
                i_b_p50_postponing=inputs.i_b_p50_postponing,
                i_b_p50_anticipation=inputs.i_b_p50_anticipation,
                i_b_p50_postponement=inputs.i_b_p50_postponement,
            ),
            config=TimeShiftModel.Config(
                time_shift_strategy=config.time_shift_strategy
            )
        )

        # ---------------------------------------------------------------
        # [V.4] ModifiedInflowModel — ModelVariant on policy
        # ---------------------------------------------------------------
        _inflow_common = dict(
            ts=ts,
            ts_inflow=inputs.ts_inflow,
            ts_starting=inputs.ts_starting,
            total_base_inflow=base_flows.outputs.total_base_inflow,
            fraction_rigid=behavior.outputs.fraction_rigid,
            fraction_rigid_euro_class=behavior.outputs.fraction_rigid_euro_class,
            fraction_anticipating=behavior.outputs.fraction_anticipating,
            fraction_postponing=behavior.outputs.fraction_postponing,
            fraction_mode_shifted=behavior.outputs.fraction_mode_shifted,
            fraction_lost=behavior.outputs.fraction_lost,
            number_time_shifted=time_shift.outputs.number_time_shifted,
            number_mode_shifted=time_shift.outputs.number_mode_shifted,
            number_lost=time_shift.outputs.number_lost,
            delta_before_start=policy_window.outputs.delta_before_start,
            delta_after_end=policy_window.outputs.delta_after_end,
            delta_from_start=policy_window.outputs.delta_from_start,
            delta_to_end=policy_window.outputs.delta_to_end,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            i_p_fraction_exempted=inputs.i_p_fraction_exempted,
            i_p_cost=inputs.i_p_cost,
            i_b_p50_anticipation=inputs.i_b_p50_anticipation,
            i_b_p50_postponement=inputs.i_b_p50_postponement,
            i_b_p50_anticipating=inputs.i_b_p50_anticipating,
            i_b_p50_postponing=inputs.i_b_p50_postponing,
            euro_class_split=inputs.euro_class_split,
        )
        modified_flows: ModelVariant = ModelVariant(
            "ModifiedFlows",
            variants={
                "cordon-based": CordonModifiedFlowsModel(
                    inputs=CordonModifiedFlowsModel.Inputs(**_inflow_common)
                ),
                "area-based": AreaModifiedFlowsModel(
                    inputs=AreaModifiedFlowsModel.Inputs(**_inflow_common),
                    config=AreaModifiedFlowsModel.Config(time_shift_strategy=config.time_shift_strategy)
                ),
            },
            selector=config.policy,
        )

        # ---------------------------------------------------------------
        # [V.5] RevenueVariant — ModelVariant on policy
        # ---------------------------------------------------------------
        _revenue_common = dict(
            ts=ts,
            ts_inflow=inputs.ts_inflow,
            ts_starting=inputs.ts_starting,
            fraction_rigid=behavior.outputs.fraction_rigid,
            fraction_rigid_euro_class=behavior.outputs.fraction_rigid_euro_class,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            i_p_cost=inputs.i_p_cost,
        )
        revenue: ModelVariant = ModelVariant(
            "Revenue",
            variants={
                "cordon-based": CordonRevenueModel(
                    inputs=CordonRevenueModel.Inputs(**_revenue_common)
                ),
                "area-based": AreaRevenueModel(
                    inputs=AreaRevenueModel.Inputs(**_revenue_common)
                ),
            },
            selector=config.policy,
        )

        # ---------------------------------------------------------------
        # [E] TrafficModel
        # ---------------------------------------------------------------
        traffic = TrafficModel(
            inputs=TrafficModel.Inputs(
                ts_inflow=inputs.ts_inflow,
                ts_starting=inputs.ts_starting,
                modified_inflow=modified_flows.outputs.modified_inflow,
                modified_starting=modified_flows.outputs.modified_starting,
            )
        )

        # ---------------------------------------------------------------
        # [F] ModifiedZonalFlowsModel
        # ---------------------------------------------------------------
        modified_zonal_flows = ModifiedZonalFlowsModel(
            inputs=ModifiedZonalFlowsModel.Inputs(
                ts_inflow=inputs.ts_inflow,
                ts_starting=inputs.ts_starting,
                ts_traffic=traffic.outputs.ts_traffic,
                ts_inflow_zone_from_outside=base_zonal_flows.outputs.ts_inflow_zone_from_outside,
                ts_inflow_zone_from_inside=base_zonal_flows.outputs.ts_inflow_zone_from_inside,
                ts_inflow_zone=base_zonal_flows.outputs.ts_inflow_zone,
                ts_traffic_zone=base_zonal_flows.outputs.ts_traffic_zone,
                modified_inflow=modified_flows.outputs.modified_inflow,
                modified_starting=modified_flows.outputs.modified_starting,
                modified_traffic=traffic.outputs.modified_traffic,
            )
        )

        # ---------------------------------------------------------------
        # [G] EmissionsModel
        # ---------------------------------------------------------------
        emissions_model = EmissionsModel(
            inputs=EmissionsModel.Inputs(
                ts_traffic=traffic.outputs.ts_traffic,
                modified_traffic=traffic.outputs.modified_traffic,
                ts_traffic_zone=base_zonal_flows.outputs.ts_traffic_zone,
                modified_euro_class_split=modified_flows.outputs.modified_euro_class_split,
                modified_traffic_zone=modified_zonal_flows.outputs.modified_traffic_zone,
                euro_class_emission=inputs.euro_class_emission,
                euro_class_split=inputs.euro_class_split,
            )
        )

        # ---------------------------------------------------------------
        # [H] EthicalModel
        # ---------------------------------------------------------------
        # Built only when `enable_ethics` is set; ethics extraction always runs.
        ethical: dict[str, EthicalModelBundle] = {}
        ethical_expose: dict[str, dict[int, Index]] = {}
        if config.enable_ethics:
            _ethical_kwargs = dict(
                ztl_start=inputs.i_p_start_time,
                ztl_end=inputs.i_p_end_time,
                avail_postpone=inputs.i_b_p50_postponing,
                avail_anticipate=inputs.i_b_p50_anticipating,
                median_postpone=inputs.i_b_p50_postponement,
                median_anticipate=inputs.i_b_p50_anticipation,
                projection=config.ethical_projection,
                ticket_costs=inputs.i_p_cost,
                euro_class_split=inputs.euro_class_split,
            )
            for case_id in ("elderly", "minors", "income"):
                ethical[case_id] = EthicalModelBundle(case_id=case_id, **_ethical_kwargs)
                ethical_expose.update(_ethical_expose_fields(ethical[case_id], case_id))

        # ---------------------------------------------------------------
        # Class attributes
        # ---------------------------------------------------------------
        # Named sub-model attributes for structured access
        self.base_flows = base_flows
        self.policy_window = policy_window
        self.behavior = behavior
        self.time_shift = time_shift
        self.modified_flows = modified_flows
        self.revenue = revenue
        self.traffic = traffic
        self.modified_zonal_flows = modified_zonal_flows
        self.emissions = emissions_model
        self.ethical = ethical
        self.in_zones = config.in_zones
        self.out_zones = config.out_zones
        self.all_zones = config.all_zones
        self.pt_zones = config.pt_zones

        # View/dashboard aliases — backward-compatible flat attribute access
        # used by view.py (plot_flow, plot_traffic, plot_emissions) and the
        # download callbacks in dashboard.py.
        self.TS_inflow = inputs.ts_inflow
        self.TS_starting = inputs.ts_starting
        self.TS_inflow_zone = base_zonal_flows.outputs.ts_inflow_zone
        self.TS_inflow_zone_from_inside = base_zonal_flows.outputs.ts_inflow_zone_from_inside
        self.TS_inflow_zone_from_outside = base_zonal_flows.outputs.ts_inflow_zone_from_outside
        self.TS_traffic_zone = base_zonal_flows.outputs.ts_traffic_zone
        self.I_modified_inflow = modified_flows.outputs.modified_inflow
        self.I_modified_inflow_zone = modified_zonal_flows.outputs.modified_inflow_zone
        self.I_traffic = traffic.outputs.ts_traffic
        self.I_modified_traffic = traffic.outputs.modified_traffic
        self.I_modified_traffic_zone = modified_zonal_flows.outputs.modified_traffic_zone
        self.I_emissions = emissions_model.outputs.emissions
        self.I_emissions_zone = emissions_model.outputs.emissions_zone
        self.I_modified_emissions = emissions_model.outputs.modified_emissions
        self.I_modified_emissions_zone = emissions_model.outputs.modified_emissions_zone

        # Policy parameters — aliased for direct external access
        self.I_P_start_time = inputs.i_p_start_time
        self.I_P_end_time = inputs.i_p_end_time
        self._I_P_cost_vector = inputs.i_p_cost
        self.I_P_cost = [
            Index(
                f"{inputs.i_p_cost.name}[{euro_class}]",
                inputs.i_p_cost.concrete_default[euro_class],
            )
            for euro_class in range(len(inputs.i_p_cost.concrete_default))
        ]
        self.I_P_fraction_exempted = inputs.i_p_fraction_exempted
        self.I_P_pt_frequency_modification = inputs.i_p_pt_frequency_modification
        self.I_P_pt_capillarity_modification = inputs.i_p_pt_capillarity_modification
        self.I_P_pt_cost_modification = inputs.i_p_pt_cost_modification
        self.I_P_pt_time_modification = inputs.i_p_pt_time_modification

        # Behavioural parameters — aliased for direct external access
        self.I_B_p50_cost = inputs.i_b_p50_cost
        self.I_B_p50_anticipating = inputs.i_b_p50_anticipating
        self.I_B_p50_postponing = inputs.i_b_p50_postponing
        self.I_B_p50_anticipation = inputs.i_b_p50_anticipation
        self.I_B_p50_postponement = inputs.i_b_p50_postponement
        self.I_B_pt_capillarity = inputs.i_b_pt_capillarity
        self.I_B_pt_frequency = inputs.i_b_pt_frequency
        self.I_B_pt_cost = inputs.i_b_pt_cost
        self.I_B_pt_time = inputs.i_b_pt_time

        # Geo/zone data — needed by dashboard and tests
        # self.in_zones = in_zones
        # self.out_zones = out_zones
        # self.all_zones = all_zones
        # self.aree_gdf_inside = aree_gdf_inside
        # self.zones_in_to_names = zones_in_to_names
        # self.names_in_to_zones = names_in_to_zones
        # self.pt_zones = pt_zones
        self.pt_zone_proportions = inputs.pt_zone_proportions

                # ---------------------------------------------------------------
        # Root super().__init__
        # ---------------------------------------------------------------
        return (
            MobilityModel.Outputs(
                total_base_inflow=base_flows.outputs.total_base_inflow,
                total_modified_inflow=modified_flows.outputs.total_modified_inflow,
                total_traffic=traffic.outputs.total_traffic,
                total_modified_traffic=traffic.outputs.total_modified_traffic,
                total_mode_shifted=time_shift.outputs.total_mode_shifted,
                total_time_shifted=time_shift.outputs.total_time_shifted,
                total_lost=time_shift.outputs.total_lost,
                total_paying=revenue.outputs.total_paying,
                modified_avg_cost=revenue.outputs.modified_avg_cost,
                total_paid=revenue.outputs.total_paid,
                total_emissions=emissions_model.outputs.total_emissions,
                total_modified_emissions=emissions_model.outputs.total_modified_emissions,
            ),
            MobilityModel.Expose(
                modified_inflow=modified_flows.outputs.modified_inflow,
                modified_inflow_zone=modified_zonal_flows.outputs.modified_inflow_zone,
                traffic=traffic.outputs.ts_traffic,
                traffic_zone=base_zonal_flows.outputs.ts_traffic_zone,
                modified_traffic=traffic.outputs.modified_traffic,
                modified_traffic_zone=modified_zonal_flows.outputs.modified_traffic_zone,
                emissions=emissions_model.outputs.emissions,
                emissions_zone=emissions_model.outputs.emissions_zone,
                modified_emissions=emissions_model.outputs.modified_emissions,
                modified_emissions_zone=emissions_model.outputs.modified_emissions_zone,
                ts_inflow=inputs.ts_inflow,
                ts_starting=inputs.ts_starting,
                ts_inflow_zone_from_inside=base_zonal_flows.outputs.ts_inflow_zone_from_inside,
                ts_inflow_zone_from_outside=base_zonal_flows.outputs.ts_inflow_zone_from_outside,
                ts_inflow_zone=base_zonal_flows.outputs.ts_inflow_zone,
                ts_traffic_zone=base_zonal_flows.outputs.ts_traffic_zone,
                **ethical_expose,
            ),
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_model(model: MobilityModel, overrides: dict | None, size: int = 1, rng: int | None = None) -> EvaluationResult:
    """Evaluate *model* under optional per-index value *overrides*.

    Builds a fresh :class:`Scenario` per call instead of mutating the
    model's own ``Index`` nodes in place, so the same model instance can be
    re-evaluated under different UI-selected parameter values concurrently.

    *rng* is a seed: it is turned into a :class:`numpy.random.Generator`, so
    that distribution-backed parameters get independent draws (an integer passed
    straight to the ensemble would restart the same stream for each of them).

    Returns
    -------
    EvaluationResult
        Use :meth:`EvaluationResult.expected_value` to extract scalars/series.
    """
    if overrides:
        overrides = dict(overrides)
        cost_values = None
        for euro_class, cost_index in enumerate(model.I_P_cost):
            if cost_index in overrides:
                if cost_values is None:
                    cost_values = np.array(model._I_P_cost_vector.concrete_default, copy=True)
                cost_values[euro_class] = overrides.pop(cost_index)
        if cost_values is not None:
            overrides[model._I_P_cost_vector] = cost_values

    scenario = Scenario(model, overrides=overrides)
    return Evaluation(scenario).evaluate(
        ensemble=DistributionEnsemble(
            scenario, size=size, rng=None if rng is None else np.random.default_rng(rng)
        ),
        functions={
            "ts_solve": NumpyBackend.adapt(ts_solve),
            "ts_b_choose": NumpyBackend.adapt(ts_b_choose),
            "ts_anticipate": NumpyBackend.adapt(ts_anticipate),
            "ts_postpone": NumpyBackend.adapt(ts_postpone),
        },
        backend=NumpyBackend,
    )


def compute_kpis(m: MobilityModel, evals: EvaluationResult) -> dict:
    """Compute scalar KPIs from an :class:`EvaluationResult`.

    Parameters
    ----------
    m :
        The evaluated :class:`MobilityModel`.
    evals :
        Output of :func:`evaluate_model`.

    Returns
    -------
    dict
        Human-readable KPI name → scalar value.
    """
    def _round_eval(v, r: int = 0):
        if r == 0:
            return int(v.item())
        else:
            return round(v.item(), r)

    return {
        "Base inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_base_inflow), 0),
        "Mode-shifted inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_mode_shifted), 0),
        "Time-shifted inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_time_shifted), 0),
        "Lost inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_lost), 0),
        "Modified inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_modified_inflow), 0),
        "Inflow difference [veh/day]": (
            _round_eval(evals.expected_value(m.outputs.total_modified_inflow), 0)
            - _round_eval(evals.expected_value(m.outputs.total_base_inflow), 0)
        ),
        "Base traffic [max veh]": _round_eval(evals.expected_value(m.outputs.total_traffic), 0),
        "Modified traffic [max veh]": _round_eval(evals.expected_value(m.outputs.total_modified_traffic), 0),
        "Traffic difference [max veh]": (
            _round_eval(evals.expected_value(m.outputs.total_modified_traffic), 0)
            - _round_eval(evals.expected_value(m.outputs.total_traffic), 0)
        ),
        "Paying inflow [veh/day]": _round_eval(evals.expected_value(m.outputs.total_paying), 0),
        "Average cost [€/day]": _round_eval(evals.expected_value(m.outputs.modified_avg_cost), 2),
        "Collected fees [€/day]": _round_eval(evals.expected_value(m.outputs.total_paid), 2),
        "Base emissions [NOx gr/day]": _round_eval(evals.expected_value(m.outputs.total_emissions), 0),
        "Emissions [NOx gr/day]": _round_eval(evals.expected_value(m.outputs.total_modified_emissions), 0),
        "Emissions difference [NOx gr/day]": (
            _round_eval(evals.expected_value(m.outputs.total_modified_emissions), 0)
            - _round_eval(evals.expected_value(m.outputs.total_emissions), 0)
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging_level)
    logging.info(
        f"Building MobilityModel (policy={config_params['policy']!r}, decision={config_params['decision_strategy']!r}, "
        f"shift={config_params['time_shift_strategy']!r}, modal={config_params['modal_shift_option']!r}) …"
    )
    with timed():
        logging.debug("Default config construction")
        default_config = MobilityModel.default_config(
            config_params=config_params
        )
        logging.debug("Default inputs construction")
        default_inputs = MobilityModel.default_inputs(
            config=default_config, 
            policy_params=policy_params, 
            behavioral_params=behavioral_params
        )
        logging.debug("Model construction")
        mod = MobilityModel(
            config=default_config,
            inputs=default_inputs,
        )
        logging.debug(f"Model constructed: {mod.name}  indexes: {len(mod.indexes)}")

        sce = Scenario(model=mod)
        logging.debug(f"Scenario constructed for the model")

        logging.debug("Evaluating...")
        ensemble = DistributionEnsemble(sce, size=10, rng=np.random.default_rng(17*11))
        result = Evaluation(sce).evaluate(
            ensemble=ensemble,
            functions={
                "ts_solve": NumpyBackend.adapt(ts_solve),
                "ts_b_choose": NumpyBackend.adapt(ts_b_choose),
                "ts_anticipate": NumpyBackend.adapt(ts_anticipate),
                "ts_postpone": NumpyBackend.adapt(ts_postpone),
            },
            backend=NumpyBackend,
        )

        logging.debug("KPIs...")
        kpis = compute_kpis(m=mod, evals=result)

        logging.info("Printing all the KPIs")
        for name, value in kpis.items():
            print(f"  {name}: {value}")
