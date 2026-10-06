"""Baseline flow models — [A] BaseFlowsModel, [B] BaseZonalFlowsModel."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from civic_digital_twins.dt_model import (
    Index, TimeseriesIndex,
    Model, 
    inputs, outputs, define, config,
)
from bdt_mobility.model.axis import (
    ZoneInTimeseriesIndex
)

def zone_flows_df_to_mat(zone_flows_df, zones, flow_col):
    """Convert a zone flows DataFrame to a matrices of ZONExTIME."""
    zone_flows_index = {}
    for zone in zones:
        zone_mask = zone_flows_df["index"] == zone
        if zone_mask.any():
            zone_sorted = zone_flows_df[zone_mask].sort_values("start_t")
            flow_values = zone_sorted[flow_col].to_numpy() / 12  # Convert to 5-minute resolution
        else:
            flow_values = np.zeros(24)  # Default to zero if no data available
        zone_flows_index[int(zone)] = flow_values
    result = np.array(list(zone_flows_index.values()))
    return result


# ---------------------------------------------------------------------------
# [A] BaseFlowsModel
# ---------------------------------------------------------------------------

@define("BaseFlows")
class BaseFlowsModel(Model):
    """Baseline (unmodified) flow timeseries for the ZOI system.

    Loads inflow and starting timeseries in the ZOI from raw data and exposes the
    raw timeseries needed by downstream models.  Derived quantities (baseline
    traffic, emissions) are computed by :class:`ModifiedZonalFlowsModel` and
    :class:`EmissionsModel` respectively, following the Bologna
    pattern.

    Parameters
    ----------
    ts :
        Shared time-range :class:`TimeseriesIndex`.
    ts_inflow_data :
        Numpy array of vehicle inflow values (one entry per hour, upsampled
        internally to 5-minute resolution).
    ts_starting_data :
        Numpy array of vehicle starting values.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`BaseFlowsModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`BaseFlowsModel`."""

        total_base_inflow: Index

    def compute(
        self,
        inputs: Inputs,
    ) -> Outputs:

        total_base_inflow = Index("total base vehicle inflow", inputs.ts_inflow.sum())

        return BaseFlowsModel.Outputs(
            total_base_inflow=total_base_inflow,
        )


# ---------------------------------------------------------------------------
# [B] BaseZonalFlowsModel
# ---------------------------------------------------------------------------

@define("BaseZonalFlows")
class BaseZonalFlowsModel(Model):
    """Baseline (unmodified) flow timeseries for the zones inside the ZOI system.

    Loads per-zone inflow and traffic timeseries of the baseline scenario
    from raw data and exposes the raw timeseries needed by downstream models.

    Parameters
    ----------
    ts :
        Shared time-range :class:`TimeseriesIndex`.
    zone_io_df :
        DataFrame with columns ``id_zone``, ``start_t``,
        ``inflow_from_inside_mean``, ``inflow_from_outside_mean``.
    zone_traffic_df :
        DataFrame with columns ``id_zone``, ``start_t``,
        ``traffic_in_zone_mean``.
    zones :
        Array of internal zone ids.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`BaseZonalFlowsModel`."""

        ts: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`BaseZonalFlowsModel`."""

        ts_inflow_zone_from_inside: ZoneInTimeseriesIndex
        ts_inflow_zone_from_outside: ZoneInTimeseriesIndex
        ts_inflow_zone: ZoneInTimeseriesIndex
        ts_traffic_zone: ZoneInTimeseriesIndex

    @config
    class Config:
        zones_ifi_mat: np.ndarray
        zones_ifo_mat: np.ndarray
        zones_t_mat: np.ndarray

    # TODO: let the time frequency of 5 minutes vary depending on a config file
    @staticmethod
    def _upsample_single(values: np.ndarray) -> np.ndarray:
        """Upsample hourly values to 5-minute resolution via cubic interpolation. Works with 1D-arrays."""
        df = pd.DataFrame({"values": values})
        df["n"] = df.index * 12 + 6
        df = df.set_index("n").reindex(range(24 * 12))
        vals = list(values) if not isinstance(values, list) else values
        df.loc[0, "values"] = (vals[0] + vals[-1]) / 2
        df.loc[24 * 12 - 1, "values"] = df.loc[0, "values"]
        df = df.interpolate(method="cubic")
        df[df["values"] < 0] = 0
        return np.asarray(df["values"].values)

    @staticmethod
    def _upsample(values: np.ndarray) -> np.ndarray:
        """Upsample hourly values to 5-minute resolution via cubic interpolation. Works with 2D-arrays."""
        # If the array is 1D, then consider it as a single row
        if values.ndim == 1:
            return BaseZonalFlowsModel._upsample_single(values)
        # If it is a 2D matrix, then process each row one after the other
        result = np.zeros((values.shape[0], 24 * 12))
        for i, row in enumerate(values):
            result[i] = BaseZonalFlowsModel._upsample_single(row)
        return result

    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> Outputs:

        ts_inflow_zone_from_inside = ZoneInTimeseriesIndex(
            "zonal inflow from inside",
            BaseZonalFlowsModel._upsample(config.zones_ifi_mat)
        )

        ts_inflow_zone_from_outside = ZoneInTimeseriesIndex(
            "zonal inflow from outside",
            BaseZonalFlowsModel._upsample(config.zones_ifo_mat)
        )
  
        ts_inflow_zone = ZoneInTimeseriesIndex(
            "zonal inflow",
            ts_inflow_zone_from_inside + ts_inflow_zone_from_outside
        )

        ts_traffic_zone = ZoneInTimeseriesIndex(
            "zonal traffic",
            BaseZonalFlowsModel._upsample(config.zones_t_mat)
        )

        return BaseZonalFlowsModel.Outputs(
            ts_inflow_zone_from_inside=ts_inflow_zone_from_inside,
            ts_inflow_zone_from_outside=ts_inflow_zone_from_outside,
            ts_inflow_zone=ts_inflow_zone,
            ts_traffic_zone=ts_traffic_zone,
        )
