"""Traffic and emissions sub-models — [E] TrafficModel, [F] ModifiedZonalFlowsModel, [G] EmissionsModel."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from civic_digital_twins.dt_model import (
    Index, TimeseriesIndex, Functor,
    Model, 
    define, inputs, outputs, expose, functions
)
from civic_digital_twins.dt_model.engine.frontend import graph

from bdt_mobility.model.axis import (
    EclIndex,
    ZoneInTimeseriesIndex,
    TimeseriesEclIndex,
    ECL_AXIS
)


# ---------------------------------------------------------------------------
# [E] TrafficModel
# ---------------------------------------------------------------------------


@define("Traffic")
class TrafficModel(Model):
    """Computes traffic from inflow and starting timeseries.

    Computes baseline ``traffic`` from ``ts_inflow`` + ``ts_starting``, then
    ``modified_traffic`` from ``modified_inflow`` + ``modified_starting``.

    Parameters
    ----------
    ts_inflow :
        Baseline total inflow timeseries.
    ts_starting :
        Baseline starting timeseries.
    modified_inflow :
        Area-level modified inflow (wired from :class:`ModifiedFlowsVariant`).
    modified_starting :
        Area-level modified starting (wired from :class:`ModifiedFlowsVariant`).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ModifiedZonalFlowsModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`TrafficModel`."""

        ts_traffic: TimeseriesIndex
        total_traffic: Index
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index

    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        ts_traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("ts_solve", inputs.ts_inflow + inputs.ts_starting),
        )
        total_traffic = Index("max reference traffic", ts_traffic.max())
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("ts_solve", inputs.modified_inflow + inputs.modified_starting),
        )
        total_modified_traffic = Index("max modified traffic", modified_traffic.max())

        return TrafficModel.Outputs(
            ts_traffic=ts_traffic,
            total_traffic=total_traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
        )


# ---------------------------------------------------------------------------
# [F] ModifiedZonalFlowsModel
# ---------------------------------------------------------------------------


@define("ModifiedZonalFlows")
class ModifiedZonalFlowsModel(Model):
    """Downscales area-level modified flows and traffic to individual zones.

    Takes baseline inflow, starting and traffic time series, both in the base
    and in the modified scenario, and the zonal quantities in the base scenario.
    Then, uses the resulting base ratios to downscale per-zone flows
    in the modified scenario.

    Parameters
    ----------
    zones :
        Internal zone ids (for zone-level emission computation).
    ts_inflow :
        Baseline total inflow timeseries.
    ts_starting :
        Baseline starting timeseries.
    ts_traffic :
        Baseline traffic timeseries (wired from :class:`TrafficModel`).
    ts_inflow_zone_from_outside :
        Per-zone inflow from outside the ZOI.
    ts_inflow_zone_from_inside :
        Per-zone inflow from inside the ZOI (starting).
    ts_inflow_zone :
        Per-zone total inflow.
    ts_traffic_zone :
        Per-zone baseline traffic.
    modified_inflow :
        Modified inflow timeseries (wired from :class:`ModifiedFlowsVariant`).
    modified_starting :
        Modified starting timeseries (wired from :class:`ModifiedFlowsVariant`).
    modified_traffic :
        Modified traffic timeseries (wired from :class:`TrafficModel`).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ModifiedZonalFlowsModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        ts_traffic: TimeseriesIndex
        ts_inflow_zone_from_outside: Index
        ts_inflow_zone_from_inside: Index
        ts_inflow_zone: Index
        ts_traffic_zone: Index
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex
        modified_traffic: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`ModifiedZonalFlowsModel`."""

        inflow_ratio: TimeseriesIndex
        starting_ratio: TimeseriesIndex
        traffic_ratio: TimeseriesIndex
        modified_inflow_zone: ZoneInTimeseriesIndex
        modified_traffic_zone: ZoneInTimeseriesIndex

    @expose
    class Expose:
        """Exposed outputs of :class:`ModifiedZonalFlowsModel`."""

        delta_inflow_zone: ZoneInTimeseriesIndex
        delta_traffic_zone: ZoneInTimeseriesIndex

    def compute(
        self,
        inputs: Inputs
    ) -> tuple[Outputs, Expose]:

        inflow_ratio = TimeseriesIndex("ratio between modified flow and base flow", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio = TimeseriesIndex(
            "ratio between modified starting and base starting", inputs.ts_starting / inputs.modified_starting
        )
        traffic_ratio = TimeseriesIndex("ratio between modified traffic and base traffic", inputs.ts_traffic / inputs.modified_traffic)

        modified_inflow_zone = ZoneInTimeseriesIndex(
            "modified zonal inflow",
            inputs.ts_inflow_zone_from_outside / inflow_ratio
            + inputs.ts_inflow_zone_from_inside / starting_ratio
        )
        delta_inflow_zone = ZoneInTimeseriesIndex(
            "delta zonal inflow",
            modified_inflow_zone - inputs.ts_inflow_zone
        )

        modified_traffic_zone = ZoneInTimeseriesIndex(
            "modified zonal traffic",
            inputs.ts_traffic_zone / traffic_ratio
        )
        delta_traffic_zone = ZoneInTimeseriesIndex(
            "delta zonal traffic",
            modified_traffic_zone - inputs.ts_traffic_zone

        )

        return (
            ModifiedZonalFlowsModel.Outputs(
                inflow_ratio=inflow_ratio,
                starting_ratio=starting_ratio,
                traffic_ratio=traffic_ratio,
                modified_inflow_zone=modified_inflow_zone,
                modified_traffic_zone=modified_traffic_zone,
            ),
            ModifiedZonalFlowsModel.Expose(
                delta_inflow_zone=delta_inflow_zone,
                delta_traffic_zone=delta_traffic_zone,
            ),
        )


# ---------------------------------------------------------------------------
# [G] EmissionsModel
# ---------------------------------------------------------------------------


@define("Emissions")
class EmissionsModel(Model):
    """Baseline and modified emissions at area and zone level.

    Computes the fleet-average emission factor internally from ``euro_class_emission``
    and the module-level ``euro_class_split``, following the Bologna pattern.

    Parameters
    ----------
    euro_class_emission :
        NOx emission factors per euro class (g/veh/km).
    zones :
        Internal zone ids (for zone-level emission computation).
    traffic :
        Baseline traffic timeseries (wired from :class:`ModifiedZonalFlowsModel`).
    modified_traffic :
        Modified traffic timeseries (wired from :class:`ModifiedZonalFlowsModel`).
    ts_traffic_zone :
        Per-zone baseline traffic.
    modified_euro_class_split :
        Modified fleet euro-class split (wired from
        :class:`ModifiedFlowsVariant`).
    modified_traffic_zone :
        Per-zone modified traffic (wired from
        :class:`ModifiedZonalFlowsModel`).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`EmissionsModel`."""

        ts_traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        ts_traffic_zone: Index
        modified_euro_class_split: Index
        modified_traffic_zone: Index
        euro_class_emission: Index
        euro_class_split: Index

    @outputs
    class Outputs:
        """Outputs of :class:`EmissionsModel`."""

        emissions: TimeseriesIndex
        modified_emissions: TimeseriesIndex
        total_emissions: Index
        total_modified_emissions: Index
        emissions_zone: Index
        modified_emissions_zone: Index
        delta_emissions_zone: Index

    @expose
    class Expose:
        """Inspectable intermediates of :class:`EmissionsModel`."""

        average_emissions: Index
        modified_average_emissions: TimeseriesIndex

    def compute(
        self,
        inputs: Inputs
    ) -> tuple[Outputs, Expose]:

        average_emissions = Index(
            "average emissions (per vehicle, per km)", 
            (inputs.euro_class_emission * inputs.euro_class_split).sum(axis=ECL_AXIS)
        )

        modified_average_emissions = TimeseriesIndex(
            "modified average emissions (per vehicle, per km)", 
            (inputs.euro_class_emission * inputs.modified_euro_class_split).sum(axis=ECL_AXIS)
        )

        emissions = TimeseriesIndex("emissions", 2.5 * average_emissions * inputs.ts_traffic)
        modified_emissions = TimeseriesIndex("modified emissions", 2.5 * modified_average_emissions * inputs.modified_traffic)
        total_emissions = Index("total emissions", emissions.sum())
        total_modified_emissions = Index("total modified emissions", modified_emissions.sum())

        emissions_zone = ZoneInTimeseriesIndex(
            "zonal emissions",
            2.5 * average_emissions * inputs.ts_traffic_zone
        )

        modified_emissions_zone = ZoneInTimeseriesIndex(
            "modified zonal emissions",
            2.5 * modified_average_emissions * inputs.modified_traffic_zone
        )

        delta_emissions_zone = ZoneInTimeseriesIndex(
            "delta zonal emissions",
            modified_emissions_zone - emissions_zone
        )

        return(
            EmissionsModel.Outputs(
                emissions=emissions,
                modified_emissions=modified_emissions,
                total_emissions=total_emissions,
                total_modified_emissions=total_modified_emissions,
                emissions_zone=emissions_zone,
                modified_emissions_zone=modified_emissions_zone,
                delta_emissions_zone=delta_emissions_zone,
            ),
            EmissionsModel.Expose(
                average_emissions=average_emissions,
                modified_average_emissions=modified_average_emissions,
            ),
        )
