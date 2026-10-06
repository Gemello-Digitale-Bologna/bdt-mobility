"""Policy-modified flow and revenue sub-models — [V.4] ModifiedFlowsModel variants, [V.5] RevenueModel variants."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from civic_digital_twins.dt_model import (
    Index, TimeseriesIndex,
    Model, ModelVariant, 
    define, inputs, outputs, expose, config
)
from civic_digital_twins.dt_model.engine.frontend import graph

from .time_shift import FixedShiftFormula, FlexibleShiftFormula
from bdt_mobility.model.axis import (
    EclIndex,
    TimeseriesEclIndex,
    ECL_AXIS
)


# ---------------------------------------------------------------------------
# [V.4] ModifiedFlowsModel — ModelVariant("ModifiedFlows", selector=policy)
# ---------------------------------------------------------------------------
#
# Both variants share the same Inputs and Outputs dataclass field names.
# Expose differs: AreaModifiedFlowsModel exposes the "inside" shift indexes
# that only exist under area-based policy.
# ---------------------------------------------------------------------------


def _compute_modified_inflow(
    ts_inflow: TimeseriesIndex,
    fraction_rigid: TimeseriesIndex,
    number_time_shifted: TimeseriesIndex,
    i_p_fraction_exempted: Index,
    i_p_start_time: Index,
    i_p_end_time: Index,
    ts: TimeseriesIndex,
) -> TimeseriesIndex:
    """Shared formula for modified inflow (identical across policies)."""
    return TimeseriesIndex(
        "modified vehicle inflow",
        graph.piecewise(
            (
                (i_p_fraction_exempted + fraction_rigid) * ts_inflow,
                (ts >= i_p_start_time) & (ts <= i_p_end_time),
            ),
            (ts_inflow + number_time_shifted, True),
        ),
    )


@define("CordonModifiedFlows")
class CordonModifiedFlowsModel(Model):
    """Modified inflow and starting under cordon-based policy.

    Under cordon policy, only vehicles *entering* the ZOI are subject to the
    pricing rule.  ``modified_starting`` is therefore equal to
    ``ts_starting`` (pass-through).

    Parameters
    ----------
    euro_class_split :
        Fleet euro-class proportions.
    ts, ts_inflow, ts_starting :
        Baseline timeseries.
    total_base_inflow :
        Scalar total baseline inflow (wired from :class:`BaseStateModel`).
    fraction_rigid, fraction_rigid_euro_class :
        Wired from :class:`BehaviorModel`.
    fraction_anticipating, fraction_postponing :
        Wired from :class:`BehaviorModel`.
    fraction_mode_shifted, fraction_lost :
        Wired from :class:`BehaviorModel`.
    number_time_shifted, number_mode_shifted, number_lost :
        Wired from :class:`TimeShiftModel`.
    delta_before_start, delta_after_end :
        Wired from :class:`PolicyWindowModel` (unused in cordon; present
        for shared Inputs contract with :class:`AreaModifiedFlowsModel`).
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel` (unused in cordon; present
        for shared Inputs contract).
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_anticipation, i_b_p50_postponement :
        Unused in cordon (present for shared Inputs contract).
    i_b_p50_anticipating, i_b_p50_postponing :
        Unused in cordon (present for shared Inputs contract).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`CordonModifiedFlowsModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        total_base_inflow: Index
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: TimeseriesEclIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        number_time_shifted: TimeseriesIndex
        number_mode_shifted: TimeseriesIndex
        number_lost: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        euro_class_split: EclIndex

    @outputs
    class Outputs:
        """Outputs of :class:`CordonModifiedFlowsModel`."""

        modified_inflow: TimeseriesIndex
        total_modified_inflow: Index
        modified_starting: TimeseriesIndex
        modified_euro_class_split: TimeseriesEclIndex


    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        modified_inflow = _compute_modified_inflow(
            ts_inflow=inputs.ts_inflow,
            fraction_rigid=inputs.fraction_rigid,
            number_time_shifted=inputs.number_time_shifted,
            i_p_fraction_exempted=inputs.i_p_fraction_exempted,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            ts=inputs.ts,
        )
        total_modified_inflow = Index("total modified vehicle inflow", modified_inflow.sum())

        # Cordon: starting is unchanged — wrap the node so it has its own Index identity
        modified_starting = TimeseriesIndex("modified starting", inputs.ts_starting)

        # Cordon euro-class split: inflow-weighted formula
        modified_euro_class_split = TimeseriesEclIndex(
            "modified split euro levels",
            graph.piecewise(
                (
                    modified_inflow
                    / (modified_inflow + modified_starting)
                    * (
                        (inputs.i_p_fraction_exempted * inputs.euro_class_split)
                        + inputs.fraction_rigid_euro_class
                    )
                    / (inputs.i_p_fraction_exempted + inputs.fraction_rigid)
                    + modified_starting / (modified_inflow + modified_starting) * inputs.euro_class_split,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (inputs.euro_class_split, True),
            )
        )

        return CordonModifiedFlowsModel.Outputs(
            modified_inflow=modified_inflow,
            total_modified_inflow=total_modified_inflow,
            modified_starting=modified_starting,
            modified_euro_class_split=modified_euro_class_split,
        )


@define("AreaModifiedFlows")
class AreaModifiedFlowsModel(Model):
    """Modified inflow and starting under area-based policy.

    Under area policy, vehicles *starting* inside the ZOI are also subject
    to the pricing rule.  ``modified_starting`` is computed via the same
    time-shift formula applied to ``ts_starting``, using a second internal
    :class:`ShiftFormula` :class:`ModelVariant`.

    Parameters
    ----------
    time_shift_strategy :
        Selector for the internal :class:`ShiftFormula` variant used for
        the inside-starting shift computation.
    euro_class_split :
        Fleet euro-class proportions.
    ts, ts_inflow, ts_starting :
        Baseline timeseries.
    total_base_inflow :
        Scalar total baseline inflow.
    fraction_rigid, fraction_rigid_euro_class :
        Wired from :class:`BehaviorModel`.
    fraction_anticipating, fraction_postponing :
        Wired from :class:`BehaviorModel`.
    fraction_mode_shifted, fraction_lost :
        Wired from :class:`BehaviorModel`.
    number_time_shifted, number_mode_shifted, number_lost :
        Wired from :class:`TimeShiftModel`.
    delta_before_start, delta_after_end :
        Wired from :class:`PolicyWindowModel`.
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`.
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_anticipation, i_b_p50_postponement :
        Median-shift parameters.
    i_b_p50_anticipating, i_b_p50_postponing :
        50%-likelihood parameters.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`AreaModifiedFlowsModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        total_base_inflow: Index
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: Index
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        number_time_shifted: TimeseriesIndex
        number_mode_shifted: TimeseriesIndex
        number_lost: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        euro_class_split: EclIndex

    @outputs
    class Outputs:
        """Outputs of :class:`AreaModifiedFlowsModel`."""

        modified_inflow: TimeseriesIndex
        total_modified_inflow: Index
        modified_starting: TimeseriesIndex
        modified_euro_class_split: TimeseriesEclIndex

    @expose
    class Expose:
        """Inside-shift indexes (area-based policy only)."""

        number_anticipating_inside: TimeseriesIndex
        total_anticipating_inside: Index
        number_anticipated_inside: TimeseriesIndex
        total_anticipated_inside: Index
        number_postponing_inside: TimeseriesIndex
        total_postponing_inside: Index
        number_postponed_inside: TimeseriesIndex
        total_postponed_inside: Index
        number_time_shifted_inside: TimeseriesIndex
        total_time_shifted_inside: Index
        number_mode_shifted_inside: TimeseriesIndex
        total_mode_shifted_inside: Index
        number_lost_inside: TimeseriesIndex
        total_lost_inside: Index

    @config
    class Config:
        time_shift_strategy: str


    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> tuple[Outputs, Expose]:

        modified_inflow = _compute_modified_inflow(
            ts_inflow=inputs.ts_inflow,
            fraction_rigid=inputs.fraction_rigid,
            number_time_shifted=inputs.number_time_shifted,
            i_p_fraction_exempted=inputs.i_p_fraction_exempted,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            ts=inputs.ts,
        )
        total_modified_inflow = Index("total modified vehicle inflow", modified_inflow.sum())

        # Inside-shift computation for area-based modified_starting
        # (same ShiftFormula variant reused with ts_starting instead of ts_inflow)
        number_anticipating_inside = TimeseriesIndex(
            "anticipating vehicles inside zoi", inputs.fraction_anticipating * inputs.ts_starting
        )
        total_anticipating_inside = Index(
            "total anticipating vehicles inside zoi",
            number_anticipating_inside.sum(),
        )
        number_postponing_inside = TimeseriesIndex(
            "postponing vehicles inside zoi", inputs.fraction_postponing * inputs.ts_starting
        )
        total_postponing_inside = Index(
            "total postponing vehicles inside zoi",
            number_postponing_inside.sum(),
        )

        _inside_shift_common = dict(
            number_anticipating=number_anticipating_inside,
            total_anticipating=total_anticipating_inside,
            number_postponing=number_postponing_inside,
            total_postponing=total_postponing_inside,
            delta_before_start=inputs.delta_before_start,
            delta_after_end=inputs.delta_after_end,
            delta_from_start=inputs.delta_from_start,
            delta_to_end=inputs.delta_to_end,
            i_b_p50_anticipation=inputs.i_b_p50_anticipation,
            i_b_p50_postponement=inputs.i_b_p50_postponement,
            i_b_p50_anticipating=inputs.i_b_p50_anticipating,
            i_b_p50_postponing=inputs.i_b_p50_postponing,
        )
        inside_shift: ModelVariant = ModelVariant(
            "InsideShiftFormula",
            variants={
                "flexible": FlexibleShiftFormula(FlexibleShiftFormula.Inputs(**_inside_shift_common)),
                "fixed": FixedShiftFormula(FixedShiftFormula.Inputs(**_inside_shift_common)),
            },
            selector=config.time_shift_strategy,
        )

        number_anticipated_inside: TimeseriesIndex = inside_shift.outputs.number_anticipated
        total_anticipated_inside: Index = inside_shift.outputs.total_anticipated
        number_postponed_inside: TimeseriesIndex = inside_shift.outputs.number_postponed
        total_postponed_inside: Index = inside_shift.outputs.total_postponed

        number_time_shifted_inside = TimeseriesIndex(
            "time-shifted vehicles inside zoi", number_anticipated_inside + number_postponed_inside
        )
        total_time_shifted_inside = Index(
            "total vehicles shifted inside zoi", total_anticipated_inside + total_postponed_inside
        )
        number_mode_shifted_inside = TimeseriesIndex(
            "mode-shifted vehicles inside zoi", inputs.fraction_mode_shifted * inputs.ts_starting
        )
        total_mode_shifted_inside = Index(
            "total mode shifted vehicles inside zoi",
            number_mode_shifted_inside.sum(),
        )
        number_lost_inside = TimeseriesIndex("lost vehicles inside zoi", inputs.fraction_lost * inputs.ts_starting)
        total_lost_inside = Index("total lost vehicles inside zoi", number_lost_inside.sum())

        modified_starting = TimeseriesIndex(
            "modified starting",
            graph.piecewise(
                (
                    (inputs.i_p_fraction_exempted + inputs.fraction_rigid) * inputs.ts_starting,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (inputs.ts_starting + number_time_shifted_inside, True),
            ),
        )

        # Area euro-class split: fleet-only formula (starting also affected)
        modified_euro_class_split = TimeseriesEclIndex(
            "modified split euro levels",
            graph.piecewise(
                (
                    (
                        (inputs.i_p_fraction_exempted * inputs.euro_class_split)
                        + inputs.fraction_rigid_euro_class
                    )
                    / (inputs.i_p_fraction_exempted + inputs.fraction_rigid),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (inputs.euro_class_split, True),
            )
    )

        return (
            AreaModifiedFlowsModel.Outputs(
                modified_inflow=modified_inflow,
                total_modified_inflow=total_modified_inflow,
                modified_starting=modified_starting,
                modified_euro_class_split=modified_euro_class_split,
            ),
            AreaModifiedFlowsModel.Expose(
                number_anticipating_inside=number_anticipating_inside,
                total_anticipating_inside=total_anticipating_inside,
                number_anticipated_inside=number_anticipated_inside,
                total_anticipated_inside=total_anticipated_inside,
                number_postponing_inside=number_postponing_inside,
                total_postponing_inside=total_postponing_inside,
                number_postponed_inside=number_postponed_inside,
                total_postponed_inside=total_postponed_inside,
                number_time_shifted_inside=number_time_shifted_inside,
                total_time_shifted_inside=total_time_shifted_inside,
                number_mode_shifted_inside=number_mode_shifted_inside,
                total_mode_shifted_inside=total_mode_shifted_inside,
                number_lost_inside=number_lost_inside,
                total_lost_inside=total_lost_inside,
            )
        )


# ---------------------------------------------------------------------------
# [V.5] RevenueModel — ModelVariant("RevenueModel", selector=policy)
# ---------------------------------------------------------------------------

@define("AreaRevenue")
class AreaRevenueModel(Model):
    """Revenue computation under area-based policy.

    Under area policy, both *starting* and *inflow* is subject to pricing.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`AreaRevenueModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: Index
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: Index

    @outputs
    class Outputs:
        """Outputs of :class:`AreaRevenueModel`."""

        number_paying: TimeseriesIndex
        total_paying: Index
        modified_avg_cost: Index
        total_paid: Index


    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        number_paying = TimeseriesIndex(
            "paying vehicles",
            graph.piecewise(
                (inputs.fraction_rigid * (inputs.ts_inflow + inputs.ts_starting), (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time)),
                (0, True),
            ),
        )
        total_paying = Index("total vehicles paying", number_paying.sum())

        ts_modified_avg_cost = TimeseriesIndex(
            "ts modified average cost with respect to the vehicles paying",
            graph.piecewise(
                ((inputs.i_p_cost * inputs.fraction_rigid_euro_class).sum(axis=ECL_AXIS) 
                 / inputs.fraction_rigid, inputs.fraction_rigid > 0),
                (0.0, True),
            ),
        )
        ts_paid = TimeseriesIndex("paid fees", number_paying * ts_modified_avg_cost)
        total_paid = Index("total paid fees", ts_paid.sum())

        modified_avg_cost = Index(
            "modified average cost with respect to the vehicles paying",
            graph.piecewise(
                    (total_paid / total_paying, total_paying > 0),
                    (0.0, True),
                ),
        )

        return AreaRevenueModel.Outputs(
            number_paying=number_paying,
            total_paying=total_paying,
            modified_avg_cost=modified_avg_cost,
            total_paid=total_paid,
        )


@define("CordonRevenue")
class CordonRevenueModel(Model):
    """Revenue computation under cordon-based policy.

    Under cordon policy, only vehicle *inflow* is subject to pricing.
    Starting vehicles inside the ZOI are NOT taxed.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`CordonRevenueModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: Index
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: Index

    @outputs
    class Outputs:
        """Outputs of :class:`CordonRevenueModel`."""

        number_paying: TimeseriesIndex
        total_paying: Index
        modified_avg_cost: Index
        total_paid: Index


    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        number_paying = TimeseriesIndex(
            "paying vehicles",
            graph.piecewise(
                (inputs.fraction_rigid * inputs.ts_inflow, (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time)),
                (0, True),
            ),
        )
        total_paying = Index("total vehicles paying", number_paying.sum())

        ts_modified_avg_cost = TimeseriesIndex(
            "ts modified average cost with respect to the vehicles paying",
            graph.piecewise(
                ((inputs.i_p_cost * inputs.fraction_rigid_euro_class).sum(axis=ECL_AXIS) / inputs.fraction_rigid, inputs.fraction_rigid > 0),
                (0.0, True),
            ),
        )
        ts_paid = TimeseriesIndex("paid fees", number_paying * ts_modified_avg_cost)
        total_paid = Index("total paid fees", ts_paid.sum())

        modified_avg_cost = Index(
            "modified average cost with respect to the vehicles paying",
            graph.piecewise(
                    (total_paid / total_paying, total_paying > 0),
                    (0.0, True),
                ),
        )

        return CordonRevenueModel.Outputs(
            number_paying=number_paying,
            total_paying=total_paying,
            modified_avg_cost=modified_avg_cost,
            total_paid=total_paid,
        )
