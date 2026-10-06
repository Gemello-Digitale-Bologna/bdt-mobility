"""Time-shift sub-models — [V.3] ShiftFormula variants, [D] TimeShiftModel."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from civic_digital_twins.dt_model import (
    Index, TimeseriesIndex,
    Model, ModelVariant,
    define, inputs, outputs, config
)
from civic_digital_twins.dt_model.engine.frontend import graph

from .timeseries import P_RECORD_HEADWAY


# ---------------------------------------------------------------------------
# [V.3] ShiftFormula — variants of ModelVariant("ShiftFormula", selector=time_shift_strategy)
# ---------------------------------------------------------------------------

@define("FlexibleShift")
class FlexibleShiftFormula(Model):
    """Closed-form exponential redistribution of anticipating/postponing vehicles.

    Used when ``time_shift_strategy`` is ``"flexible"``.

    Parameters
    ----------
    number_anticipating :
        Timeseries of anticipating vehicle counts.
    total_anticipating :
        Scalar total of anticipating vehicles.
    number_postponing :
        Timeseries of postponing vehicle counts.
    total_postponing :
        Scalar total of postponing vehicles.
    delta_before_start :
        Time before policy window start.
    delta_after_end :
        Time after policy window end.
    delta_from_start :
        Unused in this variant (present for shared Inputs contract).
    delta_to_end :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_anticipation :
        Median anticipation shift (hours).
    i_b_p50_postponement :
        Median postponement shift (hours).
    i_b_p50_anticipating :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_postponing :
        Unused in this variant (present for shared Inputs contract).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`FlexibleShiftFormula`."""

        number_anticipating: TimeseriesIndex
        total_anticipating: Index
        number_postponing: TimeseriesIndex
        total_postponing: Index
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index

    @outputs
    class Outputs:
        """Outputs of :class:`FlexibleShiftFormula`."""

        number_anticipated: TimeseriesIndex
        total_anticipated: Index
        number_postponed: TimeseriesIndex
        total_postponed: Index

    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        number_anticipated = TimeseriesIndex(
            "anticipated vehicles",
            (
                np.e ** (-(inputs.delta_before_start - P_RECORD_HEADWAY) / inputs.i_b_p50_anticipation * np.log(2))
                - np.e ** (-inputs.delta_before_start / inputs.i_b_p50_anticipation * np.log(2))
            )
            * inputs.total_anticipating,
        )
        total_anticipated = Index("total vehicles anticipated", number_anticipated.sum())
        number_postponed = TimeseriesIndex(
            "postponed vehicles",
            (
                np.e ** (-(inputs.delta_after_end - P_RECORD_HEADWAY) / inputs.i_b_p50_postponement * np.log(2))
                - np.e ** (-inputs.delta_after_end / inputs.i_b_p50_postponement * np.log(2))
            )
            * inputs.total_postponing,
        )
        total_postponed = Index("total vehicles postponed", number_postponed.sum())

        return FlexibleShiftFormula.Outputs(
            number_anticipated=number_anticipated,
            total_anticipated=total_anticipated,
            number_postponed=number_postponed,
            total_postponed=total_postponed,
        )


@define("FixedShift")
class FixedShiftFormula(Model):
    """Numerically redistributed anticipating/postponing vehicles.

    Used when ``time_shift_strategy`` is ``"fixed"``.  Uses the
    ``ts_anticipate`` and ``ts_postpone`` graph functions.

    Parameters
    ----------
    number_anticipating :
        Timeseries of anticipating vehicle counts.
    total_anticipating :
        Scalar total (unused in this variant; present for contract).
    number_postponing :
        Timeseries of postponing vehicle counts.
    total_postponing :
        Scalar total (unused in this variant; present for contract).
    delta_before_start :
        Unused in this variant (present for shared Inputs contract).
    delta_after_end :
        Unused in this variant (present for shared Inputs contract).
    delta_from_start :
        Used by ``ts_anticipate``.
    delta_to_end :
        Used by ``ts_postpone``.
    i_b_p50_anticipation :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_postponement :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_anticipating :
        Distribution parameter for ``ts_anticipate``.
    i_b_p50_postponing :
        Distribution parameter for ``ts_postpone``.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`FixedShiftFormula`."""

        number_anticipating: TimeseriesIndex
        total_anticipating: Index
        number_postponing: TimeseriesIndex
        total_postponing: Index
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index

    @outputs
    class Outputs:
        """Outputs of :class:`FixedShiftFormula`."""

        number_anticipated: TimeseriesIndex
        total_anticipated: Index
        number_postponed: TimeseriesIndex
        total_postponed: Index

    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        number_anticipated = TimeseriesIndex(
            "vehicles anticipated",
            graph.function_call(
                "ts_anticipate",
                inputs.number_anticipating.node,
                inputs.delta_from_start.node,
                inputs.i_b_p50_anticipating.node,
            ),
        )
        total_anticipated = Index("total vehicles anticipated", number_anticipated.sum())
        number_postponed = TimeseriesIndex(
            "vehicles postponed",
            graph.function_call(
                "ts_postpone",
                inputs.number_postponing.node,
                inputs.delta_to_end.node,
                inputs.i_b_p50_postponing.node,
            ),
        )
        total_postponed = Index("total vehicles postponed", number_postponed.sum())

        return FixedShiftFormula.Outputs(
            number_anticipated=number_anticipated,
            total_anticipated=total_anticipated,
            number_postponed=number_postponed,
            total_postponed=total_postponed,
        )


# ---------------------------------------------------------------------------
# [D] TimeShiftModel
# ---------------------------------------------------------------------------


@define("TimeShift")
class TimeShiftModel(Model):
    """Translates behaviour fractions into vehicle shift counts.

    Owns the :class:`ShiftFormula` :class:`ModelVariant` internally.
    Policy-agnostic: the inside-shift computation for area-based
    ``modified_starting`` is the responsibility of
    :class:`AreaModifiedFlowsModel`.

    Parameters
    ----------
    time_shift_strategy :
        Selector for the internal :class:`ShiftFormula` variant.
    ts_inflow :
        Baseline inflow timeseries.
    fraction_anticipating, fraction_postponing :
        Wired from :class:`BehaviorModel`.
    fraction_mode_shifted, fraction_lost :
        Wired from :class:`BehaviorModel`.
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`.
    delta_before_start, delta_after_end :
        Wired from :class:`PolicyWindowModel`.
    i_b_p50_anticipating, i_b_p50_postponing :
        Behavioural 50%-likelihood parameters.
    i_b_p50_anticipation, i_b_p50_postponement :
        Behavioural median-shift parameters.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`TimeShiftModel`."""

        ts_inflow: TimeseriesIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index

    @outputs
    class Outputs:
        """Outputs of :class:`TimeShiftModel`."""

        number_anticipating: TimeseriesIndex
        total_anticipating: Index
        number_anticipated: TimeseriesIndex
        total_anticipated: Index
        number_postponing: TimeseriesIndex
        total_postponing: Index
        number_postponed: TimeseriesIndex
        total_postponed: Index
        number_time_shifted: TimeseriesIndex
        total_time_shifted: Index
        number_mode_shifted: TimeseriesIndex
        total_mode_shifted: Index
        number_lost: TimeseriesIndex
        total_lost: Index

    @config
    class Config:
        time_shift_strategy: str


    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> Outputs:

        number_anticipating = TimeseriesIndex("anticipating vehicles", inputs.fraction_anticipating * inputs.ts_inflow)
        total_anticipating = Index("total anticipating vehicles", number_anticipating.sum())
        number_postponing = TimeseriesIndex("postponing vehicles", inputs.fraction_postponing * inputs.ts_inflow)
        total_postponing = Index("total postponing vehicles", number_postponing.sum())

        _shift_common = dict(
            number_anticipating=number_anticipating,
            total_anticipating=total_anticipating,
            number_postponing=number_postponing,
            total_postponing=total_postponing,
            delta_before_start=inputs.delta_before_start,
            delta_after_end=inputs.delta_after_end,
            delta_from_start=inputs.delta_from_start,
            delta_to_end=inputs.delta_to_end,
            i_b_p50_anticipation=inputs.i_b_p50_anticipation,
            i_b_p50_postponement=inputs.i_b_p50_postponement,
            i_b_p50_anticipating=inputs.i_b_p50_anticipating,
            i_b_p50_postponing=inputs.i_b_p50_postponing,
        )
        shift_formula = ModelVariant(
            "ShiftFormula",
            variants={
                "flexible": FlexibleShiftFormula(
                    inputs=FlexibleShiftFormula.Inputs(**_shift_common)
                ),
                "fixed": FixedShiftFormula(
                    inputs=FixedShiftFormula.Inputs(**_shift_common)
                ),
            },
            selector=config.time_shift_strategy,
        )

        number_anticipated: TimeseriesIndex = shift_formula.outputs.number_anticipated
        total_anticipated: Index = shift_formula.outputs.total_anticipated
        number_postponed: TimeseriesIndex = shift_formula.outputs.number_postponed
        total_postponed: Index = shift_formula.outputs.total_postponed

        number_time_shifted = TimeseriesIndex("time-shifted vehicles", number_anticipated + number_postponed)
        total_time_shifted = Index("total vehicles shifted", total_anticipated + total_postponed)
        number_mode_shifted = TimeseriesIndex("mode-shifted vehicles", inputs.fraction_mode_shifted * inputs.ts_inflow)
        total_mode_shifted = Index("total mode shifted vehicles", number_mode_shifted.sum())
        number_lost = TimeseriesIndex("lost vehicles", inputs.fraction_lost * inputs.ts_inflow)
        total_lost = Index("total lost vehicles", number_lost.sum())

        return TimeShiftModel.Outputs(
            number_anticipating=number_anticipating,
            total_anticipating=total_anticipating,
            number_anticipated=number_anticipated,
            total_anticipated=total_anticipated,
            number_postponing=number_postponing,
            total_postponing=total_postponing,
            number_postponed=number_postponed,
            total_postponed=total_postponed,
            number_time_shifted=number_time_shifted,
            total_time_shifted=total_time_shifted,
            number_mode_shifted=number_mode_shifted,
            total_mode_shifted=total_mode_shifted,
            number_lost=number_lost,
            total_lost=total_lost,
        )
