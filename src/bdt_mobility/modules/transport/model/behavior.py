"""Demand-response sub-models — [C] PolicyWindowModel, [V.1] modal-shift variants, [V.2] behavior variants."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from civic_digital_twins.dt_model import (
    Index, TimeseriesIndex, ConstIndex,
    Model, ModelVariant, 
    define, inputs, outputs, expose, config,
)
from civic_digital_twins.dt_model.engine.frontend import graph

from .timeseries import P_DWELL_TIME_ZOI, P_RECORD_HEADWAY
from bdt_mobility.model.axis import (
    EclIndex,
    ZonePtIndex,
    ZonePtTimeseriesIndex,
    TimeseriesEclIndex,
    ECL_AXIS, 
    ZONE_PT_AXIS, 
    TIME_AXIS
)

I_B_PT_BASELINE = -1.28


# ---------------------------------------------------------------------------
# [C] PolicyWindowModel
# ---------------------------------------------------------------------------

@define("PolicyWindow")
class PolicyWindowModel(Model):
    """Time-delta indexes relative to the policy window [start_time, end_time].

    Parameters
    ----------
    ts :
        Shared time-range :class:`TimeseriesIndex`.
    i_p_start_time :
        Policy start time parameter (seconds from midnight).
    i_p_end_time :
        Policy end time parameter (seconds from midnight).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`PolicyWindowModel`."""

        ts: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index

    @outputs
    class Outputs:
        """Outputs of :class:`PolicyWindowModel`."""

        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex

    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        delta_from_start = TimeseriesIndex(
            "delta time from start",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_start_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts >= inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )
        delta_to_end = TimeseriesIndex(
            "delta time to end",
            graph.piecewise(
                (
                    (inputs.i_p_end_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts <= inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )
        delta_before_start = TimeseriesIndex(
            "delta time before start",
            graph.piecewise(
                (
                    (inputs.i_p_start_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts < inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )
        delta_after_end = TimeseriesIndex(
            "delta time after end",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_end_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts > inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )

        return PolicyWindowModel.Outputs(
            delta_from_start=delta_from_start,
            delta_to_end=delta_to_end,
            delta_before_start=delta_before_start,
            delta_after_end=delta_after_end,
        )


# ---------------------------------------------------------------------------
# [V.1] ModalShiftModel — variants of ModelVariant("ModalShift", selector=modal_shift_option)
# ---------------------------------------------------------------------------


@define("LptModalShift")
class LptModalShiftModel(Model):
    """Modal-shift to local public transport (LPT) fractions computed via a logit model.

    Used when ``modal_shift_option`` is ``"lpt"`` or ``"lpt+active"``.

    Parameters
    ----------
    i_b_pt_capillarity :
        Behavioural importance of PT capillarity.
    i_b_pt_frequency :
        Behavioural importance of PT frequency.
    i_b_pt_cost :
        Behavioural importance of PT cost.
    i_b_pt_time :
        Behavioural importance of PT time difference.
    i_p_pt_capillarity_modification :
        Policy modification multiplier for capillarity.
    i_p_pt_frequency_modification :
        Policy modification multiplier for frequency.
    i_p_pt_cost_modification :
        Policy modification multiplier for cost.
    i_p_pt_time_modification :
        Policy modification multiplier for time.
    pt_zones :
        Array of zone ids considered for the pt.
    pt_capillarity :
        Per-zone PT capillarity values.
    pt_frequency :
        Per-zone PT frequency values.
    pt_avg_cost :
        Per-zone PT average cost values.
    pt_avg_time_diff :
        Per-zone PT average time-difference values.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`LptModalShiftModel`."""

        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index
        pt_capillarity: ZonePtIndex
        pt_frequency: ZonePtIndex
        pt_avg_cost: ZonePtIndex
        pt_avg_time_diff: ZonePtIndex

    @outputs
    class Outputs:
        """Outputs of :class:`LptModalShiftModel`."""

        fraction_p_mode_shifted_zone: ZonePtIndex

    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:

        fraction_p_mode_shifted_zone = ZonePtIndex(
            "fraction of possibly mode-shifted vehicles in pt zones",
            1
            / (
                1
                + np.e
                ** (
                    -(
                        I_B_PT_BASELINE
                        + inputs.i_b_pt_capillarity
                        * (1 + inputs.i_p_pt_capillarity_modification)
                        * inputs.pt_capillarity
                        + inputs.i_b_pt_frequency * (1 + inputs.i_p_pt_frequency_modification) * inputs.pt_frequency
                        + inputs.i_b_pt_cost * (1 + inputs.i_p_pt_cost_modification) * inputs.pt_avg_cost
                        + inputs.i_b_pt_time * (1 + inputs.i_p_pt_time_modification) * inputs.pt_avg_time_diff
                    )
                )
            )
        )

        return LptModalShiftModel.Outputs(
            fraction_p_mode_shifted_zone=fraction_p_mode_shifted_zone    
        )


@define("ZeroModalShift")
class ZeroModalShiftModel(Model):
    """Modal-shift fractions set to zero (no TPM).

    Used when ``modal_shift_option`` is ``"no"`` or ``"active"``.

    Declares the same :class:`Inputs` contract as :class:`LptModalShiftModel`
    (fields are received but unused), ensuring the :class:`ModelVariant`
    contract is symmetric across both variants.

    Parameters
    ----------
    i_b_pt_capillarity :
        Behavioural importance of PT capillarity (declared, unused).
    i_b_pt_frequency :
        Behavioural importance of PT frequency (declared, unused).
    i_b_pt_cost :
        Behavioural importance of PT cost (declared, unused).
    i_b_pt_time :
        Behavioural importance of PT time difference (declared, unused).
    i_p_pt_capillarity_modification :
        Policy modification multiplier for capillarity (declared, unused).
    i_p_pt_frequency_modification :
        Policy modification multiplier for frequency (declared, unused).
    i_p_pt_cost_modification :
        Policy modification multiplier for cost (declared, unused).
    i_p_pt_time_modification :
        Policy modification multiplier for time (declared, unused).
    pt_zones :
        Array of zone ids considered for the pt.
    pt_capillarity :
        Per-zone PT capillarity values (declared, unused).
    pt_frequency :
        Per-zone PT frequency values (declared, unused).
    pt_avg_cost :
        Per-zone PT average cost values (declared, unused).
    pt_avg_time_diff :
        Per-zone PT average time-difference values (declared, unused).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`LptModalShiftModel`."""

        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index
        pt_capillarity: ZonePtIndex
        pt_frequency: ZonePtIndex
        pt_avg_cost: ZonePtIndex
        pt_avg_time_diff: ZonePtIndex

    @outputs
    class Outputs:
        """Outputs of :class:`LptModalShiftModel`."""

        fraction_p_mode_shifted_zone: ZonePtIndex


    def compute(
        self,
        inputs: Inputs
    ) -> Outputs:
        zero_index = ConstIndex("zero value", 0.0)
        fraction_p_mode_shifted_zone = ZonePtIndex(
            "fraction of possibly mode-shifted vehicles in pt zone", 
            zero_index.broadcast(ZONE_PT_AXIS),
        )

        return ZeroModalShiftModel.Outputs(
            fraction_p_mode_shifted_zone=fraction_p_mode_shifted_zone
        )


# ---------------------------------------------------------------------------
# [V.2] BehaviorModel — ModelVariant("BehaviorModel", selector=decision_strategy)
# ---------------------------------------------------------------------------


@define("ParallelBehavior")
class ParallelBehaviorModel(Model):
    """Behaviour fractions using the parallel (competitive) decision model.

    Vehicles independently decide whether to be rigid, anticipate, postpone,
    or mode-shift; joint probabilities are computed via :func:`ts_b_choose`.

    Parameters
    ----------
    policy :
        ``"cordon-based"`` or ``"area-based"`` — controls the anticipation
        probability formula.
    modal_shift_option :
        Selector for the internal :class:`ModalShiftModel` variant.
    pt_zones :
        Considered zone ids.
    zone_proportions :
        Per-zone weights for ``ts_w_sum_list``.
    euro_class_split :
        Fleet euro-class proportions.
    pt_capillarity, pt_frequency, pt_avg_cost, pt_avg_time_diff :
        Per-zone PT metrics (length must match ``pt_zones``).
    ts, delta_from_start, delta_to_end :
        Wired from upstream models.
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_cost, i_b_p50_anticipating, i_b_p50_postponing :
        Behavioural parameter indexes.
    i_b_pt_* / i_p_pt_* :
        PT behavioural and policy parameter indexes.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ParallelBehaviorModel`."""

        ts: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: EclIndex
        i_b_p50_cost: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index
        euro_class_split: EclIndex
        pt_capillarity: ZonePtIndex
        pt_frequency: ZonePtIndex
        pt_avg_cost: ZonePtIndex
        pt_avg_time_diff: ZonePtIndex
        pt_zone_proportions: ZonePtIndex

    @outputs
    class Outputs:
        """Outputs of :class:`ParallelBehaviorModel`."""

        fraction_rigid: TimeseriesIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        fraction_rigid_euro_class: TimeseriesEclIndex

    @expose
    class Expose:
        """Inspectable intermediates of :class:`ParallelBehaviorModel`."""

        fraction_p_rigid_euro_class: EclIndex
        fraction_p_rigid: Index
        fraction_p_anticipating: TimeseriesIndex
        fraction_p_postponing: TimeseriesIndex
        fraction_rigid_zone: ZonePtTimeseriesIndex
        fraction_anticipating_zone: ZonePtTimeseriesIndex
        fraction_postponing_zone: ZonePtTimeseriesIndex
        fraction_mode_shifted_zone: ZonePtTimeseriesIndex
        fraction_lost_zone: ZonePtTimeseriesIndex

    @config
    class Config:
        policy: str
        modal_shift_option: str


    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> tuple[Outputs, Expose]:

        # Possibly-rigid fractions per euro class
        fraction_p_rigid_euro_class = EclIndex(
            "fraction of possibly rigid vehicles per euro level",
            graph.piecewise(
                (0.0, inputs.i_b_p50_cost == 0),
                (np.e ** (inputs.i_p_cost / inputs.i_b_p50_cost * np.log(0.5)), True),
            ),
        )

        fraction_p_rigid = Index(
            "fraction of possibly rigid vehicles", 
            (fraction_p_rigid_euro_class * inputs.euro_class_split).sum(axis=ECL_AXIS)
        )

        # Possibly-anticipating fraction — formula differs by policy
        if config.policy == "cordon-based":
            fraction_p_anticipating = Index(
                "fraction of possibly anticipating vehicles",
                graph.piecewise(
                    (0.0, inputs.i_b_p50_anticipating == 0),
                    (
                        np.e
                        ** (
                            (inputs.delta_from_start + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_anticipating * np.log(0.5)
                        ),
                        True,
                    ),
                ),
            )
        else:  # area-based
            anticipate_reduction = Index(
                "anticipation reduction",
                (np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 / P_DWELL_TIME_ZOI))
                / (1 - np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 - 1 / P_DWELL_TIME_ZOI)),
            )
            fraction_p_anticipating = Index(
                "fraction of possibly anticipating vehicles",
                graph.piecewise(
                    (0.0, inputs.i_b_p50_anticipating == 0),
                    (
                        np.e
                        ** (
                            (inputs.delta_from_start + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_anticipating * np.log(0.5)
                        )
                        * anticipate_reduction,
                        True,
                    ),
                ),
            )

        fraction_p_postponing = Index(
            "fraction of possibly postponing vehicles",
            graph.piecewise(
                (0.0, inputs.i_b_p50_postponing == 0),
                (
                    np.e ** ((inputs.delta_to_end + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_postponing * np.log(0.5)),
                    True,
                ),
            ),
        )

        # Modal shift variant
        ms_key = "lpt" if config.modal_shift_option in ("lpt", "lpt+active") else "no"
        _ms_common = dict(
            i_b_pt_capillarity=inputs.i_b_pt_capillarity,
            i_b_pt_frequency=inputs.i_b_pt_frequency,
            i_b_pt_cost=inputs.i_b_pt_cost,
            i_b_pt_time=inputs.i_b_pt_time,
            i_p_pt_capillarity_modification=inputs.i_p_pt_capillarity_modification,
            i_p_pt_frequency_modification=inputs.i_p_pt_frequency_modification,
            i_p_pt_cost_modification=inputs.i_p_pt_cost_modification,
            i_p_pt_time_modification=inputs.i_p_pt_time_modification,
            pt_capillarity=inputs.pt_capillarity,
            pt_frequency=inputs.pt_frequency,
            pt_avg_cost=inputs.pt_avg_cost,
            pt_avg_time_diff=inputs.pt_avg_time_diff,
        )
        modal_shift: ModelVariant = ModelVariant(
            "ModalShift",
            variants={
                "lpt": LptModalShiftModel(
                    inputs=LptModalShiftModel.Inputs(**_ms_common)
                ),
                "no": ZeroModalShiftModel(
                    inputs=ZeroModalShiftModel.Inputs(**_ms_common)
                ),
            },
            selector=ms_key,
        )
        fraction_p_mode_shifted_zone: Index = modal_shift.outputs.fraction_p_mode_shifted_zone

        # Per-zone fractions using ts_b_choose (competitive allocation)
        fraction_rigid_zone = ZonePtTimeseriesIndex(
            "fraction of rigid vehicles in pt zones",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                        fraction_p_postponing.node,
                        fraction_p_mode_shifted_zone.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (1 - inputs.i_p_fraction_exempted, True),
            )
        )
        fraction_anticipating_zone = ZonePtTimeseriesIndex(
            "fraction of anticipating vehicles in pt zones",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_anticipating.node,
                        fraction_p_rigid.node,
                        fraction_p_postponing.node,
                        fraction_p_mode_shifted_zone.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
                )
        )
        fraction_postponing_zone = ZonePtTimeseriesIndex(
            "fraction of postponing vehicles in pt zones",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_postponing.node,
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                        fraction_p_mode_shifted_zone.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            )
        )
        fraction_mode_shifted_zone = ZonePtTimeseriesIndex(
            "fraction of mode-shifted vehicles in pt zones",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_mode_shifted_zone.node,
                        fraction_p_postponing.node,
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            )
        )
        fraction_lost_zone = ZonePtTimeseriesIndex(
            "fraction of lost vehicles in pt zones",
            graph.piecewise(
                (
                    (1 - fraction_p_rigid)
                    * (1 - fraction_p_anticipating)
                    * (1 - fraction_p_postponing)
                    * (1 - fraction_p_mode_shifted_zone)
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            )
        )

        # Zone-weighted aggregate fractions
        fraction_rigid = TimeseriesIndex(
            "fraction of rigid vehicles", 
            (fraction_rigid_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )
    
        # Post-hoc euro-class rescaling (parallel only)
        fraction_rigid_euro_class = TimeseriesEclIndex(
            "fraction rigid vehicles per euro level",
            graph.piecewise(
                (
                    fraction_rigid
                    * (fraction_p_rigid_euro_class * inputs.euro_class_split)
                    / fraction_p_rigid,
                    fraction_p_rigid > 0,
                ),
                (fraction_rigid * inputs.euro_class_split, True),
            )
        )

        fraction_anticipating = TimeseriesIndex(
            "fraction of anticipating vehicles",
            (fraction_anticipating_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_postponing = TimeseriesIndex(
            "fraction of postponing vehicles",
            (fraction_postponing_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_mode_shifted = TimeseriesIndex(
            "fraction of mode-shifted vehicles",
            (fraction_mode_shifted_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_lost = TimeseriesIndex(
            "fraction of lost vehicles",
            (fraction_lost_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )


        return (
            ParallelBehaviorModel.Outputs(
                fraction_rigid=fraction_rigid,
                fraction_anticipating=fraction_anticipating,
                fraction_postponing=fraction_postponing,
                fraction_mode_shifted=fraction_mode_shifted,
                fraction_lost=fraction_lost,
                fraction_rigid_euro_class=fraction_rigid_euro_class,
            ),
            ParallelBehaviorModel.Expose(
                fraction_p_rigid_euro_class=fraction_p_rigid_euro_class,
                fraction_p_rigid=fraction_p_rigid,
                fraction_p_anticipating=fraction_p_anticipating,
                fraction_p_postponing=fraction_p_postponing,
                fraction_rigid_zone=fraction_rigid_zone,
                fraction_anticipating_zone=fraction_anticipating_zone,
                fraction_postponing_zone=fraction_postponing_zone,
                fraction_mode_shifted_zone=fraction_mode_shifted_zone,
                fraction_lost_zone=fraction_lost_zone,
            ),
        )


@define("SequentialBehavior")
class SequentialBehaviorModel(Model):
    """Behaviour fractions using the sequential (chained-subtraction) model.

    Each behaviour type takes its share from the remaining non-allocated
    fraction in a fixed priority order: rigid → anticipating → postponing
    → mode-shifted → lost.

    Parameters
    ----------
    policy :
        ``"cordon-based"`` or ``"area-based"`` — controls the anticipation
        probability formula.
    modal_shift_option :
        Selector for the internal :class:`ModalShiftModel` variant.
    pt_zones :
        External zone ids.
    zone_proportions :
        Per-zone weights for ``ts_w_sum_list``.
    euro_class_split :
        Fleet euro-class proportions.
    pt_capillarity, pt_frequency, pt_avg_cost, pt_avg_time_diff :
        Per-zone PT metrics.
    ts, delta_from_start, delta_to_end :
        Wired from upstream models.
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_cost, i_b_p50_anticipating, i_b_p50_postponing :
        Behavioural parameter indexes.
    i_b_pt_* / i_p_pt_* :
        PT behavioural and policy parameter indexes.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`SequentialBehaviorModel`."""

        ts: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: EclIndex
        i_b_p50_cost: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index
        euro_class_split: EclIndex
        pt_capillarity: ZonePtIndex
        pt_frequency: ZonePtIndex
        pt_avg_cost: ZonePtIndex
        pt_avg_time_diff: ZonePtIndex
        pt_zone_proportions: ZonePtIndex

    @outputs
    class Outputs:
        """Outputs of :class:`SequentialBehaviorModel`."""

        fraction_rigid: TimeseriesIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        fraction_rigid_euro_class: TimeseriesEclIndex

    @expose
    class Expose:
        """Inspectable intermediates of :class:`SequentialBehaviorModel`."""

        fraction_rigid_zone: ZonePtTimeseriesIndex
        fraction_anticipating_zone: ZonePtTimeseriesIndex
        fraction_postponing_zone: ZonePtTimeseriesIndex
        fraction_mode_shifted_zone: ZonePtTimeseriesIndex
        fraction_lost_zone: ZonePtTimeseriesIndex

    @config
    class Config:
        policy: str
        modal_shift_option: str


    def compute(
        self,
        inputs: Inputs,
        config: Config
    ) -> tuple[Outputs, Expose]:

        # Rigid fractions (per euro class, then zone — identical across zones in sequential)
        fraction_rigid_euro_class = TimeseriesEclIndex(
            "fraction rigid vehicles per euro level",
            (graph.piecewise(
                (0, inputs.i_b_p50_cost == 0),
                (
                    (1 - inputs.i_p_fraction_exempted)
                    * np.e ** (inputs.i_p_cost / inputs.i_b_p50_cost * np.log(0.5))
                    * inputs.euro_class_split,
                    True,
                )
            )).broadcast(TIME_AXIS, ECL_AXIS)
        )

        fraction_rigid_zone = ZonePtTimeseriesIndex(
            "rigid vehicles in pt zones",
            fraction_rigid_euro_class.sum(axis=ECL_AXIS).broadcast(ZONE_PT_AXIS, TIME_AXIS)
        ) 

        # Anticipating fractions — formula differs by policy
        if config.policy == "cordon-based":
            fraction_anticipating_zone = ZonePtTimeseriesIndex(
                "anticipating vehicles in pt zones",
                graph.piecewise(
                    (0.0, inputs.i_b_p50_anticipating == 0),
                    (
                        np.e
                        ** (
                            (inputs.delta_from_start + P_RECORD_HEADWAY / 2)
                            / inputs.i_b_p50_anticipating
                            * np.log(0.5)
                        )
                        * (1 - inputs.i_p_fraction_exempted - fraction_rigid_zone),
                        True,
                    )
                )
            )
        else:  # area-based
            anticipate_reduction = Index(
                "anticipation reduction",
                (np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 / P_DWELL_TIME_ZOI))
                / (1 - np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 - 1 / P_DWELL_TIME_ZOI)),
            )
            fraction_anticipating_zone = ZonePtTimeseriesIndex(
                "anticipating vehicles in pt zones",
                graph.piecewise(
                    (0.0, inputs.i_b_p50_anticipating == 0),
                    (
                        np.e
                        ** (
                            (inputs.delta_from_start + P_RECORD_HEADWAY / 2)
                            / inputs.i_b_p50_anticipating
                            * np.log(0.5)
                        )
                        * (1 - inputs.i_p_fraction_exempted - fraction_rigid_zone)
                        * anticipate_reduction,
                        True,
                    ),
                )
            )

        # Postponing fractions (chained)
        fraction_postponing_zone = ZonePtTimeseriesIndex(
            "postponing vehicles in pt zones",
            graph.piecewise(
                (0.0, inputs.i_b_p50_postponing == 0),
                (
                    np.e ** ((inputs.delta_to_end + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_postponing * np.log(0.5))
                    * (1 - inputs.i_p_fraction_exempted - fraction_rigid_zone - fraction_anticipating_zone),
                    True,
                ),
            )
        )

        # Modal shift variant (sequential: chained residual applied inside the formula)
        ms_key = "lpt" if config.modal_shift_option in ("lpt", "lpt+active") else "no"
        _ms_common = dict(
            i_b_pt_capillarity=inputs.i_b_pt_capillarity,
            i_b_pt_frequency=inputs.i_b_pt_frequency,
            i_b_pt_cost=inputs.i_b_pt_cost,
            i_b_pt_time=inputs.i_b_pt_time,
            i_p_pt_capillarity_modification=inputs.i_p_pt_capillarity_modification,
            i_p_pt_frequency_modification=inputs.i_p_pt_frequency_modification,
            i_p_pt_cost_modification=inputs.i_p_pt_cost_modification,
            i_p_pt_time_modification=inputs.i_p_pt_time_modification,
            pt_capillarity=inputs.pt_capillarity,
            pt_frequency=inputs.pt_frequency,
            pt_avg_cost=inputs.pt_avg_cost,
            pt_avg_time_diff=inputs.pt_avg_time_diff,
        )
        modal_shift: ModelVariant = ModelVariant(
            "ModalShift",
            variants={
                "lpt": LptModalShiftModel(
                    inputs=LptModalShiftModel.Inputs(**_ms_common),
                ),
                "no": ZeroModalShiftModel(
                    inputs=ZeroModalShiftModel.Inputs(**_ms_common),
                ),
            },
            selector=ms_key,
        )
        raw_mode_shifted_zone = modal_shift.outputs.fraction_p_mode_shifted_zone

        # Sequential: mode-shifted multiplies the chained residual (within policy window)
        fraction_mode_shifted_zone = ZonePtTimeseriesIndex(
            "fraction of mode-shifted vehicles in pt zones",
            graph.piecewise(
                (
                    raw_mode_shifted_zone
                    * (
                        1
                        - inputs.i_p_fraction_exempted
                        - fraction_rigid_zone
                        - fraction_anticipating_zone
                        - fraction_postponing_zone
                    ),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            )
        )

        fraction_lost_zone = ZonePtTimeseriesIndex(
            "fraction of lost vehicles in pt zones",
            graph.piecewise(
                (
                    1
                    - fraction_rigid_zone
                    - fraction_anticipating_zone
                    - fraction_postponing_zone
                    - inputs.i_p_fraction_exempted
                    - fraction_mode_shifted_zone,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            )
        )

        # Zone-weighted aggregates

        fraction_rigid = TimeseriesIndex(
            "fraction of rigid vehicles", 
            (fraction_rigid_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_anticipating = TimeseriesIndex(
            "fraction of anticipating vehicles",
            (fraction_anticipating_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_postponing = TimeseriesIndex(
            "fraction of postponing vehicles",
            (fraction_postponing_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_mode_shifted = TimeseriesIndex(
            "fraction of mode-shifted vehicles",
            (fraction_mode_shifted_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        fraction_lost = TimeseriesIndex(
            "fraction of lost vehicles",
            (fraction_lost_zone * inputs.pt_zone_proportions).sum(axis=ZONE_PT_AXIS)
        )

        return(
            SequentialBehaviorModel.Outputs(
                fraction_rigid=fraction_rigid,
                fraction_anticipating=fraction_anticipating,
                fraction_postponing=fraction_postponing,
                fraction_mode_shifted=fraction_mode_shifted,
                fraction_lost=fraction_lost,
                fraction_rigid_euro_class=fraction_rigid_euro_class,
            ),
            SequentialBehaviorModel.Expose(
                fraction_rigid_zone=fraction_rigid_zone,
                fraction_anticipating_zone=fraction_anticipating_zone,
                fraction_postponing_zone=fraction_postponing_zone,
                fraction_mode_shifted_zone=fraction_mode_shifted_zone,
                fraction_lost_zone=fraction_lost_zone,
            ),
        )
