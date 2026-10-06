from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from civic_digital_twins.dt_model import Index
from civic_digital_twins.dt_model.engine.frontend import graph


# NOTE
# policies are defined through classes
# abstact signature of ALL the Policy classes (see below)
class Policy(ABC):
    @abstractmethod
    def required_params(self) -> set[str]: ...

    @abstractmethod
    def apply(
        self,
        indicators_by_area,
        params: dict[str, Index],
        sensibility_weights: dict,
        *,
        k_scale: float = 1.0,
        accept_zone_by_area: dict[int, float] | None = None,
        vulnerability_by_area: dict | None = None,
        exempt_share_by_area: dict[int, float] | None = None,
    ) -> dict: ...

    def validate(self, params: dict[str, Index]) -> None:
        missing = self.required_params() - params.keys()
        if missing:
            raise ValueError(f"Missing policy params: {missing}")


# not available as graph.minimum (copy-pasted from maximum)
def _sym_min(a, b) -> graph.Node:
    a_node = graph.ensure_node(a)
    b_node = graph.ensure_node(b)
    return graph.where(a_node <= b_node, a_node, b_node)

# graph.maximum wrapper
def _sym_max(a, b) -> graph.Node:
    return graph.maximum(graph.ensure_node(a), graph.ensure_node(b))


def _sym_clip_overlap(ztl_start, ztl_end, start_hour: float, end_hour: float) -> graph.Node:
    clipped_end = _sym_min(ztl_end, end_hour)
    clipped_start = _sym_max(ztl_start, start_hour)
    raw_overlap = clipped_end - clipped_start
    return _sym_max(raw_overlap, 0.0)


# [elderly, minors] policy class
@dataclass(frozen=True)
class CareTimePolicy(Policy):
    care_intervals: tuple[tuple[float, float], ...] = ()
    overlap_exponent: float = 0.7
    overlap_fallback: float = 0.2

    def required_params(self) -> set[str]:
        return {
            "ztl_start", "ztl_end",
            "avail_postpone", "avail_anticipate",
            "median_postpone", "median_anticipate",
        }

    def apply(
        self,
        indicators_by_area,
        params: dict[str, Index],
        sensibility_weights: dict,
        *,
        k_scale: float = 1.0,
        accept_zone_by_area: dict[int, float] | None = None,
        vulnerability_by_area: dict | None = None,
        exempt_share_by_area: dict[int, float] | None = None,
    ) -> dict:
        ztl_start = params["ztl_start"]
        ztl_end = params["ztl_end"]

        if self.care_intervals:
            total = float(sum(end_h - start_h for start_h, end_h in self.care_intervals))
            overlap_parts = [
                _sym_clip_overlap(ztl_start, ztl_end, float(s), float(e))
                for s, e in self.care_intervals
            ]
            overlap_node: graph.Node = graph.constant(0.0)
            for part in overlap_parts:
                overlap_node = overlap_node + part

            overlap_raw = overlap_node / graph.constant(total) if total > 0 else graph.constant(0.0)

            overlap_factor: graph.Node = graph.where(
                overlap_raw > graph.constant(0.0),
                overlap_raw ** graph.constant(self.overlap_exponent),
                graph.constant(self.overlap_fallback),
            )
        else:
            overlap_factor = graph.constant(self.overlap_fallback)

        avail_postpone = params["avail_postpone"]
        avail_anticipate = params["avail_anticipate"]
        median_postpone = params["median_postpone"]
        median_anticipate = params["median_anticipate"]

        flexibility = (graph.ensure_node(avail_postpone) + graph.ensure_node(avail_anticipate)) / graph.constant(2.0)
        median_hours = (graph.ensure_node(median_postpone) + graph.ensure_node(median_anticipate)) / graph.constant(2.0)

        # stress factor computation
        stress_factor = overlap_factor * (graph.constant(1.0) - flexibility) * median_hours

        # apply to indicators
        modified_by_area: dict = {}
        for area, year_map in indicators_by_area.items():
            for year, indicator_map in year_map.items():
                if area not in vulnerability_by_area or year not in vulnerability_by_area[area]:
                    continue

                vulnerability = vulnerability_by_area[area][year]
                modified_by_area.setdefault(area, {}).setdefault(year, {})

                for indicator_name, baseline_index in indicator_map.items():
                    if indicator_name not in sensibility_weights:
                        modified_by_area[area][year][f"I_modified_{indicator_name}"] = baseline_index
                        continue

                    sensitivity = graph.ensure_node(sensibility_weights[indicator_name])
                    delta = graph.constant(k_scale) * stress_factor
                    modified_node = graph.ensure_node(baseline_index) * (
                        graph.constant(1.0) + delta * (sensitivity * graph.ensure_node(vulnerability))
                    )
                    modified_index = Index(
                        f"I_modified_{indicator_name}[{area},{year}]",
                        modified_node,
                    )
                    modified_by_area[area][year][f"I_modified_{indicator_name}"] = modified_index

        return modified_by_area


# [income] policy class
@dataclass(frozen=True)
class EconomicBurdenPolicy(Policy):
    income_threshold: float

    def required_params(self) -> set[str]:
        return {"ticket_costs", "euro_class_split"}

    def apply(
        self,
        indicators_by_area,
        params: dict[str, Index],
        sensibility_weights: dict,
        *,
        k_scale: float = 1.0,
        accept_zone_by_area: dict[int, float] | None = None,
        vulnerability_by_area: dict | None = None,
        exempt_share_by_area: dict[int, float] | None = None,
    ) -> dict:
        if accept_zone_by_area is None:
            raise ValueError("EconomicBurdenPolicy requires accept_zone_by_area.")

        from .core.burden import compute_ticket_effect
        from .core.ticket_indicators import apply_ticket_to_indicators

        ticket_effect_by_area = compute_ticket_effect(
            accept_zone_by_area,
            params["ticket_costs"],
            params["euro_class_split"],
            exempt_share_by_area=exempt_share_by_area,
        )

        return apply_ticket_to_indicators(
            indicators_by_area,
            ticket_effect_by_area,
            sensibility_weights,
            k_scale=k_scale,
        )
