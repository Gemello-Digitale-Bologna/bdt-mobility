from civic_digital_twins.dt_model import Index
from civic_digital_twins.dt_model.engine.frontend import graph


def apply_ticket_to_indicators(
    indicators_by_area: dict,
    ticket_effect_by_area: dict[int, Index],
    sensibility_weights: dict,
    *,
    k_scale: float = 1.0,
) -> dict:
    modified_by_area: dict = {}

    for area, year_map in indicators_by_area.items():
        ticket_effect = ticket_effect_by_area.get(int(area))
        if ticket_effect is None:
            continue

        for year, indicator_map in year_map.items():
            modified_by_area.setdefault(area, {}).setdefault(year, {})

            for indicator_name, baseline_index in indicator_map.items():
                if indicator_name not in sensibility_weights:
                    modified_by_area[area][year][f"I_modified_{indicator_name}"] = baseline_index
                    continue

                sensitivity = graph.ensure_node(sensibility_weights[indicator_name])
                delta = graph.constant(k_scale) * graph.ensure_node(ticket_effect)
                modified_node = graph.ensure_node(baseline_index) * (
                    graph.constant(1.0) + delta * sensitivity
                )
                modified_by_area[area][year][f"I_modified_{indicator_name}"] = Index(
                    f"I_modified_{indicator_name}[{area},{year}]",
                    modified_node,
                )

    return modified_by_area
