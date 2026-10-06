from civic_digital_twins.dt_model import Index
from civic_digital_twins.dt_model.engine.frontend import graph

from bdt_mobility.model.axis import ECL_AXIS, EclIndex


def compute_ticket_effect(
    accept_zone_by_area: dict[int, float],
    ticket_prices: Index,
    euro_class_split: Index,
    exempt_share_by_area: dict[int, float] | None = None,
) -> dict[int, Index]:
    result: dict[int, Index] = {}

    for area, accept_zone in accept_zone_by_area.items():

        burden_euro_class = EclIndex(
            f"burden per euro level per area {area}",
            graph.piecewise(
                (
                    0 * ticket_prices,
                    ticket_prices < accept_zone,
                ),
                (
                    1 + 0 * ticket_prices,
                    ticket_prices > 2*accept_zone,
                ),
                (
                    ticket_prices / accept_zone - 1,
                    True
                ),
            )
        )

        burden = Index(
            f"burden per area {area}",
            (burden_euro_class * euro_class_split).sum(axis=ECL_AXIS)
        )

        if exempt_share_by_area is not None:
            exempt = exempt_share_by_area.get(area, 0.0)
            burden = burden * graph.constant(1.0 - exempt)

        result[area] = Index(f"I_ticket_effect[{area}]", burden)

    return result
