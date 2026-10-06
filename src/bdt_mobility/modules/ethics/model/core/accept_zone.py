def compute_accept_zone(
    income_by_area: dict[int, float],
    income_ref: float,
    global_min: float = 4.0,
    global_max: float = 7.0,
) -> dict[int, float]:
    midpoint = (global_min + global_max) / 2.0
    result: dict[int, float] = {}
    for area, income in income_by_area.items():
        raw = midpoint * (income / income_ref) if income_ref != 0.0 else midpoint
        result[area] = max(global_min, min(global_max, raw))
    return result
