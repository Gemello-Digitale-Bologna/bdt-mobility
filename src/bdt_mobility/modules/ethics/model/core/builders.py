from civic_digital_twins.dt_model import Index

from .mpi import compute_mpi_index_by_area


def compute_mpi_by_area(
    data: dict,
    mpi_weights: dict,
    *,
    mpi_name: str,
    prefix: str = "I_modified_",
) -> dict:
    result = {}
    for area, year_map in data.items():
        for year in year_map:
            mpi_idx = compute_mpi_index_by_area(
                data, area, year, mpi_weights, prefix=prefix, mpi_name=mpi_name,
            )
            if mpi_idx is None:
                continue
            result.setdefault(area, {})
            result[area][year] = mpi_idx
    return result


def compute_indicator_deltas(baseline_by_area: dict, modified_by_area: dict) -> dict:
    deltas = {}
    for area, year_map in modified_by_area.items():
        for year, indicators in year_map.items():
            if area not in baseline_by_area or year not in baseline_by_area[area]:
                continue
            deltas.setdefault(area, {})
            deltas[area].setdefault(year, {})
            for modified_name, modified_idx in indicators.items():
                indicator_name = modified_name.removeprefix("I_modified_")
                baseline_idx = baseline_by_area[area][year].get(indicator_name)
                if baseline_idx is None:
                    continue
                delta_name = f"I_delta_{indicator_name}"
                deltas[area][year][delta_name] = Index(
                    f"{delta_name}[{area},{year}]",
                    modified_idx - baseline_idx,
                )
    return deltas


def compute_mpi_deltas(
    baseline_mpi_data: dict,
    modified_mpi_data: dict,
    *,
    baseline_mpi_name: str,
    modified_mpi_name: str,
    delta_mpi_name: str,
) -> dict:
    deltas = {}
    for area, year_map in modified_mpi_data.items():
        for year, mpi_map in year_map.items():
            if area not in baseline_mpi_data or year not in baseline_mpi_data[area]:
                continue
            baseline_idx = baseline_mpi_data[area][year].get(baseline_mpi_name)
            modified_idx = mpi_map.get(modified_mpi_name)
            if baseline_idx is None or modified_idx is None:
                continue
            deltas.setdefault(area, {})
            deltas[area][year] = {
                delta_mpi_name: Index(
                    f"{delta_mpi_name}[{area},{year}]",
                    modified_idx - baseline_idx,
                )
            }
    return deltas
