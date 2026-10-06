# NOTE:
# this module likely will be ported to a class once projection ownership,
# metadata, and integration points across EthicalModel and MobilityModel are fully defined

from __future__ import annotations

from typing import TypeAlias

import pandas as pd

from civic_digital_twins.dt_model import Index
from civic_digital_twins.dt_model.engine.frontend import graph

# this is a area-weight couple
ProjectionEntry: TypeAlias = tuple[int, float]
# this is a dict of many area-weigth couples -> represent the mapping between a new representation and all the portions form the old representation
ProjectionMap: TypeAlias = dict[int, tuple[ProjectionEntry, ...]]


def build_projection_map_from_zone_ids(
    crosswalk: pd.DataFrame,
    od_zones: pd.DataFrame,
    *,
    area_col: str = "codice_area_statistica",
    zone_id_col: str = "id_zone",
    od_zone_id_col: str = "id_zone",
    od_zone_index_col: str = "index",
    weight_col: str = "perc_area_intersect",
) -> ProjectionMap:

    # crosswalk.id_zone  ->  aree_gdf_inside.id_zone  ->  aree_gdf_inside.index

    # crosswalk validity check
    required_crosswalk_cols = [area_col, zone_id_col, weight_col]
    missing_crosswalk_cols = [col for col in required_crosswalk_cols if col not in crosswalk.columns]
    if missing_crosswalk_cols:
        raise KeyError(f"Missing crosswalk columns: {missing_crosswalk_cols}")

    # simulation geography validity check
    required_od_zone_cols = [od_zone_id_col, od_zone_index_col]
    missing_od_zone_cols = [col for col in required_od_zone_cols if col not in od_zones.columns]
    if missing_od_zone_cols:
        raise KeyError(f"Missing OD zone columns: {missing_od_zone_cols}")

    # keep last occurrence for duplicate zone ids
    zone_lookup = (
        od_zones[[od_zone_id_col, od_zone_index_col]]
        .drop_duplicates()
        .drop_duplicates(subset=[od_zone_id_col], keep="last")
        .copy()
    )
    # generate a dict containing the mapping
    id_to_index = dict(zip(zone_lookup[od_zone_id_col], zone_lookup[od_zone_index_col]))

    translated = crosswalk[required_crosswalk_cols].copy()
    # ACTUAL TRANSLATION -> available columns: area_statistica, pums, weight, sim-index
    translated["_zone_index"] = translated[zone_id_col].map(id_to_index)

    # post translation sanity check
    unmapped_zone_ids = (
        translated.loc[translated["_zone_index"].isna(), zone_id_col]
        .drop_duplicates()
        .tolist()
    )
    if unmapped_zone_ids:
        raise ValueError(
            "Crosswalk contains raw zone ids that are not present in the simulation zones: "
            f"{unmapped_zone_ids[:10]}"
        )

    return _build_projection_map(
        translated,
        area_col=area_col,
        zone_col="_zone_index",
        weight_col=weight_col,
    )


def _build_projection_map(
    df: pd.DataFrame,
    *,
    area_col: str,
    zone_col: str,
    weight_col: str,
) -> ProjectionMap:

    # init check
    required_cols = [area_col, zone_col, weight_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing projection columns: {missing_cols}")

    # filter
    cleaned = df.loc[:, required_cols].dropna().copy()
    if cleaned.empty:
        raise ValueError("Projection dataframe is empty after dropping null rows.")
    cleaned[area_col] = cleaned[area_col].astype(int)
    cleaned[zone_col] = cleaned[zone_col].astype(int)
    cleaned[weight_col] = cleaned[weight_col].astype(float)

    # values check
    negative = cleaned[cleaned[weight_col] < 0.0]
    if not negative.empty:
        sample = negative.head().to_dict(orient="records")
        raise ValueError(f"Projection weights must be non-negative. Sample bad rows: {sample}")

    # validity check
    grouped = (
        cleaned.groupby([zone_col, area_col], as_index=False, sort=True)[weight_col]
        .sum()
        .sort_values([zone_col, area_col])
    )
    if grouped.empty:
        raise ValueError("Projection dataframe contains no valid area/zone mappings.")

    # build the map
    projection: ProjectionMap = {}
    for zone, zone_group in grouped.groupby(zone_col, sort=True):
        entries = tuple(
            (int(area), float(weight))
            for area, weight in zone_group[[area_col, weight_col]].itertuples(index=False, name=None)
        )
        if not entries:
            raise ValueError(f"Zone {zone} has no projection entries after normalization.")
        projection[int(zone)] = entries

    return projection


def project_float_map(
    values_by_area: dict[int, float],
    projection: ProjectionMap,
) -> dict[int, float]:

    # project to float
    projected: dict[int, float] = {}
    for zone, entries in projection.items(): # exploit the map
        total = 0.0
        matched = False
        for area, weight in entries:
            value = values_by_area.get(area)
            if value is None:
                continue
            total += weight * value
            matched = True
        if matched:
            projected[zone] = total
    return projected


def project_index_map(
    values_by_area: dict[int, Index],
    projection: ProjectionMap,
    *,
    index_name: str,
) -> dict[int, Index]:

    # project to Index objects
    projected: dict[int, Index] = {}
    for zone, entries in projection.items(): # exploit the map
        node = None
        for area, weight in entries:
            idx = values_by_area.get(area)
            if idx is None:
                continue
            term = graph.constant(weight) * graph.ensure_node(idx)
            node = term if node is None else node + term
        if node is None:
            continue
        projected[zone] = Index(f"{index_name}[{zone}]", node)
    return projected
