
import numpy as np
import osmnx as ox
import datetime
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import json

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import TMP_PATH, logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("hex_centroids_enhanced", PathRegistry.HEX_POP_POI_STAT, "table"),
]


"""
Population distribution over H3 hexagons
Method: Dasymetric Mapping with building footprints from OSM

Logic:
  For each statistical area:
    1. Download residential building footprints from OSM
    2. Calculate built-up area per hexagon
    3. Distribute population proportionally to residential surface area
       (instead of geographic area as in uniform interpolation)

This is much more realistic because population is assigned
only where residential buildings actually exist.


"""

# TODO Performing using population data from bologna municipality. They have a dataset with 
# the exact number of resident for each civic number (building) in the city. This would be 
# the best possible dasymetric mapping, as we would have a direct population count per building. 
# We would just need to geocode the civic numbers to get their location and then assign them to hexagons.

def getResidentialBuilding(gdf_buildings, boundary, residential_tags):
    """
    Downloads building footprints from OSM within the given boundary polygon
    and filters them to retain only residential buildings.

    If fewer than 10% of buildings are tagged as residential, all buildings
    are used as a proxy for residential areas.

    Returns a GeoDataFrame in EPSG:32632 with a 'res_area' column (m²),
    or None if the OSM download fails.
    """
    gdf_res = None

    logging.debug(f"   Buildings downloaded: {len(gdf_buildings):,}")
    if "building" in gdf_buildings.columns:
        is_res = gdf_buildings["building"].isin(residential_tags["building"])
        gdf_res = gdf_buildings[is_res].copy()
    else:
        gdf_res = gdf_buildings.copy()

    # If too few residential buildings, use all as proxy
    res_frac = len(gdf_res) / max(len(gdf_buildings), 1)
    if res_frac < 0.10:
        logging.debug(f"   Only {res_frac:.1%} tagged as residential → using all buildings")
        gdf_res = gdf_buildings.copy()
    else:
        logging.debug(f"   Residential buildings: {len(gdf_res):,} ({res_frac:.1%})")

    gdf_res = gdf_res.to_crs(CRS_PROJECTED)[["geometry"]].copy()
    gdf_res["res_area"] = gdf_res.geometry.area
    logging.debug("   Dasymetric mapping active")

    return gdf_res


def populationMapping(hex_ids, gdf_areas_m, gdf_hex_m, gdf_res_m):
    """
    Distributes population from statistical areas onto H3 hexagons.

    For each statistical area with non-zero population:
      - Identifies intersecting hexagons.
      - If residential building data is available (dasymetric method),
        population is weighted by the residential surface area within
        each hexagon's intersection with the area.
      - If no buildings are found within an area, falls back to uniform
        distribution weighted by intersection area.

    Returns:
      pop_hex   (pd.Series): population assigned to each hexagon.
      n_dasy    (int): number of areas processed via dasymetric mapping.
      n_uniform (int): number of areas processed via uniform fallback.
    """
    pop_hex   = pd.Series(0.0, index=hex_ids)
    n_dasy    = 0
    n_uniform = 0

    for _, area_row in gdf_areas_m.iterrows():
        area_pop  = area_row["residenti"]
        if area_pop == 0:
            continue

        area_geom = area_row["geometry"]

        # Hexagons intersecting this statistical area
        hex_int = gdf_hex_m[gdf_hex_m.intersects(area_geom)].copy()
        if len(hex_int) == 0:
            continue

        assigned = False

        # ── Dasymetric mapping (if building data is available) ──────────
        if gdf_res_m is not None:
            buildings_in_area = gdf_res_m[gdf_res_m.intersects(area_geom)]

            if len(buildings_in_area) > 0:
                buildings_clipped = gpd.clip(buildings_in_area, area_geom)

                res_surf = {}
                for _, hrow in hex_int.iterrows():
                    # Compute residential surface within each hexagon
                    bld_in_hex = buildings_clipped[
                        buildings_clipped.intersects(hrow.geometry)
                    ]
                    if len(bld_in_hex) == 0:
                        res_surf[hrow["hex_id"]] = 0.0
                    else:
                        clipped = gpd.clip(bld_in_hex, hrow.geometry)
                        res_surf[hrow["hex_id"]] = clipped.geometry.area.sum()

                total_surf = sum(res_surf.values())
                if total_surf > 0:
                    # Distribute population proportionally to residential area
                    for hid, surf in res_surf.items():
                        pop_hex[hid] += area_pop * surf / total_surf
                    n_dasy += 1
                    assigned = True

        # ── Uniform fallback ────────────────────────────────────────────
        if not assigned:
            # Distribute population proportionally to intersection area
            hex_int["int_area"] = hex_int.geometry.apply(
                lambda g: g.intersection(area_geom).area
            )
            total_int = hex_int["int_area"].sum()
            if total_int > 0:
                for _, hrow in hex_int.iterrows():
                    pop_hex[hrow["hex_id"]] += area_pop * hrow["int_area"] / total_int
            n_uniform += 1

    return pop_hex, n_dasy, n_uniform


def main(
    ctx: dict
):
    """
    Main pipeline:
      1. Loads statistical area boundaries and 2022 population data.
      2. Loads H3 hexagon grid and centroids.
      3. Downloads residential building footprints from OSM (dasymetric method).
      4. Distributes population across hexagons.
      5. Saves the updated centroids CSV with a 'popolazione' column.
    """
    
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info(f"Reading relevant input data and preprocessing data")
        # Load population data and aggregate by statistical area for year 2022
        pop_df = pd.read_parquet(PathRegistry.FRAGILITIES_RESIDENT_POPULATION)
        pop_area = (
            pop_df[pop_df["anno"] == datetime.date(ctx["year"], 1, 1)]
            .groupby("codice_area_statistica")["residenti"]
            .sum()
            .reset_index()
        )
        pop_area["codice_area_statistica"] = pop_area["codice_area_statistica"].astype(int)

        # Merge population into area geometries
        # Utilizzo il dataset delle aree statistiche contenuta su fragilities_data_path
        gdf_areas = gpd.read_parquet(PathRegistry.FRAGILITIES_STATISTICAL_AREAS)
        # Drop centroid column and set geometry to 'geo_shape'
        gdf_areas = (gdf_areas.drop(columns=['geo_point_2d'])
                               .rename(columns={'geo_shape': 'geometry'})
                               .set_geometry('geometry')
                    )
        
        # Original dataset
        gdf_areas["codice_area_statistica"] = gdf_areas["codice_area_statistica"].astype(int)
        gdf_areas = gdf_areas.merge(pop_area, on="codice_area_statistica", how="left")
        gdf_areas["residenti"] = gdf_areas["residenti"].fillna(0)

        if gdf_areas.crs.to_epsg() != 4326:
            gdf_areas = gdf_areas.to_crs(CRS_LATLONG)

        # Load hexagon centroids and grid
        hex_poi_stats_gdf = gpd.read_parquet(PathRegistry.HEX_POI_STAT)
        centroids = hex_poi_stats_gdf[["hex_id", "centroid_lat", "centroid_lon", "poi_count"]].copy()
        hex_ids   = centroids["hex_id"].tolist()
        N         = len(hex_ids)

        gdf_hex = gpd.read_parquet(PathRegistry.HEX_POI_STAT).drop(columns=["poi_count"], errors="ignore").to_crs(CRS_LATLONG)

        # Compute the union of all statistical areas to use as OSM query boundary
        boundary = unary_union(gdf_areas.geometry)

        logging.info(f"Processing residential building footprints from OSM for dasymetric mapping")
        gdf_buildings = pd.read_parquet(PathRegistry.POI_BUILDINGS)
        geometry = gpd.GeoSeries.from_wkt(gdf_buildings['geometry'])
        gdf_buildings = gpd.GeoDataFrame(gdf_buildings, geometry=geometry, crs=CRS_LATLONG)
        gdf_res = getResidentialBuilding(gdf_buildings, boundary, ctx['osm_residential_tags'])

        method = "dasymetric" if gdf_res is not None else "uniform"
        logging.info(f"Distributing population ({method})")

        # Reproject to metric CRS for area calculations
        gdf_hex_m   = gdf_hex.to_crs(CRS_PROJECTED)
        gdf_areas_m = gdf_areas.to_crs(CRS_PROJECTED)

        pop_hex, n_dasy, n_uniform = populationMapping(hex_ids, gdf_areas_m, gdf_hex_m, gdf_res)

        #diagnostic(pop_area, pop_hex, n_dasy, n_uniform, N)

        # Save results back to the centroids CSV
        logging.info(f"Saving population assigned to hexagons")
        centroids["popolazione"] = pop_hex.reindex(hex_ids).fillna(0).values
        centroids.to_parquet(PathRegistry.HEX_POP_POI_STAT)
 


def diagnostic(pop_area, pop_hex, n_dasy, n_uniform, N):
    """
    Prints a diagnostic summary comparing the original population totals
    against the population distributed across hexagons, and writes the
    results to a JSON file.

    Warns if the discrepancy exceeds 1%, which may indicate misalignment
    between statistical area boundaries and the hexagon grid.

    Args:
        pop_area    (pd.DataFrame): original population by statistical area.
        pop_hex     (pd.Series):    population distributed across hexagons.
        n_dasy      (int):          number of areas processed via dasymetric mapping.
        n_uniform   (int):          number of areas processed via uniform fallback.
        N           (int):          total number of hexagons.
        output_path (Path):         directory where the diagnostic file will be saved.
    """


    pop_totale_orig = pop_area["residenti"].sum()
    pop_totale_hex  = pop_hex.sum()
    diff_pct        = abs(pop_totale_hex - pop_totale_orig) / pop_totale_orig

    report = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method_breakdown": {
            "dasymetric_areas": n_dasy,
            "uniform_areas":    n_uniform,
        },
        "population": {
            "original":            round(pop_totale_orig, 2),
            "on_hexagons":         round(pop_totale_hex, 2),
            "discrepancy_pct":     round(diff_pct * 100, 4),
        },
        "hexagons": {
            "total":               N,
            "with_population":     int((pop_hex > 0).sum()),
        },
        "warnings": (
            ["Discrepancy > 1% — check area/grid alignment"] if diff_pct > 0.01 else []
        ),
    }

    # Write to JSON file
    report_dir = TMP_PATH / "diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "diagnostic_population_mapping.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
 
  
if __name__ == "__main__":
    
    main(ctx=context)