
import numpy as np

import pandas as pd
import geopandas as gpd

from bdt_mobility.utils import data_reader
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("hex_centroids_enhanced", PathRegistry.HEX_POP_POI_STAT, "table"),
]


"""
Hexagon-to-Zone Aggregation with Heterogeneity Metrics
=======================================================
Aggregates all metrics from H3 hexagons (resolution 9) to larger municipal zones,
accounting for spatial heterogeneity within each zone.
 
For additive metrics (population, POI counts): direct sum
For derived metrics (CDI, travel times): population-weighted mean + heterogeneity measure
 
Heterogeneity metrics
---------------------
For each derived metric M within zone z:
  - Mean: Σ_h p(h) · M(h) / Σ_h p(h)  [population-weighted]
  - Std dev: sqrt(Σ_h p(h) · (M(h) - mean)²) / sqrt(Σ_h p(h))  [weighted]
  - Gini coefficient: measures inequality in metric distribution across hexagons
  
High heterogeneity → some areas in the zone are much better/worse served than others
Low heterogeneity → uniform service quality across the zone
 
Inputs
------
  hex_centroids.csv
      Hexagon IDs, locations, population
  
  opportunity_scores_by_category.geojson (or baseline opportunity_scores.geojson)
      Hexagon-level CDI, opportunity scores, geometry
  
  tt_car_matrix.npy, tt_pt_matrix.npy
      N×N travel time matrices
  
  poi_classified.geojson
      POI assignments to hexagons with categories
  
  gdf_zones from data_reader.AOI_shapes(...)
      Target municipal zones with geometry
 
Outputs
-------
  zones_aggregated_complete.geojson
      Zone-level geometry + all aggregated metrics
  
  zones_metrics_summary.csv
      Tabular summary: mean, std, Gini for each metric per zone
  
  zones_heterogeneity_map.png
      Visual: CDI heterogeneity (std dev) by zone
"""

# ── Heterogeneity metrics ─────────────────────────────────────
def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean: Σ(v·w) / Σw"""
    valid = ~np.isnan(values) & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return (values[valid] * weights[valid]).sum() / weights[valid].sum()
 
 
def weighted_std(values: np.ndarray, weights: np.ndarray, mean: float = None) -> float:
    """Weighted standard deviation"""
    valid = ~np.isnan(values) & (weights > 0)
    if valid.sum() < 2:
        return np.nan
    if mean is None:
        mean = weighted_mean(values, weights)
    var = ((values[valid] - mean) ** 2 * weights[valid]).sum() / weights[valid].sum()
    return np.sqrt(var)
 
 
def gini_coefficient(values: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Gini coefficient: measures inequality in distribution.
    0 = perfect equality, 1 = perfect inequality
    
    Formula: G = (1 / (2·n·mean)) · Σ_i Σ_j |x_i - x_j|
    
    For weighted version, treats each weight as multiple instances.
    """
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.nan
 
    vals = values[valid]
    
    if weights is None:
        weights = np.ones(len(vals))
    else:
        weights = weights[valid]
 
    # Expand values by weights (approximate for continuous weights)
    # For efficiency, use weighted formula directly
    mean_val = weighted_mean(vals, weights)
    if mean_val == 0:
        return np.nan
 
    # Pairwise differences (weighted)
    n_eff = weights.sum()
    gini = 0.0
    for i in range(len(vals)):
        for j in range(len(vals)):
            gini += weights[i] * weights[j] * np.abs(vals[i] - vals[j])
    
    gini = gini / (2 * n_eff ** 2 * mean_val)
    return min(gini, 1.0)  # cap at 1
  
def coefficient_of_variation(values: np.ndarray, weights: np.ndarray) -> float:
    """CV = std / mean (if mean > 0)"""
    mean = weighted_mean(values, weights)
    if np.isnan(mean) or mean == 0:
        return np.nan
    std = weighted_std(values, weights, mean)
    return std / np.abs(mean)

# ── Spatial join hex to zones ─────────────────────────────────
def spatial_join_hex_to_zones(gdf_hex: gpd.GeoDataFrame,
                              gdf_zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Assigns each hexagon to its containing zone.
    
    Returns DataFrame with: hex_id, zone_id, (hex metrics)
    """
    logging.info("Performing spatial join (hexagon → zone)...")
    
    joined = gpd.sjoin(
            gdf_hex[["hex_id", "geometry"]],
            gdf_zones[["geometry", "id_zone"]].reset_index(drop=True),
            how='left',
            predicate='within'
          )
    
    
    # Merge back metrics from gdf_hex
    metric_cols = [c for c in gdf_hex.columns if c not in ("hex_id", "geometry")]
    joined = joined.merge(
        gdf_hex[["hex_id"] + metric_cols].drop("geometry", axis=1, errors="ignore"),
        on="hex_id", how="left"
    )
    
    unassigned = joined["id_zone"].isna().sum()
    if unassigned > 0:
        logging.warning(f"{unassigned:,} hexagons not assigned to any zone")
    
    return joined

# ── Count POIs by zone and category ──────────────────────────
def count_pois_by_zone_and_category(gdf_poi: gpd.GeoDataFrame,
                                     joined_hex_zone: pd.DataFrame) -> pd.DataFrame:
    """
    Counts POIs per zone per macrocategory.
    """
    logging.info("Counting POIs by zone and macrocategory...")
    
    # Build hex_id → zone_id mapping
    hex_to_zone = dict(zip(joined_hex_zone["hex_id"], joined_hex_zone["zone_id"]))
    
    # Assign POIs to zones via their hexagon assignments
    gdf_poi_with_zone = gdf_poi.copy()
    gdf_poi_with_zone["id_zone"] = gdf_poi_with_zone["hex_id"].map(hex_to_zone)
    
    # Count by zone and macrocategory
    poi_counts = (
        gdf_poi_with_zone
        .groupby(["id_zone", "macro_categoria"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Add total POI count
    poi_counts["poi_total"] = poi_counts.sum(axis=1)
    
    return poi_counts

# ── Aggregate metrics to zones ────────────────────────────────
def aggregate_to_zones(joined_hex_zone: pd.DataFrame,
                       gdf_zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aggregates all metrics from hexagons to zones with heterogeneity measures.
    """
    logging.info("Aggregating metrics to zones...")
        
    results = []
    
    for zone_id, group in joined_hex_zone.groupby("id_zone", dropna=True):
        zone_rec = {"id_zone": zone_id}
        
        # ── Additive metrics ──────────────────────────────────
        pop_total = group["popolazione"].sum()
        zone_rec["popolazione_totale"] = pop_total
        
        # ── CDI aggregation ───────────────────────────────────
        if "cdi" in group.columns:
            cdi_vals = group["cdi"].values
            pop_weights = group["popolazione"].values
            
            # Mean (population-weighted)
            zone_rec["cdi_mean"] = weighted_mean(cdi_vals, pop_weights)
            
            # Standard deviation (population-weighted)
            zone_rec["cdi_std"] = weighted_std(cdi_vals, pop_weights, zone_rec["cdi_mean"])
            
            # Gini coefficient (inequality)
            zone_rec["cdi_gini"] = gini_coefficient(cdi_vals, pop_weights)
            
            # Coefficient of variation
            zone_rec["cdi_cv"] = coefficient_of_variation(cdi_vals, pop_weights)
            
            # Percentile breakdown
            valid_cdi = cdi_vals[~np.isnan(cdi_vals)]
            if len(valid_cdi) > 0:
                zone_rec["cdi_p25"] = np.nanpercentile(valid_cdi, 25)
                zone_rec["cdi_p50"] = np.nanpercentile(valid_cdi, 50)
                zone_rec["cdi_p75"] = np.nanpercentile(valid_cdi, 75)
                zone_rec["pct_car_dependent"] = (valid_cdi > 0).mean()
        
        # ── Opportunity scores by category ────────────────────
        for cat_col in group.columns:
            if cat_col.startswith("score_") and "cdi" not in cat_col:
                score_vals = group[cat_col].values
                
                zone_rec[f"{cat_col}_mean"] = weighted_mean(score_vals, pop_weights)
                zone_rec[f"{cat_col}_std"] = weighted_std(score_vals, pop_weights, 
                                                           zone_rec.get(f"{cat_col}_mean", np.nan))
                zone_rec[f"{cat_col}_gini"] = gini_coefficient(score_vals, pop_weights)
        
        results.append(zone_rec)
    
    agg_df = pd.DataFrame(results)
    
    # Merge with zone geometries
    gdf_result = gdf_zones.reset_index(drop=True).copy()
    gdf_result = gdf_result.merge(agg_df, on="id_zone", how="left")
    
    logging.info(f"Aggregated to {len(gdf_result):,} zones")
    return gdf_result

def main(
    ctx: dict
):
        
    logging.basicConfig(level=logging_level)
    with timed():
        
        logging.info("Loading data...")
    
        # Load hexagon geometry and metrics
        opportunity_score_df = pd.read_parquet(PathRegistry.OPPORTUNITY_SCORES)
        gdf_hex = gpd.GeoDataFrame(
                        opportunity_score_df,
                        geometry=gpd.points_from_xy(opportunity_score_df["centroid_lon"], opportunity_score_df["centroid_lat"]),
                        crs=CRS_LATLONG
        ).to_crs(CRS_PROJECTED)
        
        # Load hexagon centroids (for population and IDs)
        hex_pop_poi_df = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
        gdf_hex = gdf_hex.merge(hex_pop_poi_df[["hex_id", "popolazione"]], on="hex_id", how="left")
        
        logging.info(f"Loading PUMS zones for aggregation...")
        zoi = data_reader.ZOI_shape(path=PathRegistry.ZOI_BORDER)

        pums_zones_gdf = data_reader.AOI_shapes(
            path=PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od",
            df_around=zoi,
            return_type="inside",
            ).to_crs(CRS_PROJECTED)
        
        
        logging.info("Performing spatial join of hexagons to zones...")
        # Spatial join
        joined = spatial_join_hex_to_zones(gdf_hex, pums_zones_gdf)

        # Count POIs by zone and category
        gdf_zones_result = aggregate_to_zones(joined, pums_zones_gdf)
        
        # Save results
        gdf_zones_result.to_parquet(PathRegistry.MAP_TO_PUMS_ZONES, engine='pyarrow', index=False)
        logging.info(f"Saved: {PathRegistry.MAP_TO_PUMS_ZONES}")
    
  
if __name__ == "__main__":
    
    main(ctx=context)