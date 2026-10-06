"""
Scenario Analysis: Public Transport Enhancement & CDI
====================================================
Scenario pipeline that diagnoses public-transport coverage gaps, adds
synthetic stops/routes/trips to the GTFS directory, recomputes the public
transport travel-time matrix with CSA, and aggregates the resulting CDI by
PUMS zone.

Workflow
--------
1. Diagnose current PT coverage gaps by zone
2. Select zones below the configured coverage target
3. Generate synthetic PT improvements (new stops, routes, trips)
4. Merge the synthetic records into the GTFS scenario directory
5. Recompute the PT travel-time matrix using CSA
6. Calculate the scenario CDI at hexagon level
7. Aggregate scenario scores and CDI statistics by PUMS zone

The scenario is speculative (no cost considerations) and designed to answer:
"If we invested in PT in the worst-served zones, how would CDI change?"

Inputs
------
  PathRegistry.TRAVEL_TIMES_PT
      Baseline PT travel-time matrix as a parquet DataFrame. Its index and
      columns are expected to contain the hexagon IDs.

  PathRegistry.HEX_POP_POI_STAT
      Hexagon centroids, IDs, population, and POI counts.

  PathRegistry.OPPORTUNITY_SCORES
      Baseline car opportunity scores.

  PathRegistry.GTFS_DATA
      Original GTFS directory containing stops.txt, routes.txt, trips.txt,
      stop_times.txt, and calendar.txt.

  PathRegistry.OD_SHAPE_ZONE
      PUMS zone geometries used to assign hexagons to zones.

Outputs
-------
  PathRegistry.GTFS_PT_SCENARIO
      Directory containing the original GTFS records plus synthetic records.

  PathRegistry.TRAVEL_TIMES_PT_SCENARIO
      Scenario PT travel-time matrix as a parquet DataFrame.

  PathRegistry.PT_SCENARIO_CDI_IMPACT
      Zone-level scenario metrics as a parquet GeoDataFrame, including
      population-weighted CDI and opportunity-score statistics.

Notes
-----
The current implementation stores scenario results only. It does not yet
calculate or save a baseline-to-scenario CDI delta table or comparison map.
"""
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.utils.timing import timed


from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context
from bdt_mobility.modules.transport.extraction.transport_efficiency.step_03_travel_times_public_transport import compute_travel_times
from bdt_mobility.modules.transport.extraction.transport_efficiency.step_05_car_dependency_estimation import getOpportunityScores
from bdt_mobility.modules.transport.extraction.transport_efficiency.step_07_from_hex_to_zoning import aggregate_to_zones, spatial_join_hex_to_zones

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
import zipfile
import tempfile

# ── PT Coverage Analysis ──────────────────────────────────────
def analyze_pt_coverage(tt_pt: pd.DataFrame,
                        gdf_hex: gpd.GeoDataFrame,
                        gdf_zones: gpd.GeoDataFrame,
                        threshold_min: int = 45) -> pd.DataFrame:
    """
    Analyzes PT accessibility per zone.
    
    Parameters
    ----------
    tt_pt : pd.DataFrame
        PT travel time matrix (seconds). Index and columns are hex_ids.
    gdf_hex : gpd.GeoDataFrame
        GeoDataFrame containing hexagon geometries and an 'hex_id' column.
    gdf_zones : gpd.GeoDataFrame
        Municipal zones.
    threshold_min : int
        Coverage threshold in minutes.
    
    Returns
    -------
    pd.DataFrame with n_hexagons, coverage_pct, avg_tt_pt_min per zone.
    """
    logging.info(f"Analyzing PT coverage (threshold: {threshold_min} min)...")
    
    # 1. Estraggo la lista ordinata degli hex_id dall'indice della matrice tt_pt
    tt_hex_ids = list(tt_pt.index)
    
    # Mappatura per lookup O(1) dell'indice di riga nella matrice numpy
    hex_to_idx = {hex_id: idx for idx, hex_id in enumerate(tt_hex_ids)}
    
    # 2. Convert DataFrame to numpy for faster processing
    tt_pt_array = tt_pt.values
    
    # 3. Spatial Join direttamente sul gdf_hex fornito in input
    joined = gpd.sjoin(
        gdf_hex[["hex_id", "geometry"]],
        gdf_zones[["geometry", "id_zone"]].reset_index(drop=True),
        how="left",
        predicate="within"
    ).astype({"id_zone": int})
    
    results = []
    for zone_id, group in joined.groupby("id_zone", dropna=True):
        
        # Mappiamo gli hex_id presenti nella zona con i loro indici nella matrice numpy
        hex_indices = [hex_to_idx[h] for h in group["hex_id"] if h in hex_to_idx]
        
        if not hex_indices:
            continue
        
        # Coverage: % of destinations reachable within threshold
        tt_zone = tt_pt_array[hex_indices, :].astype(float)  # shape (len(hex_indices), N)
        threshold_sec = float(threshold_min) * 60
        
        reachable = ((~np.isnan(tt_zone)) & (tt_zone > 0) & 
                     (tt_zone <= threshold_sec)).sum(axis=0)
        reachable_pct = (reachable > 0).mean()
        
        # Average travel time to reachable destinations
        valid_tt = tt_zone[~np.isnan(tt_zone) & (tt_zone > 0)]
        avg_tt = valid_tt.mean() / 60 if len(valid_tt) > 0 else np.nan
        
        results.append({
            "zone_id": zone_id,
            "n_hexagons": len(hex_indices),
            "coverage_pct": reachable_pct,
            "avg_tt_pt_min": avg_tt,
        })
    
    return pd.DataFrame(results)

# ── Generate synthetic GTFS additions ──────────────────────────
def generate_synthetic_gtfs_additions(coverage_df: pd.DataFrame,
                                      hex_centroids: pd.DataFrame,
                                      service_id: str,
                                      hex_zone_mapping: dict,
                                      target_coverage: float = 0.80) -> dict:
    """
    Generates synthetic PT additions for zones below target coverage.
        
    Returns dict with synthetic GTFS components (stops, routes, trips, stop_times)
    """
    logging.info(f"Generating synthetic GTFS additions (target coverage: {target_coverage:.0%})...")
    
    # Identify underserved zones
    underserved = coverage_df[coverage_df["coverage_pct"] < target_coverage].sort_values(
        "coverage_pct"
    )
    
    logging.info(f"Found {len(underserved)} zones below target coverage")
    
    new_stops = []
    new_routes = []
    new_trips = []
    new_stop_times = []
    
    stop_counter = 10000
    route_counter = 100
    trip_counter = 10000
    
    for idx, zone_row in underserved.iterrows():
        zone_id = zone_row["zone_id"]
        logging.info(f"  Zone {zone_id}: coverage {zone_row['coverage_pct']:.1%}")
        
        # Number of new stops proportional to coverage gap
        n_new_stops = max(2, int(3 * (1 - zone_row["coverage_pct"])))
        
        # Use zone centroid (sample hex from zone)
        zone_hex_ids = hex_zone_mapping[zone_id]
        zone_hex_sample = hex_centroids[hex_centroids["hex_id"].isin(zone_hex_ids)].sample(1)
        
        base_lat = zone_hex_sample["centroid_lat"].iloc[0]
        base_lon = zone_hex_sample["centroid_lon"].iloc[0]

        # Create synthetic stops
        zone_stop_ids = []
        for i in range(n_new_stops):
            stop_id = str(stop_counter)
            new_stops.append({
                "stop_id": stop_id,
                "stop_name": f"Synth_Zone{zone_id}_Stop{i}",
                "stop_lat": base_lat + i * 0.002,
                "stop_lon": base_lon + i * 0.002,
                "stop_desc": f"Synthetic stop for scenario (Zone {zone_id})",
            })
            zone_stop_ids.append(stop_id)
            stop_counter += 1
        
        # Create synthetic route
        route_id = str(route_counter)
        new_routes.append({
            "route_id": route_id,
            "agency_id": "TPER",
            "route_short_name": f"S{zone_id}",
            "route_long_name": f"Scenario Zone {zone_id} Service",
            "route_type": "3",  # bus
        })
        route_counter += 1
        
        # Create trips (4 per day)
        for trip_num in range(4):
            trip_id = str(trip_counter)
            new_trips.append({
                "trip_id": trip_id,
                "route_id": route_id,
                "service_id": service_id,
                "trip_headsign": f"Zone {zone_id}",
            })
            trip_counter += 1
            
            # Add stop times — ONLY for stops we just created
            base_hour = [7, 12, 16, 20][trip_num]
            for j, stop_id in enumerate(zone_stop_ids):
                new_stop_times.append({
                    "trip_id": trip_id,
                    "stop_id": stop_id,  # Use actual stop_id, not counter offset
                    "arrival_time": f"{base_hour:02d}:{j*5:02d}:00",
                    "departure_time": f"{base_hour:02d}:{j*5:02d}:30",
                    "stop_sequence": j + 1,
                })
    
    return {
        "new_stops": pd.DataFrame(new_stops),
        "new_routes": pd.DataFrame(new_routes),
        "new_trips": pd.DataFrame(new_trips),
        "new_stop_times": pd.DataFrame(new_stop_times),
    }
 
 
def apply_gtfs_modifications(gtfs_original: Path,
                             synthetic_additions: dict,
                             output_path: Path,
                             validate: bool = True) -> None:
    """
    Merges synthetic additions into the original GTFS with validation.
    
    Parameters
    ----------
    gtfs_original_zip : Path
        Path to original GTFS zip file
    synthetic_additions : dict
        Dict with keys: new_stops, new_routes, new_trips, new_stop_times
    output_zip : Path
        Path to save modified GTFS
    validate : bool
        If True, validate that all references are consistent
    """
    logging.info(f"Applying GTFS modifications...")
    logging.info(f"  New stops: {len(synthetic_additions['new_stops'])}")
    logging.info(f"  New routes: {len(synthetic_additions['new_routes'])}")
    logging.info(f"  New trips: {len(synthetic_additions['new_trips'])}")
    logging.info(f"  New stop times: {len(synthetic_additions['new_stop_times'])}")
    
    
    # Load original GTFS files
    logging.info("Loading GTFS files...")
    stops_df = pd.read_csv(gtfs_original / "stops.txt", dtype={"stop_id": str})
    routes_df = pd.read_csv(gtfs_original / "routes.txt", dtype={"route_id": str})
    trips_df = pd.read_csv(gtfs_original / "trips.txt", dtype={"trip_id": str, "route_id": str})
    stop_times_df = pd.read_csv(gtfs_original / "stop_times.txt", dtype={"trip_id": str, "stop_id": str})
    calendar_df = pd.read_csv(gtfs_original / "calendar.txt")
    
    logging.info(f"Original GTFS: {len(stops_df)} stops, {len(routes_df)} routes, "
                f"{len(trips_df)} trips, {len(stop_times_df)} stop_times")
    
    # Ensure synthetic data has correct dtypes
    new_stops = synthetic_additions["new_stops"].copy()
    new_stops["stop_id"] = new_stops["stop_id"].astype(str)
    
    new_routes = synthetic_additions["new_routes"].copy()
    new_routes["route_id"] = new_routes["route_id"].astype(str)
    
    new_trips = synthetic_additions["new_trips"].copy()
    new_trips["trip_id"] = new_trips["trip_id"].astype(str)
    new_trips["route_id"] = new_trips["route_id"].astype(str)
    
    new_stop_times = synthetic_additions["new_stop_times"].copy()
    new_stop_times["trip_id"] = new_stop_times["trip_id"].astype(str)
    new_stop_times["stop_id"] = new_stop_times["stop_id"].astype(str)
    
    # Validation: Check referential integrity
    if validate:
        logging.info("Validating referential integrity...")
        
        # Check that all new stop_ids in stop_times exist in new_stops
        orphaned_stops = new_stop_times[~new_stop_times["stop_id"].isin(new_stops["stop_id"])]["stop_id"].unique()
        if len(orphaned_stops) > 0:
            logging.error(f"Orphaned stops in stop_times: {orphaned_stops}")
            raise ValueError("New stop_times reference non-existent stops")
        
        # Check that all new trip_ids in stop_times exist in new_trips
        orphaned_trips = new_stop_times[~new_stop_times["trip_id"].isin(new_trips["trip_id"])]["trip_id"].unique()
        if len(orphaned_trips) > 0:
            logging.error(f"Orphaned trips in stop_times: {orphaned_trips}")
            raise ValueError("New stop_times reference non-existent trips")
        
        # Check that all new route_ids in trips exist in new_routes
        orphaned_routes = new_trips[~new_trips["route_id"].isin(new_routes["route_id"])]["route_id"].unique()
        if len(orphaned_routes) > 0:
            logging.error(f"Orphaned routes in trips: {orphaned_routes}")
            raise ValueError("New trips reference non-existent routes")
        
        logging.info("Referential integrity check passed")
    
    # Append synthetic data
    logging.info("Merging synthetic data...")
    stops_df = pd.concat([stops_df, new_stops], ignore_index=True)
    routes_df = pd.concat([routes_df, new_routes], ignore_index=True)
    trips_df = pd.concat([trips_df, new_trips], ignore_index=True)
    stop_times_df = pd.concat([stop_times_df, new_stop_times], ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    stops_df = stops_df.drop_duplicates(subset=["stop_id"], keep="first")
    routes_df = routes_df.drop_duplicates(subset=["route_id"], keep="first")
    trips_df = trips_df.drop_duplicates(subset=["trip_id"], keep="first")
    stop_times_df = stop_times_df.drop_duplicates(subset=["trip_id", "stop_sequence"], keep="first")
    
    logging.info(f"Modified GTFS: {len(stops_df)} stops, {len(routes_df)} routes, "
                f"{len(trips_df)} trips, {len(stop_times_df)} stop_times")
    
    # Save modified files
    logging.info("Saving modified GTFS files...")
    stops_df.to_csv(output_path / "stops.txt", index=False)
    routes_df.to_csv(output_path / "routes.txt", index=False)
    trips_df.to_csv(output_path / "trips.txt", index=False)
    stop_times_df.to_csv(output_path / "stop_times.txt", index=False)
    calendar_df.to_csv(output_path / "calendar.txt", index=False)


# ── Test functions ────────────────────────────────────────────
def test_gtfs_modifications(gtfs_original: Path, gtfs_scenario: Path) -> None:
    """
    Quick validation that the modified GTFS is structurally sound.
    """
    logging.info("Testing modified GTFS integrity...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(gtfs_scenario, 'r') as z:
            z.extractall(tmpdir)
        
        stops_df = pd.read_csv(tmpdir / "stops.txt", dtype={"stop_id": str})
        routes_df = pd.read_csv(tmpdir / "routes.txt", dtype={"route_id": str})
        trips_df = pd.read_csv(tmpdir / "trips.txt", dtype={"trip_id": str, "route_id": str})
        stop_times_df = pd.read_csv(tmpdir / "stop_times.txt", dtype={"trip_id": str, "stop_id": str})
        
        # Validation checks
        checks = {
            "No duplicate stop_ids": len(stops_df) == len(stops_df["stop_id"].unique()),
            "No duplicate trip_ids": len(trips_df) == len(trips_df["trip_id"].unique()),
            "No duplicate route_ids": len(routes_df) == len(routes_df["route_id"].unique()),
            "All stop_times reference existing stops": 
                stop_times_df["stop_id"].isin(stops_df["stop_id"]).all(),
            "All trips reference existing routes": 
                trips_df["route_id"].isin(routes_df["route_id"]).all(),
            "All stop_times reference existing trips": 
                stop_times_df["trip_id"].isin(trips_df["trip_id"]).all(),
        }
        
        all_passed = True
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            logging.info(f"  {status} {check_name}")
            if not result:
                all_passed = False
        
        if all_passed:
            logging.info("All integrity checks passed!")
        else:
            logging.error("Some integrity checks failed!")
            raise ValueError("Modified GTFS has integrity issues")

def survival_function(t_sec: np.ndarray, tau: int):
    """f(t) = (1/τ) * exp(-t/τ)"""
    return (1 / tau) * np.exp(-t_sec / tau)

# ── Calculate CDI from travel times ──────────────────────────
def compute_opportunity_scores(tt_car_df: pd.DataFrame,
                               tt_pt: pd.DataFrame | np.ndarray,
                               poi_counts: pd.Series,
                               tau_sec: float,
                               dt: float) -> tuple:
    """
    Computes opportunity scores and CDI from travel time DataFrames.
    
    Parameters
    ----------
    tt_car_df : pd.DataFrame
        Car travel times (N×N, seconds), index and columns are hex_ids
    tt_pt_df : pd.DataFrame
        PT travel times (N×N, seconds), index and columns are hex_ids
    poi_counts : pd.Series
        POI count per hex (indexed by hex_id)
    tau_sec : float
        Decay parameter in seconds
    
    Returns
    -------
    (scores_car, scores_pt, cdi) as np.arrays
    """
    tt_car = tt_car_df.values
    if isinstance(tt_pt, pd.DataFrame):
      tt_pt = tt_pt.values
    else:
      tt_pt = tt_pt
    poi_counts_arr = poi_counts.values
    N = len(poi_counts)
    
    valid_car = (~np.isnan(tt_car)) & (tt_car > 0)
    valid_pt = (~np.isnan(tt_pt)) & (tt_pt > 0)
    
    scores_car = np.zeros(N)
    scores_pt = np.zeros(N)
    
    for i in range(N):
        if poi_counts_arr[i] > 0:
            weights_car = np.where(
                valid_car[i, :],
                poi_counts_arr[i] * survival_function(tt_car[i, :], tau_sec) * dt,
                0.0
            )
            weights_pt = np.where(
                valid_pt[i, :],
                poi_counts_arr[i] * survival_function(tt_pt[i, :], tau_sec) * dt,
                0.0
            )
            scores_car[i] = weights_car.sum()
            scores_pt[i] = weights_pt.sum()
    
    denom = scores_car + scores_pt
    cdi = np.where(denom > 0, (scores_car - scores_pt) / denom, np.nan)
    
    return scores_car, scores_pt, cdi


# ── Plotting ──────────────────────────────────────────────────
def plot_cdi_comparison(baseline_cdi: np.ndarray,
                        scenario_cdi: np.ndarray,
                        gdf_hex: gpd.GeoDataFrame,
                        output_path: Path) -> None:
    """
    Plots side-by-side CDI maps: baseline vs scenario.
    """
    gdf_hex_m = gdf_hex.to_crs(epsg=32632)
    minx, miny, maxx, maxy = gdf_hex_m.total_bounds

    gdf_hex_m["cdi_baseline"] = baseline_cdi
    gdf_hex_m["cdi_scenario"] = scenario_cdi
    gdf_hex_m["cdi_delta"] = scenario_cdi - baseline_cdi

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("CDI Comparison: Baseline vs PT Enhancement Scenario", fontsize=14, fontweight="bold")

    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for ax, col, title in [
        (axes[0], "cdi_baseline", "Baseline CDI"),
        (axes[1], "cdi_scenario", "Scenario CDI"),
        (axes[2], "cdi_delta", "Delta CDI (improvement)"),
    ]:
        gdf_hex_m[gdf_hex_m[col].isna()].plot(ax=ax, color="#d0d0d0", linewidth=0.1, edgecolor="white")
        valid = gdf_hex_m[gdf_hex_m[col].notna()]

        if col == "cdi_delta":
            norm_delta = mcolors.Normalize(vmin=valid[col].min(), vmax=0)
            valid.plot(ax=ax, column=col, cmap="Greens", norm=norm_delta,
                      linewidth=0.15, edgecolor="white", legend=False)
            sm = cm.ScalarMappable(cmap="Greens", norm=norm_delta)
        else:
            valid.plot(ax=ax, column=col, cmap="RdBu_r", norm=norm,
                      linewidth=0.15, edgecolor="white", legend=False)
            sm = cm.ScalarMappable(cmap="RdBu_r", norm=norm)

        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.75)
        cbar.set_label("CDI" if col != "cdi_delta" else "Δ CDI", fontsize=9)

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────
def main(
  ctx: dict
):
    """Complete pipeline: diagnose → modify GTFS → recompute → compare."""
    logging.basicConfig(level=logging_level)

    with timed():
      logging.info("WHAT-IF SCENARIO: PT Enhancement & CDI Impact")
      
      # Load baseline data
      logging.info("\nLoading baseline data...")
      tt_pt_baseline = pd.read_parquet(PathRegistry.TRAVEL_TIMES_PT)
      centroids = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
      hex_ids = centroids["hex_id"].tolist()
      N = len(hex_ids)
      gdf_hex = gpd.GeoDataFrame(
        centroids,
        geometry=gpd.points_from_xy(centroids["centroid_lon"], centroids["centroid_lat"]),
        crs=CRS_LATLONG
      ).to_crs(CRS_PROJECTED)
      
      baseline_scores = pd.read_parquet(PathRegistry.OPPORTUNITY_SCORES)
      baseline_scores = gpd.GeoDataFrame(
        baseline_scores,
        geometry=gpd.points_from_xy(centroids["centroid_lon"], centroids["centroid_lat"]),
        crs=CRS_LATLONG
      ).to_crs(CRS_PROJECTED)
            
      logging.info(f"Loading PUMS zones for aggregation...")
      zoi = data_reader.ZOI_shape(path=PathRegistry.ZOI_BORDER)
      
      pums_zones_gdf = data_reader.AOI_shapes(
          path=PathRegistry.OD_SHAPE_ZONE,
          aoi_type="od",
          df_around=zoi,
          return_type="inside",
          ).to_crs(CRS_PROJECTED)

      joined = gpd.sjoin(
        gdf_hex[["hex_id", "geometry"]],
        pums_zones_gdf[["geometry", "id_zone"]],
        how='left',
        predicate='within'
      )
      
      joined = joined.groupby('id_zone').agg({'hex_id': list}).reset_index().astype({'id_zone': int})
      
      hex_zone_mapping = dict(zip(joined["id_zone"], joined["hex_id"]))
      
      # Step 1: Diagnose baseline PT coverage
      logging.info("\nAnalyzing baseline PT coverage...")
      coverage_df = analyze_pt_coverage(tt_pt_baseline, gdf_hex, pums_zones_gdf)
      
      # Step 2: Generate synthetic GTFS additions
      logging.info("\nGenerating synthetic GTFS additions...")
      
      # Given the day that we are interested, we load valid service_id from calendar file of original GTFS
      day_of_week = ctx['day_of_week']
      calendar = pd.read_csv(PathRegistry.GTFS_DATA / "calendar.txt")
      weekday_services = calendar[calendar[day_of_week] == 1]["service_id"].iloc[0]
      
      synthetic = generate_synthetic_gtfs_additions(coverage_df,
                                                    centroids,
                                                    service_id = weekday_services,
                                                    hex_zone_mapping = hex_zone_mapping,
                                                    target_coverage = ctx['target_coverage'])

      # Step 3: Modify GTFS
      logging.info("\nModifying GTFS...")
      gtfs_original = PathRegistry.GTFS_DATA
      gtfs_scenario = PathRegistry.GTFS_PT_SCENARIO

      if gtfs_original.exists():
          apply_gtfs_modifications(gtfs_original, synthetic, gtfs_scenario)
      else:
          logging.warning(f"Original GTFS not found at {gtfs_original}")
          logging.warning("Skipping GTFS modification — will use baseline for demonstration")
          gtfs_scenario = gtfs_original

      # Step 4: Recompute PT travel times with modified GTFS
      logging.info("\nRecomputing PT travel times with modified GTFS...")

      if gtfs_scenario.exists():
          hex_coords = np.column_stack([
              gdf_hex.geometry.x.values,
              gdf_hex.geometry.y.values
            ])  
          
          tt_pt_scenario = compute_travel_times(
            hex_ids= gdf_hex["hex_id"].tolist(),
            hex_coords= hex_coords,
            gtfs_path= gtfs_scenario,
            departure_hours= list(range(ctx["start_time_hour"], ctx["end_time_hour"] + 1)),
            max_walk_meters= ctx['max_walk_meters'],
            walk_speed_ms= ctx['walk_speed_ms'],
            max_travel_time_secs= ctx['max_travel_time_secs'],
            day_of_week= ctx['day_of_week'])

      else:
          logging.warning("Using baseline PT matrix for comparison")
          tt_pt_scenario = tt_pt_baseline.copy().values

      # Save travel times on disk
      pd.DataFrame(tt_pt_scenario, index=hex_ids, columns=hex_ids).to_parquet(PathRegistry.TRAVEL_TIMES_PT_SCENARIO)
      # Step 5: Calculate CDI for both scenarios
      logging.info("\nComputing CDI for baseline and scenario...")
      poi_counts = centroids["poi_count"]
      
      pt_opportunities_scores_scenario = getOpportunityScores(
																						tt_matrix = tt_pt_scenario,
																						poi_counts = poi_counts,
																						hex_ids = hex_ids,
																						tau = ctx['tau_sec'],
																						dt = ctx['dt_sec'],
																						t_max = ctx['t_max']
																					)
      
      car_opportunities_scores = baseline_scores["score_car"].values

      scenario_scores = gdf_hex[["hex_id",
                                 "centroid_lat",
                                 "centroid_lon",
                                 "geometry",
                                 'poi_count',
                                 'popolazione'
                                 ]].copy()
      
      scenario_scores["score_car"] = car_opportunities_scores
      scenario_scores["score_pt"] = pt_opportunities_scores_scenario
      denom = pt_opportunities_scores_scenario + car_opportunities_scores
      scenario_scores["cdi"] = np.where(denom > 0, (car_opportunities_scores - pt_opportunities_scores_scenario) / denom, np.nan)
      
      # Step 6: Aggregate to zones
      logging.info("\nAggregating CDI to PUMS zones")
      
      joined = spatial_join_hex_to_zones(scenario_scores, gdf_zones=pums_zones_gdf)
      
      gdf_zones_result_scenario = aggregate_to_zones(joined, pums_zones_gdf)

      # Save results
      gdf_zones_result_scenario.to_parquet(PathRegistry.PT_SCENARIO_CDI_IMPACT)
      logging.info(f"Saved: {PathRegistry.PT_SCENARIO_CDI_IMPACT}")

"""
      # Report
      logging.info("\n" + "="*70)
      logging.info("SCENARIO SUMMARY")
      logging.info("="*70)
      logging.info(f"\nBaseline PT Coverage:")
      logging.info(f"  Mean: {coverage_df['coverage_pct'].mean():.1%}")
      logging.info(f"  Min: {coverage_df['coverage_pct'].min():.1%}")
      logging.info(f"  Max: {coverage_df['coverage_pct'].max():.1%}")

      logging.info(f"\nCDI Change (Baseline → Scenario):")
      logging.info(f"  Mean CDI baseline: {cdi_baseline[~np.isnan(cdi_baseline)].mean():.4f}")
      logging.info(f"  Mean CDI scenario: {cdi_scenario[~np.isnan(cdi_scenario)].mean():.4f}")
      logging.info(f"  Mean delta CDI: {(cdi_scenario - cdi_baseline)[~np.isnan(cdi_baseline)].mean():.4f}")

      logging.info(f"\nZone-Level Impact:")
      logging.info(f"  Zones with CDI reduction: {(comparison_df['delta_cdi'] < 0).sum()}/{len(comparison_df)}")
      logging.info(f"  Mean CDI reduction: {comparison_df[comparison_df['delta_cdi'] < 0]['delta_cdi'].mean():.4f}")

      top_impact = comparison_df.nsmallest(5, "delta_cdi")[
          ["zone_id", "cdi_mean_baseline", "cdi_mean_scenario", "delta_cdi"]
      ]
      logging.info(f"\nTop 5 zones for CDI reduction:")
      for idx, row in top_impact.iterrows():
          logging.info(
              f"  Zone {row['zone_id']:2.0f}: {row['cdi_mean_baseline']:.4f} → "
              f"{row['cdi_mean_scenario']:.4f} (Δ {row['delta_cdi']:.4f})"
          )
"""


if __name__ == "__main__":
    main(ctx = context)
  