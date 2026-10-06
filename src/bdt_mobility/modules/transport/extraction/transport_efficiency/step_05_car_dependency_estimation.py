import json
import numpy as np
import pandas as pd
import geopandas as gpd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import TMP_PATH, logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context
from scipy.integrate._ivp.radau import T

MAIN_OUTPUTS = [
    PlatformOutput("opportunity_scores", PathRegistry.OPPORTUNITY_SCORES, "table"),
]

def utility(t_sec, tau : int):
    """f(t) = (1/τ) * exp(-t/τ) — eq. (3) del paper."""
    return (1 / tau) * np.exp(-t_sec / tau)

def getOpportunityScores(tt_matrix, poi_counts, hex_ids,
                         tau : int, dt : int, t_max : int):
  N = len(hex_ids)
  poi_arr = np.array(poi_counts, dtype=float)
  scores = np.zeros(N)
 
  # Griglia temporale per l'integrazione numerica
  t_grid = np.arange(0, t_max + dt, dt)  # secondi
 
  for i in range(N):
    tt_row = tt_matrix[i]   # travel times dall'esagono i a tutti gli altri
 
    # Calcola P_{h,m}(t) = numero POI raggiungibili entro tempo t
    # Per ogni t nella griglia, conta i POI negli esagoni con tt <= t
    score_i = 0.0
    for t in t_grid:
      if t == 0:
        continue
      reachable_mask = (tt_row <= t) & (~np.isnan(tt_row))
      p_t = poi_arr[reachable_mask].sum()
      # Contributo: f(2t) * P(t) * dt  (il 2t per il viaggio di ritorno)
      score_i += utility(2 * t, tau) * p_t * dt
 
    scores[i] = score_i
 
  return scores

def main(
):
  logging.basicConfig(level=logging_level)
  with timed():
    logging.info(f"Reading relevant input data")
    centroids  = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
    hex_ids    = centroids["hex_id"].tolist()
    poi_counts = centroids["poi_count"].values
    N = len(hex_ids)
    
    # Tavel time with private transport
    tt_car_df = pd.read_parquet(PathRegistry.TRAVEL_TIMES_CAR, engine='pyarrow')
    tt_car = tt_car_df.to_numpy()
    
    # Travel Time with Public Transport
    tt_pt_df = pd.read_parquet(PathRegistry.TRAVEL_TIMES_PT, engine='pyarrow')
    tt_pt = tt_pt_df.to_numpy()
          
    # opportunity scores with cars
    logging.info(f"Computing opportunity scores with cars")
    scores_car = getOpportunityScores(tt_car, 
                                      poi_counts, 
                                      hex_ids,
                                      tau = context['tau_sec'],
                                      dt = context['dt_sec'],
                                      t_max = context['t_max'])
    
    # Opportunity scores with public transport
    scores_pt = getOpportunityScores(tt_pt, poi_counts, hex_ids,
                                     tau = context['tau_sec'],
                                     dt = context['dt_sec'],
                                     t_max = context['t_max'])
    
    # Car dependency index
    denom = scores_car + scores_pt
    cdi = np.where(denom > 0, (scores_car - scores_pt) / denom, np.nan)
    
    cdi_stats = {
        "cdi_mean":                  round(float(np.nanmean(cdi)), 3),
        "cdi_min":                   round(float(np.nanmin(cdi)), 3),
        "cdi_max":                   round(float(np.nanmax(cdi)), 3),
        "pct_car_dependent_cdi_gt0": round(float((cdi > 0).mean()) * 100, 2),
        "pct_pt_friendly_cdi_lt0":   round(float((cdi < 0).mean()) * 100, 2),
    }

    logging.info("Saving CDI statistics")
    # Is like a diagnostic info, save it on the diagnostic foldern
    report_dir = TMP_PATH / "diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    cdi_stats_output_path = report_dir / "cdi_stats.json"

    with open(cdi_stats_output_path, "w", encoding="utf-8") as f:
      json.dump(cdi_stats, f, indent=2)
    
    # ─────────────────────────────────────────
    # Results GeoDataframe building
    # ─────────────────────────────────────────
    gdf_hex = gpd.read_parquet(PathRegistry.HEX_POI_STAT)
    gdf_hex = gdf_hex.merge(
        pd.DataFrame({
            "hex_id":       hex_ids,
            "score_car":    scores_car,
            "score_pt":     scores_pt,
            "cdi":          cdi,
        }),
        on="hex_id", how="left"
    )
    
    # Save results
    gdf_hex.to_parquet(PathRegistry.OPPORTUNITY_SCORES)

if __name__ == "__main__": 
  main()