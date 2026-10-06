import logging
import requests
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("travel_times_car", PathRegistry.TRAVEL_TIMES_CAR, "artifact"),
]

# ─────────────────────────────────────────
#    OSRM TABLE SERVICE (batch)
#    Calculates an NxN matrix of travel times in a single request
#    Public API limit: ~100 coordinates per request
# ─────────────────────────────────────────

def osrm_table_batch(origins_idx, 
                     destinations_idx, 
                     all_coords,
                     osrm_base_url,
                     logging):
    """
    Call OSRM /table for a subset of origins and destinations.
    Returns a [len(origins) x len(destinations)] matrix of durations in seconds.
    """
    # Build coordinate string: all coordinates used (origins + destinations)
    all_idx = list(dict.fromkeys(origins_idx + destinations_idx))
    coord_str = ";".join(f"{all_coords[i][0]},{all_coords[i][1]}" for i in all_idx)
 
    # Local indeces for OSRM request
    idx_map = {global_i: local_i for local_i, global_i in enumerate(all_idx)}
    src_str = ";".join(str(idx_map[i]) for i in origins_idx)
    dst_str = ";".join(str(idx_map[i]) for i in destinations_idx)
 
    url = f"{osrm_base_url}/table/v1/driving/{coord_str}"
    params = {
        "sources": src_str,
        "destinations": dst_str,
        "annotations": "duration"
    }
 
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            if data.get("code") == "Ok":
                return np.array(data["durations"])
            else:
                logging.warning(f"OSRM error: {data.get('code')} — {data.get('message')}")
                return None
        except Exception as e:
            logging.error(f"Tentativo {attempt+1} fallito: {e}")
            time.sleep(2 ** attempt)
    return None
 
 
def apply_traffic_correction(durations_sec : int,
                             delta_traffic : float,
                             parking_buffer_sec : int,                             
                             ):
    """Apply the correction from the paper: (1 + Δ_traffic) * t_OSRM + 15 min"""
    return (1 + delta_traffic) * durations_sec + parking_buffer_sec

def compute_travel_time_matrix(hex_ids, 
                               coords,
                               osrm_base_url : str,
                               batch_size : int = 80,
                               delta_traffic : float = 0.22,
                               parking_buffer_sec : int = 15 * 60,
                               ):
    

  N = len(hex_ids)
  
  tt_car = np.full((N, N), np.nan)
 
  # Prepare batch blocks
  origin_batches = [list(range(i, min(i + batch_size, N)))
                    for i in range(0, N, batch_size)]
  dest_batches   = [list(range(i, min(i + batch_size, N)))
                    for i in range(0, N, batch_size)]

  total_batches = len(origin_batches) * len(dest_batches)
  pbar = tqdm(total=total_batches, desc="Batches OSRM")
  
  
  for o_batch in origin_batches:
    for d_batch in dest_batches:
        mat = osrm_table_batch(o_batch, d_batch, coords, osrm_base_url, logging)
        if mat is not None:
            corrected = apply_traffic_correction(mat,
                                                 delta_traffic ,
                                                 parking_buffer_sec,
                                                 )
            for li, gi in enumerate(o_batch):
                for lj, gj in enumerate(d_batch):
                    tt_car[gi][gj] = corrected[li][lj]
        pbar.update(1)
        time.sleep(0.3) 
 
  pbar.close()

  # 
  nan_frac = np.isnan(tt_car).mean()
  
  if nan_frac > 0.05:
      logging.warning("Many NaNs: Consider using a local OSRM instance")
      
  return tt_car

def main(
    ctx: dict
):
  logging.basicConfig(level=logging_level)
  with timed():
    logging.info(f"Reading relevant input data")
    hex_centroids = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
    coords = list(zip(hex_centroids['centroid_lon'], hex_centroids['centroid_lat']))
    hex_ids = hex_centroids['hex_id'].tolist()
    
    logging.info(f"Computing travel time matrix for {len(hex_ids)} hexagons")
    tt_car_matrix = compute_travel_time_matrix(hex_ids, 
                                                coords,
                                                batch_size = context['batch_size'],
                                                osrm_base_url = context['osrm_base_url'],
                                                parking_buffer_sec = context['parking_buffer_sec'],)
    
    travel_times_car_df = pd.DataFrame(tt_car_matrix, index=hex_ids, columns=hex_ids)
    
    travel_times_car_df.to_parquet(PathRegistry.TRAVEL_TIMES_CAR, engine='pyarrow', index=True)
  
if __name__ == "__main__":
  
  main(ctx=context)