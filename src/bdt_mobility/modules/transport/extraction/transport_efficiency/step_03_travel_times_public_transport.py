
import gc
import json
import os
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import pickle

from bdt_mobility.utils.paths import TMP_DATA_PATH, PathRegistry
from bdt_mobility.utils.config import TMP_PATH, logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("travel_time_public_transport", PathRegistry.TRAVEL_TIMES_PT, "artifact"),
]

"""
STEP 2 — Public transit travel times with CSA rewritten from scratch
No external routing libraries required.
 
CSA logic:
  - For each departure time (8 a.m.–9 p.m.): run CSA from each origin hex
  - CSA propagates the minimum arrival time to each stop
  - Transferring between nearby stops via walking is allowed
  - Initial walk (hexagon → stop) and final walk (stop → hexagon) included
  - Result: average of travel times across departure hours
 
"""
  
def load_and_filter_services(gtfs_path, day: str):
  """
  Load GTFS files and filter services active on a given weekday.

  Parameters
  ----------
  gtfs_path : Path
      Path to GTFS directory.
  day : str
      Weekday column name in calendar.txt (e.g. 'monday').

  Returns
  -------
  stop_times_wd : pd.DataFrame
      Filtered stop_times for selected weekday.
  stops : pd.DataFrame
      Stops dataset.
  """
  
  # Load gtfs data
  stops      = pd.read_csv(gtfs_path / "stops.txt")
  stop_times = pd.read_csv(gtfs_path / "stop_times.txt", dtype={"trip_id": str})
  trips      = pd.read_csv(gtfs_path / "trips.txt", dtype={"trip_id": str, "service_id": str})
  calendar   = pd.read_csv(gtfs_path / "calendar.txt", dtype={"service_id": str})
  
  # Select service_ids that are active on Mondays (a typical weekday)
  weekday_services = calendar[calendar[day] == 1]["service_id"].tolist()
  weekday_trips_ids    = trips[trips["service_id"].isin(weekday_services)]["trip_id"].tolist()
  stop_times_wd    = stop_times[stop_times["trip_id"].isin(weekday_trips_ids)].copy()
  
  def time_to_seconds(t):
    """Converte HH:MM:SS in secondi da mezzanotte (gestisce orari >24h del GTFS)."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s
  
  stop_times_wd["dep_sec"] = stop_times_wd["departure_time"].apply(time_to_seconds)
  stop_times_wd["arr_sec"] = stop_times_wd["arrival_time"].apply(time_to_seconds)
  stop_times_wd['stop_sequence'] = pd.to_numeric(stop_times_wd['stop_sequence'], errors='coerce')
  stop_times_wd = stop_times_wd.dropna(subset=['dep_sec', 'arr_sec', 'stop_sequence']).copy()
  
  stop_times_wd["dep_sec"] = stop_times_wd["dep_sec"].astype(int)
  stop_times_wd["arr_sec"] = stop_times_wd["arr_sec"].astype(int)
  stop_times_wd["stop_sequence"] = stop_times_wd["stop_sequence"].astype(int)
  
  return stop_times_wd, stops

def CSA_preparation(st, max_walk_m : int = 1250, walk_speed_s : float = 5000 / 3600):

  """
  Build CSA connection list from stop_times.

  Each connection represents a movement between two consecutive stops.

  Returns
  -------
  connections : pd.DataFrame
  """

  st_sorted = st.sort_values(by=["trip_id", "stop_sequence"]).reset_index(drop=True)

  # Each connection links consecutive stops (row i -> row i+1) within the same trip.
  nxt_trip_id = st_sorted["trip_id"].shift(-1)
  nxt_stop_id = st_sorted["stop_id"].shift(-1)
  nxt_arr_sec = st_sorted["arr_sec"].shift(-1)

  same_trip = nxt_trip_id == st_sorted["trip_id"]
  # Skip if arr_time < dep_time (data error)
  valid = same_trip & (nxt_arr_sec >= st_sorted["dep_sec"])

  connections = pd.DataFrame({
    "dep_stop": st_sorted["stop_id"][valid].values,
    "arr_stop": nxt_stop_id[valid].values.astype(st_sorted["stop_id"].dtype),
    "dep_time": st_sorted["dep_sec"][valid].values,
    "arr_time": nxt_arr_sec[valid].values.astype(int),
    "trip_id":  st_sorted["trip_id"][valid].values,
  }).sort_values("dep_time").reset_index(drop=True)

  return connections

def stop2stop_travel_time(stops: pd.DataFrame,
                          max_walk_m : int = 1250,
                          walk_speed_s : float = 5000 / 3600):
  
  """
    Compute walking transfers between nearby stops.

    Returns
    -------
    coords : np.ndarray
        Stop coordinates in projected CRS.
    stop_transfers : dict
        Mapping: stop_idx -> list of (neighbor_idx, walk_time_sec)
    stop_id_to_idx : dict
        Mapping stop_id -> index
  """
  
  gdf_stops = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
    crs=CRS_LATLONG
  ).to_crs(CRS_PROJECTED)
  
  stop_ids_list = stops["stop_id"].tolist()
  stop_id_to_idx = {s: i for i, s in enumerate(stop_ids_list)}
  S = len(stop_ids_list)

  # For each stop, list of (nearby_stop_idx, walk_time_sec)
  stop_transfers = {i: [] for i in range(S)}
  
  coords = np.column_stack([
    gdf_stops.geometry.x.values,
    gdf_stops.geometry.y.values
  ])
  
  for i in range(S):
    dists = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
    nearby = np.where((dists > 0) & (dists <= max_walk_m))[0]
    for j in nearby:
        walk_sec = int(dists[j] / walk_speed_s)
        stop_transfers[i].append((j, walk_sec))
 
  
  return coords, stop_transfers, stop_id_to_idx

def _csa_earliest_arrival(origin_hex_idx, dep_time_sec,
                          conn_dep_idx, conn_arr_idx,
                          conn_dep_time, conn_arr_time, conn_trip_code,
                          n_stops, n_trips,
                          transfer_indptr, transfer_nb_idx, transfer_nb_walk,
                          hex_indptr, hex_stop_idx, hex_walk_sec,
                          max_tt_secs):
  """
  Run Connection Scan Algorithm (CSA) from one origin hex.

  Stop transfers and hex-to-stop mappings are given as flat CSR-style arrays
  (indptr + values), and trips as integer codes: only NumPy arrays are read
  here, so the whole input set is compact (~40 MB) and cheap to ship to the
  spawned worker processes.

  Returns
  -------
  earliest : np.ndarray
        Earliest arrival time at each stop.
  """

  INF = 10**9
  earliest = np.full(n_stops, INF, dtype=np.int64)  # Earliest arrival at each stop
  in_trip  = np.zeros(n_trips, dtype=bool)  # trip code → Are we already on this trip?

  # Initialize: stops within walking distance of the starting point
  for p in range(hex_indptr[origin_hex_idx], hex_indptr[origin_hex_idx + 1]):
    stop_idx = hex_stop_idx[p]
    arr = dep_time_sec + hex_walk_sec[p]
    if arr < earliest[stop_idx]:
      earliest[stop_idx] = arr
      # Immediately extend the walking routes from the starting stop
      for q in range(transfer_indptr[stop_idx], transfer_indptr[stop_idx + 1]):
        t = arr + transfer_nb_walk[q]
        if t < earliest[transfer_nb_idx[q]]:
          earliest[transfer_nb_idx[q]] = t

  # Starting index in the connection list (first available connection)
  start_idx = np.searchsorted(conn_dep_time, dep_time_sec)

  # Scansione connessioni in ordine cronologico
  for k in range(start_idx, len(conn_dep_time)):
    dep_t  = conn_dep_time[k]

    # Optimization: if all future connections start
    # after dep_time + MAX_TT, they can't improve anything
    if dep_t > dep_time_sec + max_tt_secs:
      break

    arr_t  = conn_arr_time[k]
    trip   = conn_trip_code[k]

    dep_s_idx = conn_dep_idx[k]
    arr_s_idx = conn_arr_idx[k]
    if dep_s_idx < 0 or arr_s_idx < 0:
      continue

    # I can board this connection if:
    # (a) I'm already on this trip, OR
    # (b) I arrived at the departure stop on time
    can_board  = in_trip[trip] or (earliest[dep_s_idx] <= dep_t)

    if can_board:
      in_trip[trip] = True
      if arr_t < earliest[arr_s_idx]:
        earliest[arr_s_idx] = arr_t
        # Extend transfers on foot from the arrival stop
        # (allows for transfers by walking to nearby stops)
        for q in range(transfer_indptr[arr_s_idx], transfer_indptr[arr_s_idx + 1]):
          t = arr_t + transfer_nb_walk[q]
          if t < earliest[transfer_nb_idx[q]]:
            earliest[transfer_nb_idx[q]] = t

  return earliest


# Read-only inputs for _compute_origin_row; each spawned worker receives the
# dict once through the pool initializer.
_worker_data = {}


def _worker_init(worker_data):
  _worker_data.update(worker_data)


def _compute_origin_row(i):
  """Compute one origin hexagon's row of the travel-time matrix (mean over departure hours)."""
  d = _worker_data
  N = d["N"]
  INF = 10**9

  tt_sum   = np.zeros(N)
  tt_count = np.zeros(N, dtype=int)

  for dep_sec in d["departure_secs"]:
      # CSA: earliest arrival at each stop
    earliest = _csa_earliest_arrival(
      i, dep_sec,
      d["conn_dep_idx"], d["conn_arr_idx"],
      d["conn_dep_time"], d["conn_arr_time"], d["conn_trip_code"],
      d["n_stops"], d["n_trips"],
      d["transfer_indptr"], d["transfer_nb_idx"], d["transfer_nb_walk"],
      d["hex_indptr"], d["hex_stop_idx"], d["hex_walk_sec"],
      d["max_tt_secs"]
    )

    # Convert stops → destination hexagons
    # For each destination hexagon: min travel time = min(earliest[stop] + final_walk)
    #
    # Vectorized re-expression of the original per-hex/per-stop double loop.
    # NOTE: reproduces the original loop's behavior exactly, including the
    # fact that "best" is accumulated into tt_sum/tt_count once per reached
    # stop (not just once per hex) -- i.e. tt_sum[j]/tt_count[j] receive one
    # contribution (of the running-minimum "best" at that point) for every
    # stop near hexagon j that CSA reached, not only for the closest one.
    totals   = earliest[d["hex_stop_idx"]] + d["hex_walk_sec"] - dep_sec
    reached  = earliest[d["hex_stop_idx"]] < INF
    cand_ok  = reached & (totals > 0) & (totals <= d["max_tt_secs"])
    vals     = np.where(cand_ok, totals, INF)

    running_best = np.empty_like(vals)
    for j in range(N):
      s, e = d["hex_indptr"][j], d["hex_indptr"][j + 1]
      if e > s:
        running_best[s:e] = np.minimum.accumulate(vals[s:e])

    emit = reached & (running_best < INF)
    if emit.any():
      tt_sum   += np.bincount(d["flat_hex_j"][emit], weights=running_best[emit], minlength=N)
      tt_count += np.bincount(d["flat_hex_j"][emit], minlength=N)

  # Mean on departure hours
  row = np.full(N, np.nan)
  valid = tt_count > 0
  row[valid] = tt_sum[valid] / tt_count[valid]
  return i, row

def compute_travel_times(hex_ids: list,
                         hex_coords: np.ndarray,
                         day_of_week: int,
                         departure_hours: list,
                         max_walk_meters: float,
                         walk_speed_ms: float,
                         max_travel_time_secs: int,
                         gtfs_path: Path) -> np.ndarray:
  
  N = len(hex_ids)
  
  logging.info("Preprocessing GTFS data")
  st, stops = load_and_filter_services(gtfs_path, 
                                            day = day_of_week
                                            )
      
  logging.info("Connection Scan Algorithm preparation")
  connections = CSA_preparation(st, 
                                    max_walk_m = max_walk_meters, 
                                    walk_speed_s = walk_speed_ms
                                    )
      
  # Convert everything the CSA kernel reads to NumPy arrays: trips become
  # integer codes and stop_ids integer indices. Besides being faster, this
  # keeps the worker dataset compact (~40 MB) and quick to pickle to the
  # spawned worker processes.
  conn_dep_time = connections["dep_time"].values.astype(int)
  conn_arr_time = connections["arr_time"].values.astype(int)
  conn_trip_code, trip_uniques = pd.factorize(connections["trip_id"])
  conn_trip_code = conn_trip_code.astype(np.int64)
  n_trips = len(trip_uniques)
  
  logging.info("Calculating walking distances between stops")
  stops_coords, stop_transfers, stop_id_to_idx = stop2stop_travel_time(stops,
                                                                           max_walk_meters,
                                                                           walk_speed_ms
                                                                           )
  n_stops = len(stop_id_to_idx)
  
  conn_dep_idx = connections["dep_stop"].map(stop_id_to_idx).fillna(-1).astype(np.int64).values
  conn_arr_idx = connections["arr_stop"].map(stop_id_to_idx).fillna(-1).astype(np.int64).values
  
  # Flatten stop_transfers into CSR-style arrays:
  # neighbors of stop s live at positions transfer_indptr[s]:transfer_indptr[s+1]
  transfer_counts  = np.array([len(stop_transfers[s]) for s in range(n_stops)], dtype=np.int64)
  transfer_indptr  = np.concatenate([[0], np.cumsum(transfer_counts)])
  transfer_nb_idx  = np.array([nb for s in range(n_stops) for nb, _ in stop_transfers[s]], dtype=np.int64)
  transfer_nb_walk = np.array([w for s in range(n_stops) for _, w in stop_transfers[s]], dtype=np.int64)
  
  # hex-to-stop mapping, same CSR layout: stops reachable on foot from
  # hexagon j live at positions hex_indptr[j]:hex_indptr[j+1]
  hex_counts_list, hex_stop_idx_list, hex_walk_sec_list = [], [], []
  for i in range(N):
    dists = np.sqrt(((stops_coords - hex_coords[i]) ** 2).sum(axis=1))
    nearby = np.where(dists <= max_walk_meters)[0]
    hex_counts_list.append(len(nearby))
    hex_stop_idx_list.extend(nearby)
    hex_walk_sec_list.extend(int(dists[j] / walk_speed_ms) for j in nearby)
  hex_counts   = np.array(hex_counts_list, dtype=np.int64)
  hex_indptr   = np.concatenate([[0], np.cumsum(hex_counts)])
  hex_stop_idx = np.array(hex_stop_idx_list, dtype=np.int64)
  hex_walk_sec = np.array(hex_walk_sec_list, dtype=np.int64)
  flat_hex_j   = np.repeat(np.arange(N), hex_counts)
  
  covered = int((hex_counts > 0).sum())
  
  # Free the large intermediate structures; only the compact arrays below
  # are needed from here on.
  del st, connections, stop_transfers
  gc.collect()
  
  worker_data = {
    "N": N,
    "n_stops": n_stops,
    "n_trips": n_trips,
    "departure_secs": [hour * 3600 for hour in departure_hours],
    "max_tt_secs": max_travel_time_secs,
    "conn_dep_idx": conn_dep_idx,
    "conn_arr_idx": conn_arr_idx,
    "conn_dep_time": conn_dep_time,
    "conn_arr_time": conn_arr_time,
    "conn_trip_code": conn_trip_code,
    "transfer_indptr": transfer_indptr,
    "transfer_nb_idx": transfer_nb_idx,
    "transfer_nb_walk": transfer_nb_walk,
    "hex_indptr": hex_indptr,
    "hex_stop_idx": hex_stop_idx,
    "hex_walk_sec": hex_walk_sec,
    "flat_hex_j": flat_hex_j,
  }
  
  tt_pt = np.full((N, N), np.nan)
  origins = [i for i in range(N) if hex_counts[i] > 0]  # hexagons without nearby stops keep NaN rows
  
  n_workers = min(os.cpu_count() or 1, len(origins))
  logging.info(f"Running CSA for {len(origins)} origin hexagons on {n_workers} workers")
  
  # "spawn" (rather than fork) works identically on Linux, macOS and Windows:
  # each worker starts a fresh interpreter and receives worker_data once via
  # the initializer.
  with get_context("spawn").Pool(processes=n_workers,
                                 initializer=_worker_init,
                                 initargs=(worker_data,)) as pool:
    for i, row in pool.imap_unordered(_compute_origin_row, origins, chunksize=4):
      tt_pt[i] = row
  
  logging.info("Computing diagnostics after CSA computation")
  return tt_pt
      
      
def main(
  ctx: dict
):
  logging.basicConfig(level=logging_level)
  with timed():
    departure_hours = list(range(ctx["start_time_hour"], ctx["end_time_hour"] + 1)) 
    
    logging.info("Reading relevant input data")
      
    # Read hexagons dataset
    centroids = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
    hex_ids = centroids["hex_id"].tolist()
    N = len(hex_ids)
    gdf_centroids = gpd.GeoDataFrame(
      centroids,
      geometry=gpd.points_from_xy(centroids["centroid_lon"], centroids["centroid_lat"]),
      crs=CRS_LATLONG
    ).to_crs(CRS_PROJECTED)
      
    hex_coords = np.column_stack([
      gdf_centroids.geometry.x.values,
      gdf_centroids.geometry.y.values
    ])  
    
    tt_pt = compute_travel_times(
      hex_ids = hex_ids,
      hex_coords = hex_coords,
      day_of_week = ctx['day_of_week'],
      departure_hours = departure_hours,
      max_walk_meters = ctx['max_walk_meters'],
      walk_speed_ms = ctx['walk_speed_ms'],
      max_travel_time_secs = ctx['max_travel_time_secs'],
      gtfs_path = PathRegistry.GTFS_DATA
    )
    
    travel_times_pt_df = pd.DataFrame(tt_pt, index=hex_ids, columns=hex_ids)
    
    travel_times_pt_df.to_parquet(PathRegistry.TRAVEL_TIMES_PT, engine='pyarrow', index=True)

  
def tt_matrix_diagnostic(tt_pt):
    """
    Computes diagnostic statistics for a travel time matrix
    and writes results to TRANSPORT_EFFICIENCY_PATH.

    Parameters
    ----------
    tt_pt            : np.ndarray  - NxN travel time matrix in seconds
    logging          : logging     - logger instance
    transport_mode   : str         - label used in the output filename ("pt" or "car")
    """
    # ── Coverage ─────────────────────────────────────────────
    coverage = (~np.isnan(tt_pt)).mean()
    flat     = tt_pt[~np.isnan(tt_pt) & (tt_pt > 0)] / 60  # convert to minutes

    # ── Unreachable origins / destinations ───────────────────
    unreachable_origins = int(np.isnan(tt_pt).all(axis=1).sum())
    unreachable_dests   = int(np.isnan(tt_pt).all(axis=0).sum())

    # ── Build stats dict ─────────────────────────────────────
    stats = {
        "coverage_pct":           round(float(coverage) * 100, 2),
        "mean_travel_time_min":   round(float(flat.mean()), 2),
        "median_travel_time_min": round(float(np.median(flat)), 2),
        "p10_travel_time_min":    round(float(np.percentile(flat, 10)), 2),
        "p90_travel_time_min":    round(float(np.percentile(flat, 90)), 2),
        "unreachable_origins":    unreachable_origins,
        "unreachable_dests":      unreachable_dests,
    }
    # ── Write to file ─────────────────────────────────────────

    report_dir = TMP_PATH / "diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "travel_time_pt_diagnostic.json" ##TODO save somewhere else

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
if __name__ == "__main__":
  
  main(ctx = context)


  