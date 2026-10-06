"""
STEP 5b — Population Accessibility Score (vectorized)
Reads population from hex_centroids.csv (already computed by step 05a).

Formula (transposed from paper):
  A_{j,m} = Σ_t  f(2t) · Σ_{i: tt[i,j] ≤ t} pop[i]  · Δt

Vectorized implementation:
  - Sorts travel time matrix columns by travel time
  - Computes weighted cumsum of population in a single numpy operation
  → No Python loops over hexagons or time steps
"""

import pandas as pd
import numpy as np
import geopandas as gpd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("population_accessibility", PathRegistry.POP_ACCESSIBILITY, "table"),
]

def survival(t_sec, tau: int = 1800):
    """
    Decay function: F(t) = ∫_t^∞ f(2s) ds = (1/2) · exp(-2t/τ)

    Represents the contribution of an origin at travel time t
    to the accessibility score of a destination.

    Args:
        t_sec (np.ndarray | float): travel time in seconds.
        tau   (int):                decay parameter in seconds (default 1800).

    Returns:
        np.ndarray | float: decay weight in [0, 0.5].
    """
    return 0.5 * np.exp(-2.0 * t_sec / tau)


def compute_population_score_vectorized(tt_matrix, pop_array, tau: int = 1800, t_max: int = 5400):
    """
    Vectorized computation of the Population Accessibility Score.

    For each destination hexagon j, aggregates the weighted population
    of all origin hexagons i reachable within t_max seconds:
      A[j] = Σ_i  pop[i] · F(tt[i, j])

    Args:
        tt_matrix (pd.DataFrame): (N, N) matrix — tt[i, j] = travel time
                                from origin i to destination j, in seconds.
        pop_array (np.ndarray): (N,) array — population per origin hexagon.
        tau       (int):        decay parameter in seconds (default 1800).
        t_max     (int):        maximum travel time threshold in seconds (default 5400).

    Returns:
        np.ndarray: (N,) array of accessibility scores, one per destination hexagon.
    """
    # Mask invalid values and those exceeding t_max
    valid = (~np.isnan(tt_matrix)) & (tt_matrix > 0) & (tt_matrix <= t_max)

    # Weight matrix: F(tt[i, j]) for each origin-destination pair
    # shape (N, N) — weights[i, j] = contribution of origin i to destination j
    weights = np.where(valid, survival(tt_matrix, tau), 0.0)

    # Accessibility score for each destination j computed as a
    # single matrix-vector product: A[j] = Σ_i pop[i] · weights[i, j]
    scores = pop_array @ weights  # shape (N,)

    return scores


def main(ctx: dict):
    """
    Main pipeline:
      1. Loads hexagon grid, centroids, and population array.
      2. Loads car and public transport travel time matrices.
      3. Computes vectorized Population Accessibility Score for both modes.
      4. Saves results as GeoJSON and CSV.
    """
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Reading relevant input data")
        gdf_hex   = gpd.read_parquet(PathRegistry.HEX_POI_STAT).drop(columns=["poi_count"], errors="ignore")
        centroids = pd.read_parquet(PathRegistry.HEX_POP_POI_STAT)
        hex_ids   = centroids["hex_id"].tolist()
        N         = len(hex_ids)
        pop_array = centroids["popolazione"].fillna(0).values.astype(float)

        travel_times_car_df = pd.read_parquet(PathRegistry.TRAVEL_TIMES_CAR, engine='pyarrow')
        travel_times_pt_df = pd.read_parquet(PathRegistry.TRAVEL_TIMES_PT, engine='pyarrow')
        
        tt_car = travel_times_car_df.to_numpy()
        tt_pt  = travel_times_pt_df.to_numpy()

        logging.info("Computing car accessibility score.")
        scores_car = compute_population_score_vectorized(
            tt_car,
            pop_array,
            tau=ctx['tau_sec'],
            t_max=ctx['t_max']
        )

        logging.info("Computing public transport accessibility score.")
        scores_pt = compute_population_score_vectorized(
            tt_pt,
            pop_array,
            tau=ctx['tau_sec'],
            t_max=ctx['t_max']
        )

        # Build results dataframe
        results_df = pd.DataFrame({
            "hex_id":         hex_ids,
            "popolazione":    pop_array,
            "access_pop_car": scores_car,
            "access_pop_pt":  scores_pt,
            "access_ratio":   np.where(scores_car > 0, scores_pt / scores_car, np.nan),
        })

        # Save GeoJSON and CSV outputs
        logging.info("Saving results to file")
        gdf_results = gdf_hex.merge(results_df, on="hex_id", how="left")
        gdf_results.to_parquet(
            PathRegistry.POP_ACCESSIBILITY
        )


if __name__ == "__main__":
    main(context)