import gc
import polars as pl
import numpy as np

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.modules.transport.extraction.od_routing.edge_matrix import (
    inflow_per_edge_min_granularity,
    preprocess_trips,
    traffic_per_edge_min_granularity,
    travel_time_per_edge_min_granularity,
)
from bdt_mobility.modules.transport.extraction.od_routing.utils import melt_matrix
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("metrics_per_road_from_simulator", PathRegistry.METRICS_PER_ROAD_SIM, "table"),
]


def compute_agg_granularity_metrics(
    trips: pl.DataFrame,
    edges: pl.DataFrame,
    time_step: int,
    aggregation_step: int,
    first_hour: int,
    seed: int = 0
) -> pl.DataFrame:
    ### Compute traffic = the max number of vehicles circulating at the same time
    # Compute the occupancy per each link per each time instant
    occ_matrix, link_occ_map = traffic_per_edge_min_granularity(
        trips=trips, edges=edges, time_step=time_step, t_min=first_hour*3600
    )
    # Aggregation in time buckets per each link
    nrow, total_cols = occ_matrix.shape
    j = aggregation_step // time_step
    ncol = total_cols * time_step // aggregation_step
    occ_matrix = occ_matrix.reshape(nrow, ncol, j).max(axis=2)
    # Melt: from matrix to dataframe
    traffic_mean = (
		melt_matrix(occ_matrix, link_occ_map, time_step=aggregation_step, t_min=first_hour*3600)
		.rename({"counts": "max_traffic_volume", "spatial_unit": "link_id", "t": "start_t"})
        .with_columns(
            ((pl.col("start_t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
		.sort(["link_id", "agg_time_unit"])
	)
    del occ_matrix
    gc.collect()

    ### Compute inflow = the tot number of vehicles passing in the link
    # Compute the inflow per each link per each time instant
    inflow_matrix, link_inf_map = inflow_per_edge_min_granularity(
        trips=trips, edges=edges, time_step=time_step, t_min=first_hour*3600
    )
    # Aggregation in time buckets per each link
    nrow, total_cols = inflow_matrix.shape
    j = aggregation_step // time_step
    ncol = total_cols * time_step // aggregation_step
    inflow_matrix = inflow_matrix.reshape(nrow, ncol, j).sum(axis=2)
    # Melt: from matrix to dataframe
    inflow_mean = (
		melt_matrix(inflow_matrix, link_inf_map, time_step=aggregation_step, t_min=first_hour*3600)
		.rename({"counts": "traffic_volume", "spatial_unit": "link_id", "t": "start_t"})
        .with_columns(
            ((pl.col("start_t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
		.sort(["link_id", "agg_time_unit"])
	)
    del inflow_matrix
    gc.collect()

    ### Compute travel times
    # Compute the average time per each link per each time instant
    tt_matrix, link_tt_map = travel_time_per_edge_min_granularity(
        trips=trips, edges=edges, time_step=time_step, t_min=first_hour*3600
    )
    # Aggregation in time buckets per each link
    nrow, total_cols = tt_matrix.shape
    j = aggregation_step // time_step
    ncol = total_cols * time_step // aggregation_step
    tt_matrix = np.where(tt_matrix > 0, tt_matrix, np.nan)
    tt_matrix = tt_matrix.reshape(nrow, ncol, j)
    tt_matrix = np.nanmean(tt_matrix, axis=2)
    # Melt: from matrix to dataframe
    tt_mean = (
		melt_matrix(tt_matrix, link_tt_map, time_step=aggregation_step, t_min=first_hour*3600)
		.rename({"counts": "average_travel_time", "spatial_unit": "link_id", "t": "start_t"})
        .with_columns(
            ((pl.col("start_t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
		.sort(["link_id", "agg_time_unit"])
	)
    del tt_matrix
    gc.collect()

    return (
        traffic_mean
        .join(tt_mean, on=["link_id", "agg_time_unit", "start_t"])
        .join(inflow_mean, on=["link_id", "agg_time_unit", "start_t"])
        .join(edges, on="link_id")
        .with_columns([
            pl.when(pl.col("average_travel_time") < 0)
                .then(None)
                .otherwise(pl.col("average_travel_time"))
                .alias("average_travel_time"),

            (pl.col("length")/pl.col("maxspeed")).alias("free_travel_time").cast(pl.Float32),

            (pl.col("traffic_volume")/pl.col("length")*pl.col("n_lanes")).alias("traffic_density").cast(pl.Float32),
            
        ])
        .with_columns([
            pl.when(pl.col("free_travel_time") != 0)
                .then(pl.col("length") / pl.col("free_travel_time"))
                .otherwise(0.0)
                .alias("free_speed"),
            
            pl.when((pl.col("average_travel_time") != 0) & (pl.col("average_travel_time").is_not_null()))
                .then(pl.col("length") / pl.col("average_travel_time"))
                .otherwise(0.0)
                .alias("average_speed")
        ])
        .with_columns([
            (-pl.col("average_speed") + pl.col("free_speed")).alias("delta_speed"),
            
            (pl.col("average_speed") / pl.col("free_speed")).alias("ratio_speed"),
            
            pl.lit(seed).alias("seed").cast(pl.Int32)
        ])
    )
    

def average_seed_results(
    df: pl.DataFrame,
) -> pl.DataFrame:
    grp_cols = ['link_id', 'orig', 'dest', 'length', 'maxspeed', 'n_lanes', 'zone_id', 'agg_time_unit', 'seed', 'start_t', 'free_speed', 'free_travel_time']
    return (
        df.group_by(grp_cols)
        .agg(
            pl.all().exclude(grp_cols, "seed").mean().name.suffix('_mean'),
            pl.all().exclude(grp_cols, "seed").std().name.suffix('_std')
        )
        .sort(by=["link_id", "agg_time_unit"])
    )


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info(f"Computing and saving edge traffic from vehicles")
        df_ed = (
            pl.read_csv(PathRegistry.SIM_EDGES)
            [['id', 'u', 'v', 'length', 'maxspeed_mps', 'lanes', 'id_zone']]
            .rename({'id':'link_id', 'u': 'orig', 'v':'dest', 'maxspeed_mps': 'maxspeed', 'lanes': 'n_lanes', 'id_zone': 'zone_id'})
            .cast({"link_id": pl.Utf8})
        )

        keys = ["time_step", "aggregation_step", "first_hour"]
        params = {k: context[k] for k in keys if k in context}
        list_seeds = [i for i in range(context['total_seeds'])]

        df_t_final = pl.DataFrame()
        for i_seed in list_seeds:
            logging.debug(f"Computing metrics for seed {i_seed}")
            df_trips = preprocess_trips(
                trips=pl.read_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)),
                platoons=pl.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=i_seed)),
                edges=df_ed
            )
            df_t_final = pl.concat([
                df_t_final, 
                    compute_agg_granularity_metrics(
                    trips=df_trips, 
                    edges=df_ed,
                    seed=i_seed,
                    **params
                )
            ])

        logging.debug(f"Avg results")
        df_t_final = average_seed_results(df_t_final)

        logging.info(f"Saving edge metrics")
        df_t_final.write_parquet(PathRegistry.METRICS_PER_ROAD_SIM)


if __name__ == "__main__":
    main()