import numpy as np
import polars as pl
import pandas as pd

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.modules.transport.extraction.od_routing.utils import (
    fold_matrix_24h,
    melt_matrix
)
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils import data_reader, constants
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("inflow_per_zone_from_simulator", PathRegistry.INFLOW_PER_ZONE_SIM, "table"),
    PlatformOutput("traffic_per_zone_from_simulator", PathRegistry.TRAFFIC_PER_ZONE_SIM, "table"),
    PlatformOutput("starting_per_zone_from_simulator", PathRegistry.STARTING_PER_ZONE_SIM, "table"),
    PlatformOutput("inflow_per_zone_from_simulator_processed", PathRegistry.INFLOW_PER_ZONE_SIM_PROCESSED, "table"),
    PlatformOutput("traffic_per_zone_from_simulator_processed", PathRegistry.TRAFFIC_PER_ZONE_SIM_PROCESSED, "table"),
    PlatformOutput("starting_per_zone_from_simulator_processed", PathRegistry.STARTING_PER_ZONE_SIM_PROCESSED, "table"),
]


def preprocess_nodes(node_file):
    nodes = (
        pl.read_csv(node_file)
        [['id', "is_in_zoi", "id_zone"]]
        .rename({'id': 'node_id'})
    )
    return nodes

def preprocess_edges(edge_file):
    edges = (
        pl.read_csv(edge_file)
        [['id', 'u', 'v', 'id_zone']]
        .rename({'id': 'link_id'})
        .cast({"link_id": pl.Utf8})
    )
    return edges

def preprocess_trips(veh, nodes, edges, platoons):
    trips_df = (
        veh
        .drop("x", "s", "v", "attribute", "dn", strict=False)
        .rename({"name": "platoon_id", "link": "link_id"})
        .cast({"orig": int, "dest": int, "platoon_id": int})
        .join(
            edges.select(
                "link_id",
                pl.col("id_zone").alias("edge_id_zone"),
            ),
            on="link_id",
            how="left"
        )
        .join(
            nodes.select(
                pl.col("node_id").alias("orig"),
                pl.col("id_zone").alias("orig_id_zone"),
                pl.col("is_in_zoi").alias("orig_is_in_zoi"),
            ),
            on="orig"
        )
        .join(
            platoons.select("platoon_id", "nr_vehicles"),
            on="platoon_id"
        )
    )
    return trips_df


def create_complete_zone_agg_time_grid(
    df: pl.DataFrame,
    aggregation_step: int
) -> pl.DataFrame:
    unique_zones = df.select("id_zone").unique().sort("id_zone")
    unique_hours = pl.DataFrame({"agg_time_unit": list(range(int(24*3600/aggregation_step)))})
    return unique_zones.join(unique_hours, how="cross")


def compute_agg_granularity_io(
    trips : pl.DataFrame,
    seed: int,
    aggregation_step: int,
    **extra_args
) -> pl.DataFrame:
    transitions_df = (
        trips
        .rename({"edge_id_zone": "to_zone_id"})
        .with_columns(
            from_zone_id = pl.col("to_zone_id").shift().over("platoon_id")
        )
        .drop_nulls("from_zone_id") # remove the first observation of the platoon
        #.filter(pl.col("link_id").str.contains(r"^[0-9_]+$"))
        .filter(
            pl.col("from_zone_id") != pl.col("to_zone_id")
        )
        .with_columns(
            ((pl.col("t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
    )

    grp_cols = ["agg_time_unit", "id_zone", "orig_is_in_zoi"]
    agg_in_flows = (
        transitions_df
        .select("agg_time_unit", "to_zone_id", "orig_is_in_zoi", "nr_vehicles")
        .group_by(
            "agg_time_unit", pl.col("to_zone_id").alias("id_zone"), "orig_is_in_zoi"
        ).agg(pl.col("nr_vehicles").sum().alias("in_flow"))
    )

    agg_in_out_flows = (
        agg_in_flows
        .sort(grp_cols)
        .pivot(
            on=["orig_is_in_zoi"],
            index=["agg_time_unit", "id_zone"],
            values=["in_flow"],
            separator="_w_orig_in_ZOI_",
        )
        .fill_null(0)
        .rename({"true": "inflow_from_inside", "false": "inflow_from_outside"})
    )
    # Create complete grid and ensure all zone-hour combinations exist
    if len(agg_in_out_flows) > 0:
        complete_grid = create_complete_zone_agg_time_grid(agg_in_out_flows, aggregation_step)

        # Get all column names except id_zone and hour for filling with zeros
        agg_in_out_flows = (
            complete_grid
            .join(agg_in_out_flows, on=["id_zone", "agg_time_unit"], how="left")
            .fill_null(0)
        )
    else:
        # If no data, create empty result with seed column
        agg_in_out_flows = pl.DataFrame({"id_zone": [], "agg_time_unit": [0], "seed": [seed], "inflow_from_inside":[0], "inflow_from_outside": [0]})

    result = (
        agg_in_out_flows
        .with_columns(
            (pl.col("agg_time_unit")*aggregation_step ).alias("start_t"),
            ((pl.col("agg_time_unit")+1)*aggregation_step).alias("end_t")
        )
        .with_columns(seed=pl.lit(seed))
        .sort(by=["id_zone","agg_time_unit"])
    )
    logging.debug(result.head(10))
    return result


def compute_agg_granularity_starting(
    trips: pl.DataFrame,
    seed: int,
    aggregation_step: int,
    **extra_args
) -> pl.DataFrame:
    agg_starting = (
        trips
        .sort(by='t', descending=False, nulls_last=True)
        .with_columns(
            pl.col("orig_id_zone").alias("id_zone"),
            ((pl.col("t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
        .group_by("platoon_id")
        .agg(pl.col("orig", "agg_time_unit", "nr_vehicles", "id_zone").first())
        .group_by("id_zone", "agg_time_unit")
        .agg(pl.col("nr_vehicles").sum().alias("starting_from_zone"))
        .with_columns(seed=pl.lit(seed))
    )
    # Create complete grid and ensure all zone-hour combinations exist
    if len(agg_starting) > 0:
        complete_grid = create_complete_zone_agg_time_grid(agg_starting, aggregation_step)

        agg_starting = (
            complete_grid
            .join(agg_starting, on=["id_zone", "agg_time_unit"], how="left")
            .fill_null(0)
        )
    else:
        # If no trips data, create empty result with seed column
        agg_starting = pl.DataFrame({"id_zone": [], "agg_time_unit": [0], "starting_from_zone": [0], "seed": [seed]})
    result = (
        agg_starting
        .with_columns(
            (pl.col("agg_time_unit")*aggregation_step ).alias("start_t"),
            ((pl.col("agg_time_unit")+1)*aggregation_step).alias("end_t")
        )
        .with_columns(seed=pl.lit(seed))
        .sort(by=["id_zone", "agg_time_unit"])
    )
    logging.debug(result)
    return result


def compute_min_granularity_traffic(
    trips: pl.DataFrame,
    time_step: int,
    first_hour: int
):
    # Preparing the time bounds
    t_min = first_hour * 3600
    time_bounds = trips.select([
        ((pl.col("t").min() - t_min) // time_step).alias("min_bin"),
        ((pl.col("t").max() - t_min) // time_step).alias("max_bin")
    ])

    if t_min > trips["t"].min():
        raise ValueError("t_min exceeds the minimum time in data")
    if trips["t"].max() < (24+first_hour)*3600:
        logging.debug("Not the whole day is simulated")

    max_bins = max(time_bounds["max_bin"][0], 24*3600/time_step )
    num_bins = int(max_bins + 1)

    # Preparing the link (mapping!)
    zone_mapping = trips.select(pl.col("id_zone").alias("spatial_unit")).unique().with_row_index("idx")
    num_zones = zone_mapping.height

    # Interval identification (entrance, exit)
    processed_trips = (
        trips
        .sort(["platoon_id", "t"])
        .with_columns([
            ((pl.col("t") - t_min) // time_step).cast(pl.Int32).alias("bin_entry"),
            ((pl.col("t").shift(-1).over("platoon_id") - t_min) // time_step).cast(pl.Int32).alias("bin_exit"),
            pl.col("nr_vehicles").cast(pl.Int32)
        ])
        .join(zone_mapping, left_on="id_zone", right_on="spatial_unit", how='inner') # Remove fictitious + add zone_idx
        .filter(pl.col("bin_exit").is_not_null()) # Remove the last record (as empty)
        .select(["idx", "bin_entry", "bin_exit", "nr_vehicles"])
    )

    # Create the matrix of the occupation differences
    # Remark: use int16 o int32 to save memory (if veh/link < 32.000)
    diff_matrix = np.zeros((num_zones, num_bins + 1), dtype=np.int32)

    z_idx = processed_trips["idx"].to_numpy()
    t_en = processed_trips["bin_entry"].to_numpy()
    t_ex = processed_trips["bin_exit"].to_numpy()
    weights = processed_trips["nr_vehicles"].to_numpy()

    np.add.at(diff_matrix, (z_idx, t_en), weights) # Matrix filling with entrances
    np.add.at(diff_matrix, (z_idx, t_ex), -weights) # Matrix filling with exit
    # Remark: use np.add.at to tackle cases with multiple veh/t

    # Create the matrix of the total occupation
    occupancy_matrix = diff_matrix.cumsum(axis=1)[:, :-1]
    occupancy_matrix = fold_matrix_24h(matrix=occupancy_matrix, operation="sum", time_step=time_step)
    return occupancy_matrix, zone_mapping


def compute_agg_granularity_traffic(
    trips: pl.DataFrame,
    seed: int,
    aggregation_step: int,
    time_step: int,
    first_hour: int
):
    trips = trips.rename({"edge_id_zone": "id_zone"})
    occupancy_matrix, zone_mapping = compute_min_granularity_traffic(trips, time_step, first_hour)
    nrow, total_cols = occupancy_matrix.shape
    j = aggregation_step // time_step
    ncol = total_cols * time_step // aggregation_step
    occupancy_matrix = occupancy_matrix.reshape(nrow, ncol, j).max(axis=2)

    traffic_mean = (
		melt_matrix(occupancy_matrix, zone_mapping, time_step=aggregation_step, t_min=first_hour*3600)
		.rename({"counts": "traffic_in_zone", "spatial_unit": "id_zone", "t": "start_t"})
        .with_columns(
            ((pl.col("start_t") // aggregation_step) % (24*3600/aggregation_step)).alias("agg_time_unit").cast(pl.Int32)
        )
		.sort(["id_zone", "agg_time_unit"])
	)

    # Create complete grid and ensure all zone-hour combinations exist
    if len(traffic_mean) > 0:
        complete_grid = create_complete_zone_agg_time_grid(traffic_mean, aggregation_step)

        traffic_mean = (
            complete_grid
            .join(traffic_mean, on=["id_zone", "agg_time_unit"], how="left")
            .fill_null(0)
        )
    else:
        # If no trips data, create empty result with seed column
        traffic_mean = pl.DataFrame({"id_zone": [], "agg_time_unit": [0], "traffic_in_zone": [0], "seed": [seed]})

    result = (
        traffic_mean
        .with_columns(
            (pl.col("agg_time_unit")*aggregation_step ).alias("start_t"),
            ((pl.col("agg_time_unit")+1)*aggregation_step).alias("end_t")
        )
        .with_columns(seed=pl.lit(seed))
        .sort(by=["id_zone", "agg_time_unit"])
    )
    logging.debug(result)
    return result


def average_seed_results(
    df: pl.DataFrame
) -> pl.DataFrame:
    grp_cols = ["id_zone", "agg_time_unit", "start_t", "end_t"]
    return (
        df.group_by(grp_cols)
        .agg(
            pl.all().exclude(grp_cols, "seed").mean().name.suffix('_mean'),
            pl.all().exclude(grp_cols, "seed").std().name.suffix('_std')
        )
        .sort(by=["id_zone", "agg_time_unit"])
    )

def postprocess_zone_flows(zone_df, aree_gdf, in_zones, other_cols):
    zone_df = zone_df.copy()[zone_df["id_zone"] != 0]

    zones_in_to_id = {int(aree_gdf.loc[aree_gdf["index"] == id]["id_zone"].iloc[0]): int(id) for id in in_zones}
    zone_df["index"] = zone_df["id_zone"].map(zones_in_to_id)

    zone_df = zone_df.sort_values(["index", "agg_time_unit"])

    missing_zones = set(in_zones) - set(zone_df["index"].dropna().unique())
    if missing_zones:
        time_units = zone_df[["agg_time_unit", "start_t", "end_t"]].drop_duplicates()
        missing_data_list = []
        for idx in missing_zones:
            id_zone = int(aree_gdf.loc[aree_gdf["index"] == idx, "id_zone"].iloc[0])
            for _, time_row in time_units.iterrows():
                dict_data = {
                    "id_zone": id_zone,
                    "agg_time_unit": int(time_row["agg_time_unit"]),
                    "start_t": int(time_row["start_t"]),
                    "end_t": int(time_row["end_t"])
                }
                for col in other_cols:
                    dict_data[col] = 0.0
                dict_data["index"] = idx
                missing_data_list.append(dict_data)
        missing_data = pd.DataFrame(missing_data_list)
        zone_df = pd.concat([zone_df, missing_data], ignore_index=True)
        zone_df = zone_df.sort_values(["index", "agg_time_unit"])

    return zone_df


def main(ctx: dict = context):
    logging.basicConfig(level=logging_level)
    with timed():
        # Reading input data
        nodes = preprocess_nodes(PathRegistry.SIM_NODES)
        edges = preprocess_edges(PathRegistry.SIM_EDGES)

        # Flow per zone
        logging.info(f"Computing zonal inflow, starting, and traffic from vehicles")
        df_t = pl.DataFrame()
        df_s = pl.DataFrame()
        df_io = pl.DataFrame()
        keys = ["aggregation_step", "time_step", "first_hour"]
        params = {k: ctx[k] for k in keys if k in ctx}
        for s in range(ctx["total_seeds"]):
            logging.debug(f"Computing for seed {s}")
            platoons=pl.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=s))
            trips = preprocess_trips(
                veh=pl.read_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=s)),
                nodes=nodes,
                edges=edges,
                platoons=platoons
            )
            df_t = pl.concat([df_t, compute_agg_granularity_traffic(trips, seed=s, **params)])
            df_s = pl.concat([df_s, compute_agg_granularity_starting(trips, seed=s, **params)])
            df_io = pl.concat([df_io, compute_agg_granularity_io(trips, seed=s, **params)])

    with timed():
        # Reading the input data
        gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)
        aree_gdf_inside_raw, aree_gdf_outside_raw = data_reader.AOI_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od", df_around=gdf_zoi, return_type="both"
        )
        aree_gdf_inside_raw["id_zone"] = aree_gdf_inside_raw["id_zone"].astype(int)
        aree_gdf_inside_raw["type_zoi"] = "inside"
        aree_gdf_outside_raw["id_zone"] = aree_gdf_outside_raw["id_zone"].astype(int)
        aree_gdf_outside_raw["type_zoi"] = "outside"
        aree_gdf_inside_raw = aree_gdf_inside_raw.to_crs(constants.CRS_LATLONG)
        aree_gdf_outside_raw = aree_gdf_outside_raw.to_crs(constants.CRS_LATLONG)
        aree_gdf = pd.concat([aree_gdf_inside_raw, aree_gdf_outside_raw], axis=0)
        aree_gdf = aree_gdf.reset_index(drop=True).reset_index().sort_values(by="index")
        in_zones = np.array(aree_gdf.loc[aree_gdf["type_zoi"] == "inside", "index"].unique())

        # Final postprocessing and saving
        logging.info(f"Processing and saving zonal inflow, starting, and traffic from vehicles")
        df_io_final = average_seed_results(df_io).to_pandas()
        df_io_final.to_parquet(PathRegistry.INFLOW_PER_ZONE_SIM)
        df_io_final = postprocess_zone_flows(zone_df=df_io_final, 
            aree_gdf=aree_gdf, in_zones=in_zones,
            other_cols=["inflow_from_outside_mean", "inflow_from_inside_mean", "inflow_from_outside_std", "inflow_from_inside_std"])
        df_io_final.to_parquet(PathRegistry.INFLOW_PER_ZONE_SIM_PROCESSED)

        df_s_final = average_seed_results(df_s).to_pandas()
        df_s_final.to_parquet(PathRegistry.STARTING_PER_ZONE_SIM)
        df_s_final = postprocess_zone_flows(zone_df=df_s_final,
            aree_gdf=aree_gdf, in_zones=in_zones,
            other_cols=['starting_from_zone_mean', 'starting_from_zone_std'])
        df_s_final.to_parquet(PathRegistry.STARTING_PER_ZONE_SIM_PROCESSED)

        df_t_final = average_seed_results(df_t).to_pandas()
        df_t_final.to_parquet(PathRegistry.TRAFFIC_PER_ZONE_SIM)
        df_t_final = postprocess_zone_flows(
            zone_df=df_t_final, aree_gdf=aree_gdf, in_zones=in_zones,
            other_cols=["traffic_in_zone_mean", "traffic_in_zone_std"],
        )
        df_t_final.to_parquet(PathRegistry.TRAFFIC_PER_ZONE_SIM_PROCESSED)
        

if __name__ == "__main__":
    main(ctx=context)
