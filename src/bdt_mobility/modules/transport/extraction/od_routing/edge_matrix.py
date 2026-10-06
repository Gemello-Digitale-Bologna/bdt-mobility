import polars as pl
import numpy as np

from bdt_mobility.modules.transport.extraction.od_routing.utils import fold_matrix_24h


def preprocess_trips(
    trips: pl.DataFrame,
    platoons: pl.DataFrame,
    edges: pl.DataFrame,
) -> pl.DataFrame:
    return (
        trips
        .drop("x", "s", "v", "attribute", "dn", strict=False)
        .rename({"name": "platoon_id", "link": "link_id"})
        .cast({"orig": int, "dest": int, "platoon_id": int})
        .join(platoons.select(['platoon_id', 'nr_vehicles']), on='platoon_id')
        .join(edges.select("link_id"), on="link_id")
    )


def traffic_per_edge_min_granularity(
    trips: pl.DataFrame,
    edges: pl.DataFrame,
    time_step: int = 1,
    t_min: int = 0,
):
    time_bounds = trips.select([
        ((pl.col("t").min() - t_min) // time_step).alias("min_bin"),
        ((pl.col("t").max() - t_min) // time_step).alias("max_bin"),
    ])
    if t_min > trips["t"].min():
        raise ValueError("t_min exceeds the minimum time in data")

    num_bins = int(time_bounds["max_bin"][0] + 1)
    link_mapping = edges.select("link_id").rename({"link_id": "spatial_unit"}).unique().with_row_index("idx")
    num_links = link_mapping.height

    processed_trips = (
        trips
        .sort(["platoon_id", "t"])
        .with_columns([
            ((pl.col("t") - t_min) // time_step).cast(pl.Int32).alias("bin_entry"),
            ((pl.col("t").shift(-1).over("platoon_id") - t_min) // time_step).cast(pl.Int32).alias("bin_exit"),
            pl.col("nr_vehicles").cast(pl.Int32),
        ])
        .filter(pl.col("bin_exit").is_not_null())
        .join(link_mapping, left_on="link_id", right_on="spatial_unit", how="inner")
        .select(["idx", "bin_entry", "bin_exit", "nr_vehicles"])
    )

    diff_matrix = np.zeros((num_links, num_bins + 1), dtype=np.int32)
    l_idx = processed_trips["idx"].to_numpy()
    t_en = processed_trips["bin_entry"].to_numpy()
    t_ex = processed_trips["bin_exit"].to_numpy()
    weights = processed_trips["nr_vehicles"].to_numpy()
    np.add.at(diff_matrix, (l_idx, t_en), weights)
    np.add.at(diff_matrix, (l_idx, t_ex), -weights)
    # np.add.at handles multiple vehicles at the same (link, t)

    occupancy_matrix = diff_matrix.cumsum(axis=1)[:, :-1]
    occupancy_matrix = fold_matrix_24h(matrix=occupancy_matrix, operation="sum", time_step=time_step)
    return occupancy_matrix, link_mapping


def inflow_per_edge_min_granularity(
    trips: pl.DataFrame,
    edges: pl.DataFrame,
    time_step: int = 1,
    t_min: int = 0,
):
    time_bounds = trips.select([
        ((pl.col("t").min() - t_min) // time_step).alias("min_bin"),
        ((pl.col("t").max() - t_min) // time_step).alias("max_bin"),
    ])
    if t_min > trips["t"].min():
        raise ValueError("t_min exceeds the minimum time in data")

    num_bins = int(time_bounds["max_bin"][0] + 1)
    link_mapping = edges.select("link_id").rename({"link_id": "spatial_unit"}).unique().with_row_index("idx")
    num_links = link_mapping.height

    processed_trips = (
        trips
        .sort(["platoon_id", "t"])
        .with_columns([
            ((pl.col("t") - t_min) // time_step).cast(pl.Int32).alias("bin_entry"),
            ((pl.col("t").shift(-1).over("platoon_id") - t_min) // time_step).cast(pl.Int32).alias("bin_exit"),
            pl.col("nr_vehicles").cast(pl.Int32),
        ])
        .filter(pl.col("bin_exit").is_not_null())
        .join(link_mapping, left_on="link_id", right_on="spatial_unit", how="inner")
        .select(["idx", "bin_entry", "bin_exit", "nr_vehicles"])
    )

    inflow_matrix = np.zeros((num_links, num_bins + 1), dtype=np.int32)
    l_idx = processed_trips["idx"].to_numpy()
    t_en = processed_trips["bin_entry"].to_numpy()
    weights = processed_trips["nr_vehicles"].to_numpy()
    np.add.at(inflow_matrix, (l_idx, t_en), weights)
    # np.add.at handles multiple vehicles at the same (link, t)

    inflow_matrix = fold_matrix_24h(matrix=inflow_matrix, operation="sum", time_step=time_step)
    return inflow_matrix, link_mapping


def travel_time_per_edge_min_granularity(
    trips: pl.DataFrame,
    edges: pl.DataFrame,
    time_step: int = 1,
    t_min: int = 0,
):
    time_bounds = trips.select([
        ((pl.col("t").min() - t_min) // time_step).alias("min_bin"),
        ((pl.col("t").max() - t_min) // time_step).alias("max_bin"),
    ])
    if t_min > trips["t"].min():
        raise ValueError("t_min exceeds the minimum time in data")

    num_bins = int(time_bounds["max_bin"][0] + 1)
    link_mapping = edges.select(pl.col("link_id")).rename({"link_id": "spatial_unit"}).unique().with_row_index("idx")
    num_links = link_mapping.height

    processed_trips = (
        trips
        .sort(["platoon_id", "t"])
        .with_columns([
            pl.col("t").alias("t_entry"),
            pl.col("t").shift(-1).over("platoon_id").alias("t_exit"),
            ((pl.col("t") - t_min) // time_step).cast(pl.Int32).alias("bin_entry"),
            ((pl.col("t").shift(-1).over("platoon_id") - t_min) // time_step).cast(pl.Int32).alias("bin_exit"),
        ])
        .filter(pl.col("bin_exit").is_not_null())
        .with_columns([
            (pl.col("t_exit") - pl.col("t_entry")).alias("travel_time").cast(pl.Int32),
            pl.col("nr_vehicles").cast(pl.Int32),
        ])
        .join(link_mapping, left_on="link_id", right_on="spatial_unit", how="inner")
        .select(["idx", "bin_entry", "bin_exit", "nr_vehicles", "travel_time"])
    )

    diff_sum_tt = np.zeros((num_links, num_bins + 1), dtype=np.int32)
    diff_count = np.zeros((num_links, num_bins + 1), dtype=np.int32)
    l_idx = processed_trips["idx"].to_numpy()
    t_en = processed_trips["bin_entry"].to_numpy()
    t_ex = processed_trips["bin_exit"].to_numpy()
    tt_vals = processed_trips["travel_time"].to_numpy()
    weights = processed_trips["nr_vehicles"].to_numpy()
    np.add.at(diff_sum_tt, (l_idx, t_en), tt_vals)
    np.add.at(diff_sum_tt, (l_idx, t_ex), -tt_vals)
    np.add.at(diff_count, (l_idx, t_en), weights)
    np.add.at(diff_count, (l_idx, t_ex), -weights)
    # np.add.at handles multiple vehicles at the same (link, t)

    sum_tt_matrix = diff_sum_tt.cumsum(axis=1)[:, :-1]
    count_matrix = diff_count.cumsum(axis=1)[:, :-1]
    avg_tt_matrix = np.floor_divide(
        sum_tt_matrix,
        count_matrix,
        out=np.zeros_like(sum_tt_matrix),
        where=count_matrix > 0,
    )
    avg_tt_matrix = fold_matrix_24h(matrix=avg_tt_matrix, operation="mean", time_step=time_step)
    return avg_tt_matrix, link_mapping
