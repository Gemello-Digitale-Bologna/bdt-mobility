import math
import pandas as pd

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("travel_distance_from_simulator", PathRegistry.TRAVEL_DISTANCE_SIM, "table"),
    PlatformOutput("travel_time_from_simulator", PathRegistry.TRAVEL_TIME_SIM, "table"),
]


def travel_time_per_hour(
    veh: pd.DataFrame
):
    df_tt_temp = veh.groupby("platoon_id")["t"].agg(min_t="min", max_t="max").reset_index()
    df_tt_temp["hour"] = df_tt_temp["min_t"].apply(lambda x: int(x / 3600))
    df_tt_temp.loc[df_tt_temp["hour"] > 23, "hour"] = (
        df_tt_temp.loc[df_tt_temp["hour"] > 23, "hour"] - 24
    )
    df_tt_temp["travel_time"] = (
        df_tt_temp["max_t"] - df_tt_temp["min_t"]
    ) / 60  ## measured in minutes

    return (
        df_tt_temp.rename(columns={"duration": "travel_time"})
        .drop(columns=["min_t", "max_t"])
        .sort_values(by="hour")
    )


def travel_distance_per_hour(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    veh: pd.DataFrame
):
    # Transform the input data
    veh_h = veh.groupby("platoon_id")["t"].agg(min_t="min").reset_index()
    veh_h["hour"] = veh_h["min_t"].apply(lambda x: int(x / 3600))
    veh_h.loc[veh_h["hour"] > 23, "hour"] = (
        veh_h.loc[veh_h["hour"] > 23, "hour"] - 24
    )
    veh = pd.merge(veh, veh_h, how="left")

    # Compute the travel distance
    td_tmp = pd.merge(veh, edges[["link", "length"]], on="link", how="left")
    td_tmp = td_tmp[td_tmp["link"] != "trip_end"]
    td_tmp = (
        td_tmp.groupby(["platoon_id", "hour"])["length"]
        .sum()
        .reset_index()
        .rename(columns={"length": "travel_dist"})
    )

    # Compute the od-distance
    odd_tmp = veh.groupby(["platoon_id", "hour"]).head(1).reset_index()
    odd_tmp = pd.merge(
        odd_tmp,
        nodes[["node", "coord_x", "coord_y"]].rename(
            columns={"node": "orig", "coord_x": "coord_x_1", "coord_y": "coord_y_1"}
        ),
        on="orig",
        how="left",
    )
    odd_tmp = pd.merge(
        odd_tmp,
        nodes[["node", "coord_x", "coord_y"]].rename(
            columns={"node": "dest", "coord_x": "coord_x_2", "coord_y": "coord_y_2"}
        ),
        on="dest",
        how="left",
    )
    odd_tmp["od_dist"] = odd_tmp.apply(
        lambda row: math.sqrt(
            (row["coord_x_1"] - row["coord_x_2"]) ** 2
            + (row["coord_y_1"] - row["coord_y_2"]) ** 2
        ),
        axis=1,
    )
    odd_tmp = odd_tmp[["platoon_id", "od_dist", "hour"]]
    return pd.merge(td_tmp, odd_tmp)


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        # Read the input data
        df_ed = (
            pd.read_csv(PathRegistry.SIM_EDGES)
            [['id', 'u', 'v', 'length', 'maxspeed_mps', 'lanes', 'id_zone']]
            .rename(columns={'id':'link', 'u': 'node1', 'v':'node2', 'maxspeed_mps': 'maxspeed', 'lanes': 'n_lanes', 'id_zone': 'zone'})
            .astype({"link": str})
        )
        df_nd = (
            pd.read_csv(PathRegistry.SIM_NODES)
            [['id', 'coord_x', 'coord_y', "is_in_zoi", "id_zone"]]
            .rename(columns={'id': 'node', "is_in_zoi": "zoi", "id_zone": "zone"})
        )

        logging.info(f"Computing and saving trip metrics from vehicles")
        list_seeds = [i for i in range(context['total_seeds'])]
        df_td_final = pd.DataFrame()
        df_tt_final = pd.DataFrame()
        for i_seed in list_seeds:
            df_veh = pd.read_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)).rename(columns={"name": "platoon_id"})
            logging.debug(f"Computing travel distance for seed {i_seed}")
            df_td = travel_distance_per_hour(edges=df_ed, nodes=df_nd, veh=df_veh)
            df_td["seed"] = i_seed
            df_td_final = pd.concat([df_td_final, df_td])
            logging.debug(f"Computing travel time for seed {i_seed}")
            df_tt = travel_time_per_hour(veh=df_veh)
            df_tt["seed"] = i_seed
            df_tt_final = pd.concat([df_tt_final, df_tt])
        df_td_final.to_parquet(PathRegistry.TRAVEL_DISTANCE_SIM)
        df_tt_final.to_parquet(PathRegistry.TRAVEL_TIME_SIM)


if __name__ == "__main__":
    main()
