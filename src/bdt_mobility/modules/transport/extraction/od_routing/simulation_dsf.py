import numpy as np
import pandas as pd
from shapely.geometry import Point
import json
from dsf import mobility
from dsf import logging as dsf_logging

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed

YEAR = 2024 # or 2022 if before works in bologna


def postprocess_vehicles(
    first_hour: int
):
    x = pd.read_csv(PathRegistry.TMP_DATA/ "agent_data.csv", sep=";")
    x = x.rename(columns={'agent_id':"name", "edge_id": "link", "time_step_in": "t"})
    x = x.sort_values(["name", "t"])

    # Add the columns with origin and destination nodes
    edge = pd.read_csv(PathRegistry.SIM_EDGES)
    edge = edge[["id", "u", "v"]]
    edge = edge.rename(columns = {"id":"link", "u":"orig", "v":"dest"})

    link_first_last = (
        x
        .groupby("name", as_index=False)["link"]
        .agg(link_orig="first", link_dest="last")
    )
    link_first_last = link_first_last.merge(edge[["link", "orig"]].rename(columns={"link": "link_orig"})).merge(edge[["link", "dest"]].rename(columns={"link": "link_dest"}))
    link_first_last = link_first_last.drop(columns=["link_orig", "link_dest"])

    # Add the row with arrival time
    time_exit = (
        x
        .groupby("name", as_index=False)["time_step_out"]
        .agg(t="last")
    )
    time_exit["link"] = "trip_end"
    x["link"] = x["link"].astype(str)

    x = pd.concat([x[["name", "link", "t"]], time_exit])
    x = x.sort_values(by="t").reset_index(drop=True)

    # Modify the time column, adding the first simulation hour
    x["t"] = x["t"] + (3600*first_hour)
    # Add the needed column
    x["dn"] = 1    
    # Add the origin and destination columns
    x = x.merge(link_first_last)

    return x


def _prepare_input_vehicles(
    scale: float,
    i_seed: int,
    time_step: int,
):
    input_platoon_df = pd.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=i_seed))

    input_platoon = input_platoon_df.groupby("second")["platoon_id"].count()
    input_platoon = input_platoon.reindex(range(int(24*60*60/time_step)), fill_value=0).values # Ensure indices in 0:total_seconds
    input_platoon = input_platoon / scale  # Scale down the number of vehicles
    input_platoon = np.clip(input_platoon, 0, None).astype(np.int64).tolist() # Rm ==0, and convert to int

    return input_platoon
    

def _prepare_input_road():
    edges_df = pd.read_csv(PathRegistry.SIM_EDGES)
    edges_df = edges_df.rename(columns={
        "u": "source", 
        "v": "target", 
        "highway" :"type",
        "lanes": "nlanes",
        "maxspeed_kph": "maxspeed"
    })
    edges_df.to_csv(PathRegistry.TMP_DATA / "edges_dsf.csv", index=False, sep=";")

    nodes_df = pd.read_csv(PathRegistry.SIM_NODES)
    nodes_df["geometry"] = nodes_df.apply(lambda x: Point(x["coord_x"], x["coord_y"]), axis=1)
    nodes_df.to_csv(PathRegistry.TMP_DATA / "nodes_dsf.csv", index=False, sep=";")
    return 0
    

def _prepare_input_config(
    i_seed: int,
    ctx_sim: dict
):
    config_dict = dict()
    config_dict["general"] = {
        "input_folder" : str(PathRegistry.TMP_DATA),
        "output_folder" : str(PathRegistry.TMP_DATA),
        "update_paths": ctx_sim["update_paths"],
        "save_data": {
            "interval": ctx_sim["save_data_interval"],
            "avg": False,
            "road": False,
            "travel": False,
            "agent": True
        },
        "name": f"simulation_{i_seed}"
    }
    config_dict["road_network"] = {
        "edges_file": "edges_dsf.csv",
        "node_properties_file": "nodes_dsf.csv",
        "set_edge_weight": ctx_sim["set_edge_weight"]
    }
    config_dict["dynamics"] = {
        "error_probability": ctx_sim["error_probability"],
        "kill_stagnant_agents": ctx_sim["kill_stagnant_agents"],
        "agent_insertion_method": ctx_sim["agent_insertion_method"],
        "importODsFromCSV": {
            "file": str(PathRegistry.NODE_ORIG_DEST_PROP.substitute(seed=i_seed)),
            "separator": ";"
        },
        "seed": i_seed
    }
    return config_dict


def main(
    ctx: dict
):    
    if ctx["simulator"] != "dsf":
        raise ValueError("The simulator is set different from dsf. Change it in the setting file.")
    
    logging.basicConfig(level=logging_level)
    if ctx["logging_level"] == "debug":
        dsf_logging.set_level(dsf_logging.LogLevel.DEBUG)
    elif ctx["logging_level"] == "trace":
        dsf_logging.set_level(dsf_logging.LogLevel.TRACE)
    elif ctx["logging_level"] is not None:
        logging.warning(f"Logging level '{ctx['logging_level']}' not recognized. Set to None or to one of 'debug' or 'trace'. No logging level will be set for DSF.")

    logging.info(f"Starting simulation with DSF for {ctx['total_seeds']} seeds")
    with timed():
        list_seeds = [i for i in range(ctx['total_seeds'])]
        keys = ["dt_agent", "time_step", "scale", "update_paths", "save_data_interval", "set_edge_weight", 
                "error_probability", "kill_stagnant_agents", "agent_insertion_method"]
        params = {k: ctx[k] for k in keys if k in ctx}

        names = []
        for i_seed in list_seeds:
            
            logging.info(f"Loading input data for seed {i_seed}")
            # Modify the input road files to be compatible with DSF, hence generates alternative csv files
            _prepare_input_road() 
            # Generate the config file for DSF, which includes the path to the input files and the simulation parameters
            config = _prepare_input_config(i_seed=i_seed, ctx_sim=params)
            config_file = str(PathRegistry.TMP_DATA / f"dsf_simconfig_{i_seed}.json")
            with open(config_file, "w") as f:
                json.dump(config, f)
            # Generate the input vehicles for this simulation, by aggregating from the distribution of platoons in the original data
            input_vehicles = _prepare_input_vehicles(i_seed=i_seed, scale=params["scale"], time_step=params["time_step"])

            logging.info(f"Simulating for seed {i_seed}")
            # Prepare the simulator
            simulator = mobility.TrafficSimulator(jsonConfigFile=config_file)
            # Run the simulator
            simulator.run(nAgentsPerTimeStep=input_vehicles, deltaT=int(params["dt_agent"] * params["time_step"]))

            logging.info(f"Saving final simulation data for seed {i_seed}")
            veh_df = postprocess_vehicles(first_hour=ctx["first_hour"])
            output_veh_name = PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)
            veh_df.to_parquet(output_veh_name)
            names.append((f"simulation_vehicles_seed_{i_seed}", output_veh_name))

    return names


if __name__ == "__main__":
    main(ctx=context)