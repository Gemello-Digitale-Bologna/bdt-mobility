import uxsim
import pandas as pd 
import gc
import polars as pl

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed


def _create_static_scenario(
    nodes_file,
    edges_file,
    seed: int,
    platoon_size: int,
    total_simulation_time: int,
    jam_density: float|None = None,
    duo_time: int|None = 600,
    reaction_time: int|None = 1,
):
    logging.debug("Generating World object")
    W = uxsim.World(
        deltan=platoon_size,
        duo_update_time=duo_time,
        tmax=total_simulation_time,
        print_mode=1 if logging.getLogger().getEffectiveLevel() == logging.DEBUG else 0,
        save_mode=1,
        show_mode=0,
        random_seed=seed,
        show_progress_deltat=60*60,
        reaction_time=reaction_time,
    )

    logging.debug("Generating nodes")
    with open(nodes_file) as f:
        reader = uxsim.csv.reader(f)
        next(reader)  # Skip header - raise exception if it does not exist
        for r in reader:
            W.addNode(name=str(r[0]), x=float(r[4]), y=float(r[5]))

    logging.debug('Generating links')
    with open(edges_file) as f:
        reader = uxsim.csv.reader(f)
        next(reader)  # Skip header - raise exception if it does not exist
        for r in reader:
            W.addLink(str(r[0]), str(r[1]), str(r[2]), length=float(r[4]), free_flow_speed=float(r[6]), number_of_lanes=int(float(r[5])), jam_density_per_lane=jam_density)

    return W


def _add_daily_demand_from_df(
    W: uxsim.World,
    platoons: pl.DataFrame,
    min_platoon_size: int | None = None,
    departure_offset: int = 10800,  # start simulation at 3am by default.
    total_simulation_time: int = 86400,  # 24 hours by default.
):
    if min_platoon_size is None:
        min_platoon_size = W.DELTAN

    sim_platoons = (
        platoons.filter(pl.col("nr_vehicles") >= min_platoon_size)
        .with_columns(
            pl.col("from_node_id", "to_node_id", "platoon_id").cast(pl.Utf8),
            (pl.col("second") - departure_offset) % 86400,
        )
        .filter(pl.col("second") <= total_simulation_time)
    )

    for p in sim_platoons.iter_rows(named=True):
        W.addVehicle(
            orig=p["from_node_id"],
            dest=p["to_node_id"],
            departure_time=p["second"],
            name=p["platoon_id"],
            direct_call=False,
            auto_rename=False,
        )
    return W


def _cast_to_pandas(
    W: uxsim.World
):
    out = [["name", "dn", "orig", "dest", "t", "link", "x", "s", "v", "attribute"]]
    for veh in W.VEHICLES.values():
        linkname_old = "ImpossibleName"
        for i in range(len(veh.log_t)):
            if veh.log_state[i] in ("wait", "run", "end", "abort"):
                if veh.log_link[i] != -1:
                    linkname = veh.log_link[i].name
                else:
                    if veh.log_state[i] == "wait":
                        linkname = "waiting_at_origin_node"
                    elif veh.log_state[i] == "abort":
                        linkname = "trip_aborted"
                    else:
                        linkname = "trip_end"
                veh_dest_name = None
                if veh.dest != None:
                    veh_dest_name = veh.dest.name
                if linkname != linkname_old or i == len(veh.log_t)-1:
                    out.append([str(veh.name), W.DELTAN, veh.orig.name, veh_dest_name,
                                veh.log_t[i], linkname, veh.log_x[i], veh.log_s[i],
                                veh.log_v[i], veh.attribute])
                linkname_old = linkname
    return pd.DataFrame(out[1:], columns=out[0])

def simulation(
    i_seed: int,
    total_hours: int,
    first_hour: int,
    jam: float,
    platoon_size: int,
    duo_time: int,
    reaction_time: int,
):
    logging.info(f"========================== SEED {i_seed} ============================")

    # Set dependent parameters
    list_hours = [i for i in range(first_hour, total_hours)] + [i for i in range(first_hour)]
    total_simulation_time = 60 * 60 * len(list_hours)
    min_platoon_size = platoon_size / 2

    logging.info(f"Reading the input files")
    # Read input files
    nodes_file = PathRegistry.SIM_NODES
    edges_file = PathRegistry.SIM_EDGES
    platoons = pl.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=i_seed))

    logging.info("Creating scenario")
    W = _create_static_scenario(
        nodes_file=nodes_file,
        edges_file=edges_file,
        seed=i_seed,
        platoon_size=platoon_size,
        duo_time=duo_time,
        jam_density=jam,
        total_simulation_time=total_simulation_time,
        reaction_time=reaction_time,
    )

    W = _add_daily_demand_from_df(
        W,
        platoons,
        min_platoon_size,
        departure_offset=first_hour * 3600,
        total_simulation_time=total_simulation_time,
    )

    logging.info("Simulation")
    W.exec_simulation(duration_t2=total_simulation_time)

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        W.analyzer.print_simple_stats()

    logging.info("Casting results")
    df = _cast_to_pandas(W=W)
    del W
    gc.collect()

    return df


def postprocess_results(
    df: pd.DataFrame,
    first_hour: int,
):
    departure_offset = first_hour * 60 * 60
    df["t"] = df["t"] + departure_offset

    column_types = {
        "name": "int64",
	    "dn": "int64",
        "orig": "int64",
        "dest": "int64",
        "t": "int64",
        "link": "string"
    }
    df = df[["name", "dn", "orig", "dest", "t", "link"]]
    df = df.astype(column_types)

    return df


def main(ctx: dict = context):
    if ctx["simulator"] != "uxsim":
        raise ValueError("The simulator is set different from uxsim. Change it in the setting file.")

    logging.basicConfig(level=logging_level)
    logging.info(f"Starting simulation with UXSIM for {ctx['total_seeds']} seeds")
    with timed():
        list_seeds = [i for i in range(ctx['total_seeds'])]
        keys = ["total_hours", "first_hour", "jam", "platoon_size", "duo_time", "reaction_time"]
        params = {k: ctx[k] for k in keys if k in ctx}
        names = []
        for i_seed in list_seeds:
            logging.info(f"Simulating for seed {i_seed}")
            df = simulation(i_seed=i_seed, **params)

            logging.info(f"Postprocessing for seed {i_seed}")
            df = postprocess_results(df=df, first_hour=ctx["first_hour"])

            logging.info(f"Saving for seed {i_seed}")
            output_veh_name = PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)
            df.to_parquet(output_veh_name, index=False)
            names.append((f"simulation_vehicles_seed_{i_seed}", output_veh_name))
            del df
            gc.collect()

    return names


if __name__ == "__main__":
    main(ctx=context)
