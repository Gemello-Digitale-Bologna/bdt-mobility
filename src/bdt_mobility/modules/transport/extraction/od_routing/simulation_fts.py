# Passa edge, node, platoon
# sistema platoon in base alla first hour
import numpy as np
import pandas as pd
from pathlib import Path

from fts import Simulator

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed

def _read_input(
    nodes_file: Path,
    edges_file: Path,
    platoons_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load nodes, edges, and vehicles from UXSim-format files."""
    nodes = (
        pd.read_csv(nodes_file, index_col='id')
        .rename(columns={'is_in_zoi': 'in-out', 'id_zone': 'zone'})
        [["in-out", "zone"]]
    )
    edges = (
        pd.read_csv(edges_file, index_col='id')
        .rename(columns={'u': 'from', 'v': 'to', 'maxspeed_mps': 'speed', 'id_zone': 'zone'})
        [["from", "to", "length", "speed", "lanes", "zone"]]
    )
    platoons = pd.read_parquet(platoons_file)[
        ['platoon_id', 'second', 'from_node_id', 'to_node_id', 'nr_vehicles']
    ].set_index('platoon_id')

    repeats = platoons['nr_vehicles'].to_numpy()
    vehicles = pd.DataFrame({
        'origin':      np.repeat(platoons['from_node_id'].to_numpy(), repeats),
        'destination': np.repeat(platoons['to_node_id'].to_numpy(), repeats),
        'start':       np.repeat(platoons['second'].to_numpy(), repeats),
    })
    vehicles.index.name = 'id'
    return nodes, edges, vehicles


def _map_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping from legacy node IDs to contiguous zero-based indices."""
    nodes_map = nodes.reset_index()[['id']]
    nodes_map.index.name = 'seq'
    return nodes_map.reset_index().set_index('id')


def _convert_edges(edges: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in edges to sequential indices."""
    converted = edges.copy()
    converted['from'] = converted['from'].map(nodes_map['seq'])
    converted['to'] = converted['to'].map(nodes_map['seq'])
    return converted[['from', 'to', 'length', 'speed', 'lanes']]


def _convert_vehicles(vehicles: pd.DataFrame, nodes_map: pd.DataFrame) -> pd.DataFrame:
    """Remap legacy node IDs in vehicles to sequential indices."""
    converted = vehicles.copy()
    converted['origin'] = converted['origin'].map(nodes_map['seq'])
    converted['destination'] = converted['destination'].map(nodes_map['seq'])
    return pd.DataFrame(converted)


def _preprocess_vehicles(
    df: pd.DataFrame,
    first_hour: int
):
    departure_offset = first_hour * 60 * 60
    df["start"] = (df["start"] - departure_offset) % 86400
    return df


def simulation(
    i_seed: int,
    first_hour: int,
    SP: int,
    random_seed: int,
    refresh_workers: int | None = None,
):
    logging.debug(f"Reading input files")
    nodes, edges, vehicles = _read_input(
        PathRegistry.SIM_NODES,
        PathRegistry.SIM_EDGES,
        PathRegistry.PLATOONS_SEED.substitute(seed=i_seed)
    )

    departure_offset = first_hour * 60 * 60

    logging.debug(f"Resetting the departure time starting to 0 from {first_hour}")
    vehicles = _preprocess_vehicles(vehicles, first_hour)

    logging.debug(f"Checking for duplicate edges")
    n_duplicated = edges.duplicated(subset=['from', 'to']).sum()
    if n_duplicated > 0:
        logging.info(f"Removing {n_duplicated} multiple edges")
        duplicated = edges[edges.duplicated(subset=['from', 'to'], keep=False)].groupby(['from', 'to'])
        for d, _ in duplicated:
            logging.debug("  * Edge cluster: [" + ", ".join([str(e) for e in d]) + "]")
        edges.drop_duplicates(subset=['from', 'to'], inplace=True)

    logging.debug(f"Converting data...")
    nodes_map = _map_nodes(nodes)
    converted_edges = _convert_edges(edges, nodes_map)
    converted_vehicles = _convert_vehicles(vehicles, nodes_map)

    logging.debug(f"Initializing the simulation")
    simulator, fixed = Simulator.build(
        edges=converted_edges, vehicles=converted_vehicles,
        fix_unreachable=True, random=True, seed=random_seed,
        refresh_workers=refresh_workers,
    )
    if fixed:
        logging.info(f"*** Rerouting {len(fixed)} vehicles with unreachable destinations ***")
        for v in fixed:
            logging.debug("  * Vehicle:" +
                  f"{vehicles.index[v]}: {vehicles.iloc[v]['origin']} -> {vehicles.iloc[v]['destination']}")

    logging.debug(f"Executing the simulation")
    trip_logs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    h = 0
    while True:
        for s in range(60 * 60):
            simulator.step(s % SP == SP - 1)
            abs_step = h * 3600 + s

            waiting = (simulator.vehicles.started_now & ~simulator.vehicles.entered_edge_now).nonzero()[0]
            entered = simulator.vehicles.entered_edge_now.nonzero()[0]
            arrived = simulator.vehicles.arrived_now.nonzero()[0]
            if len(waiting) + len(entered) + len(arrived) > 0:
                v_idx = np.concatenate([waiting, entered, arrived])
                kind = np.repeat(np.array([0, 1, 2], dtype=np.int8),
                                 [len(waiting), len(entered), len(arrived)])
                edge_idx = np.full(len(v_idx), -1, dtype=np.int32)
                edge_idx[len(waiting):len(waiting) + len(entered)] = simulator.vehicles.edge[entered]
                trip_logs.append((abs_step, v_idx, kind, edge_idx))

        h += 1
        nr_waiting = (simulator.vehicles.status == Simulator.VehicleStatus.WAITING.value).sum()
        nr_at_node = (simulator.vehicles.status == Simulator.VehicleStatus.AT_NODE.value).sum()
        nr_in_edge = (simulator.vehicles.status == Simulator.VehicleStatus.IN_EDGE.value).sum()
        nr_arrived = (simulator.vehicles.status == Simulator.VehicleStatus.ARRIVED.value).sum()
        logging.debug(f"Simulation time after {h} hours")
        logging.debug(f" * waiting: {nr_waiting} vehicles")
        logging.debug(f" * at node: {nr_at_node} vehicles")
        logging.debug(f" * in edge: {nr_in_edge} vehicles")
        logging.debug(f" * arrived: {nr_arrived} vehicles")

        if nr_waiting + nr_at_node + nr_in_edge == 0:
            break

    v_idx_all = np.concatenate([b[1] for b in trip_logs])
    t_all = np.concatenate([np.full(len(b[1]), b[0], dtype=np.int64) for b in trip_logs])
    kind_all = np.concatenate([b[2] for b in trip_logs])
    edge_idx_all = np.concatenate([b[3] for b in trip_logs])

    link_labels = edges.index.astype(str).to_numpy()
    link = np.empty(len(v_idx_all), dtype=object)
    link[kind_all == 0] = 'waiting_at_origin_node'
    link[kind_all == 1] = link_labels[edge_idx_all[kind_all == 1]]
    link[kind_all == 2] = 'trip_end'

    trips = pd.DataFrame({
        'name': vehicles.index.to_numpy()[v_idx_all],
        'dn':   np.ones(len(v_idx_all), dtype=np.int64),
        'orig': vehicles['origin'].to_numpy()[v_idx_all],
        'dest': vehicles['destination'].to_numpy()[v_idx_all],
        't':    (t_all + departure_offset).astype(np.int64),
        'link': pd.array(link, dtype='string'),
    })

    logging.debug(f"Done for seed {i_seed}")
    return trips


def main(ctx: dict = context):
    if ctx["simulator"] != "fts":
        raise ValueError("The simulator is set different from fts. Change it in the setting file.")

    logging.basicConfig(level=logging_level)
    logging.info(f"Starting simulation with FTS for {ctx['total_seeds']} seeds")
    with timed():
        names = []
        list_seeds = [i for i in range(ctx['total_seeds'])]
        keys = ["first_hour", "SP", "random_seed", "refresh_workers"]
        params = {k: ctx[k] for k in keys if k in ctx}
        for i_seed in list_seeds:
            logging.info(f"Simulating for seed {i_seed}")
            r = simulation(i_seed=i_seed, **params)

            logging.info(f"Saving for seed {i_seed}")
            output_veh_name = PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)
            r.to_parquet(output_veh_name, index=False)
            names.append((f"simulation_vehicles_seed_{i_seed}", output_veh_name))

    return names


if __name__ == "__main__":
    main(ctx=context)
