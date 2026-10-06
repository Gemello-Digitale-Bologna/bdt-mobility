import pandas as pd
import numpy as np
import geopandas as gpd


from bdt_mobility.utils import data_reader, data_cleaner
from bdt_mobility.modules.transport.extraction import spatial_assignment

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import DynamicPlatformOutputs

MAIN_OUTPUTS = DynamicPlatformOutputs(PathRegistry.TRAFFIC_SIMULATOR, "table")


def _create_zone_to_nodes_dict(
    nodes: gpd.GeoDataFrame,
    id_node_col: str,
    areas: gpd.GeoDataFrame,
    id_area_col: str,
    random_seed: int
) -> dict[int, list[int]]:

    # Handle a possible issue
    if id_node_col == id_area_col:
        areas = areas.rename(columns={id_area_col: id_area_col + "_area"})
        id_area_col = id_area_col + "_area"

    ### Assign nodes to zones
    ## Understand which zones are internal and which are external to the road network bbox
    bbox = nodes.total_bounds
    id_in_zones, id_out_zones = spatial_assignment.classify_in_out_zones_bbox(bbox=bbox, zones=areas, id_zone_col=id_area_col)

    ## For internal zones, assign all the nodes that are internal to that zone -> have a dict zone: list[nodes]
    ## If no internal nodes, find the closest node to the zone -> add it to the dict zone: list[single node]
    dict_nodes_per_in_zones = spatial_assignment.get_inside_nodes_per_zone(
        gdf_nodes=nodes, id_node_col=id_node_col,
        gdf_zones=areas.loc[areas[id_area_col].isin(id_in_zones)], id_zone_col=id_area_col,
        random_seed=random_seed
    )
    id_in_zones_missing = set(id_in_zones) - set(dict_nodes_per_in_zones.keys())
    if len(id_in_zones_missing) > 0:
        dict_nodes_per_in_zones_missing = spatial_assignment.get_n_closest_nodes_per_zone(
            gdf_nodes=nodes, id_node_col=id_node_col,
            gdf_zones=areas.loc[areas[id_area_col].isin(id_in_zones_missing)], id_zone_col=id_area_col,
            n_closest=1
        )
        dict_nodes_per_in_zones = {**dict_nodes_per_in_zones, **dict_nodes_per_in_zones_missing}

    ## For external zones, assign n nodes that are the closest to that zone -> have a dict zone: list[nodes]
    dict_nodes_per_out_zones = spatial_assignment.get_n_closest_nodes_per_zone(
        gdf_nodes=nodes, id_node_col=id_node_col,
        gdf_zones=areas.loc[areas[id_area_col].isin(id_out_zones)], id_zone_col=id_area_col,
        n_closest=1
    )

    ## Merge the two
    zone_dict = {**dict_nodes_per_in_zones, **dict_nodes_per_out_zones}

    assert areas[id_area_col].isin(zone_dict.keys()).all(), "Some zones do not have assigned nodes in the zone_dict"

    return zone_dict


def _preprocess_od(
    matrice_OD: pd.DataFrame,
    areas_od_zoi: gpd.GeoDataFrame,
    p2v: float,
    reduction_factor: float = 0.0
):
    ## Remove flows that have no geometry
    matrice_OD = data_cleaner.OD_flows(matrice_OD, areas_od_zoi[["id"]])

    ## Add the info of whether the origin or destination is in the ZOI, to be used for the time assignment
    matrice_OD = matrice_OD.merge(
        areas_od_zoi[["id", "mostly_within_zoi"]].rename(
            columns={"id": "from", "mostly_within_zoi": "from_zoi"}
        ),
        on="from",
        how="left",
    ).merge(
        areas_od_zoi[["id", "mostly_within_zoi"]].rename(
            columns={"id": "to", "mostly_within_zoi": "to_zoi"}
        ),
        on="to",
        how="left",
    )

    ## Remove flows that are outside the ZOI (both origin and destination outside)
    matrice_OD = matrice_OD.loc[(matrice_OD["from_zoi"])|(matrice_OD["to_zoi"])]

    ## Remove null flows
    matrice_OD = matrice_OD.loc[matrice_OD["flow"]>0]

    ## New unit: from passengers to cars; remove cars if needed
    matrice_OD["flow"] = matrice_OD["flow"] * p2v * (1 - reduction_factor)
    matrice_OD["flow"] = matrice_OD["flow"].astype(int) # round it for memory efficency

    return matrice_OD


def _get_vehicles(
    od: pd.DataFrame,
    deltan: int,
):
    ### Create vehicles by expanding the od: 1 element per each flow per od
    vehicles = (
        od
        .loc[lambda df: df.index.repeat(df["flow"])]  # repeat rows
        .reset_index(drop=True)                       # clean index
        .reset_index(names="vehicle_id")              # create row index column
        .drop(columns="flow")
    )
    assert od["flow"].sum() == vehicles.shape[0]

    ### Groups vehicles in platoons, adding a platoon_id
    od_cols = ["from", "to"]
    vehicles["internal_platoon_id"] = (
        vehicles.groupby(od_cols).cumcount() // deltan
    )
    agg = (
        vehicles
        .groupby(od_cols + ["internal_platoon_id"], as_index=False)
        .size()
        .rename(columns={"size": "nr_vehicles"}) ## add the platoon size
        .sort_values(od_cols + ["internal_platoon_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    agg["platoon_id"] = agg.index  ## add the platoon id

    # join (merge in pandas)
    vehicles = vehicles.merge(
        agg,
        on=od_cols + ["internal_platoon_id"],
        how="left",
    )

    assert od["flow"].sum() == vehicles.shape[0]

    return vehicles


def _assign_start_hour(
    df: pd.DataFrame,
    weights_inflow: list,
    weights_traffic: list,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Assign start hours to platoons based on different weight distributions.

    Platoons entering the ZOI use inflow weights,
    while others use traffic weights.

    Args:
        df: DataFrame containing platoon data
        weights_inflow: Probability weights for inflow hours (length 24)
        weights_traffic: Probability weights for traffic hours (length 24)
        rng: Random number generator

    Returns:
        DataFrame with assigned start hours
    """
    # Normalize weights to ensure they sum to 1
    weights_inflow_norm = np.asarray(weights_inflow) / np.sum(weights_inflow)
    weights_traffic_norm = np.asarray(weights_traffic) / np.sum(weights_traffic)

    # Generate random hours for both distributions
    n_rows = df.shape[0]
    hours_inflow = rng.choice(24, size=n_rows, p=weights_inflow_norm)
    hours_traffic = rng.choice(24, size=n_rows, p=weights_traffic_norm)

    # Add both hour selections to the dataframe
    df = df.copy()
    df["hour_inflow"] = hours_inflow.astype(np.uint8)
    df["hour_traffic"] = hours_traffic.astype(np.uint8)

    # Select the appropriate hour based on origin/destination
    mask = (~df["from_zoi"]) & (df["to_zoi"])
    df["hour"] = np.where(mask, df["hour_inflow"], df["hour_traffic"])
    df = df.drop(columns=["hour_inflow", "hour_traffic"])

    # Assert no NaN values in the hour column
    assert df["hour"].notna().all(), "Generated 'hour' column contains NaN values"

    return df


def _distribute_platoons_in_time_and_space(
    platoons: pd.DataFrame,
    zone_dict: dict[int, list[int]],
    list_inflow_weights: list[float],
    list_traffic_weights: list[float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    ### First step: assign a starting time to the flows depending on the traffic/inflow list
    platoons = platoons.sort_values(by=["platoon_id"])
    platoons = _assign_start_hour(
        df=platoons,
        weights_inflow=list_inflow_weights, weights_traffic=list_traffic_weights,
        rng=rng
    )
    # From "hour" to "second"
    platoons["second"] = (
        (platoons["hour"].astype(np.uint32) * 3600) + rng.integers(0, 3600, size=platoons.shape[0])
    ).astype("Int32")

    # Check that there are no missing values in the second assignment
    assert platoons["second"].notna().all(), "Missing values in 'second' column after assignment"

    ### Second step: assign a starting and ending node to the flows
    platoons = platoons.sort_values(by=["platoon_id", "second"])
    platoons["from_node_id"] = platoons["from"].map(lambda x: rng.choice(zone_dict[int(x)]))
    platoons["to_node_id"] = platoons["to"].map(lambda x: rng.choice(zone_dict[int(x)]))

    # Check that there are no missing values in the node assignment
    assert platoons["from_node_id"].notna().all(), "Missing values in 'from_node_id' column after assignment"
    assert platoons["to_node_id"].notna().all(), "Missing values in 'to_node_id' column after assignment"

    return platoons


def _distribute_vehicles_in_car_fuel( #TODO rewrite this function
    df: pd.DataFrame,
    areas_to_com,
    cars_joint,
    rng: np.random.Generator
) -> pd.DataFrame:

    ### Assign car and fuel types to vehicles (randomly, but weighted by the distribution of cars per comune)
    nr_vehicles = df.shape[0]

    df = df.copy()
    # Randomly assign a comune to each zone
    df = df.sort_values(by=["vehicle_id"])
    df["com_id"] = df["from"].apply(lambda x: rng.choice(areas_to_com[x]))

    # Randomly assign a vehicle type to each vehicle
    df["draw"] = rng.random(len(df))
    df = df.merge(
        cars_joint,
        on="com_id",
        how="left"
    )
    df = df[df["draw"] < df["prop_cum"]]
    df = df.groupby("vehicle_id", as_index=False).first()

    assert df.shape[0] == nr_vehicles
    return df


def prepare_platoons_and_vehicles(
    zonal_flows: pd.DataFrame,
    list_inflow_weights: list[float],
    list_traffic_weights: list[float],
    zone_dict: dict[int, list[int]],
    areas_to_com,
    cars_joint,
    seed: int,
    platoon_size: int = 1,
):

    ### Build the vehicles and platoons between od
    ## Build the vehicle dataframe and divide them in platoons
    vehicles = _get_vehicles(od=zonal_flows, deltan=platoon_size)

    ## Build the platoon dataframe
    platoons = vehicles.drop(columns="vehicle_id").drop_duplicates()

    assert platoons["nr_vehicles"].sum() == vehicles.shape[0]

    ## Set the starting point, the ending point, and the starting time per platoon
    rng = np.random.default_rng(seed=seed * 42 + 17)
    platoons = _distribute_platoons_in_time_and_space(
        platoons,
        zone_dict,
        list_inflow_weights,
        list_traffic_weights,
        rng,
    )

    ### Add the distribution between car and fuel types
    rng = np.random.default_rng(seed * 17 + 42)
    vehicles = _distribute_vehicles_in_car_fuel(
        df=vehicles,
        areas_to_com=areas_to_com,
        cars_joint=cars_joint,
        rng=rng
    )

    # Fix dtypes
    vehicles = vehicles[
        ["vehicle_id", "platoon_id", "from", "to",
         "com_id", "euro_level", "fuel", "segment"]
    ].astype({
        "vehicle_id": "int64",
        "platoon_id": "int64",
        "from": "int64",
        "to": "int64",
        "com_id": "int32",
        "euro_level": "int32",
        "fuel": "string",
        "segment": "string",
    })

    platoons = platoons[[
        "platoon_id", "hour", "internal_platoon_id", "from", "to",
        "nr_vehicles", "second", "from_node_id", "to_node_id"
    ]].astype({
        "platoon_id": "int64",
        "hour": "int64",
        "internal_platoon_id": "int64",
        "from": "int64",
        "to": "int64",
        "nr_vehicles": "int64",
        "second": "int64",
        "from_node_id": "int64",
        "to_node_id": "int64",
    })

    return vehicles, platoons

def get_zonal_dicts(
    df_od: pd.DataFrame,
    group_col: str,
    count_col: str = "count"
) -> dict[int, float]:
    '''
    This function takes a dataframe with OD flows and produces a dictionary with the proportion of flows for each zone.
    '''
    df_od = df_od.copy()
    df_od = df_od.groupby(group_col)[count_col].sum().reset_index()
    total = df_od[count_col].sum()
    df_od[count_col] = df_od[count_col] / total

    return dict(zip(df_od[group_col], df_od[count_col]))


def get_node_prop(
    zone_node_dict: dict[int, list[int]],
    zonal_prop: dict[int, float],
) -> dict[int, float]:
    '''
    This function takes a dictionary with the zones as keys and the list of nodes within the zones as values,
    and a dictionary with the zones as keys and the proportion of flows for each zone as values,
    and returns a dictionary with the nodes as keys and the proportion of flows for each node as values,
    by dividing the proportion of flows for each zone by the number of nodes within the zone and assigning it to each node within the zone.
    If a node is within multiple zones, it will receive the sum of the proportions of flows of those zones.
    '''
    node_prop_dict = {}
    for key, val in zonal_prop.items():
        if zone_node_dict.get(key):
            how_many_nodes = len(zone_node_dict[key])
            weight = val / how_many_nodes
            for node in zone_node_dict[key]:
                if node_prop_dict.get(node):
                    node_prop_dict[node] = node_prop_dict[node] + weight
                else:
                    node_prop_dict[node] = weight
    return node_prop_dict


def prepare_node_proportions(
    zonal_flows: pd.DataFrame,
    zone_dict: dict[int, list[int]],
    nodes: gpd.GeoDataFrame
) -> pd.DataFrame:
    nodes_p = nodes.copy()
    nodes_p = nodes_p[["id"]]

    # Let's start with the ORIGIN
    o_zonal_prop = get_zonal_dicts(zonal_flows, group_col="from", count_col="flow")
    o_node_prop = get_node_prop(
        zone_node_dict=zone_dict,
        zonal_prop=o_zonal_prop,
    )
    nodes_p["o_prop"] = nodes_p["id"].apply(lambda x: round(o_node_prop[int(x)],6) if o_node_prop.get(int(x)) else 0.0)

    # Then repeat for destinations
    d_zonal_prop = get_zonal_dicts(zonal_flows, group_col="to", count_col="flow")
    d_node_prop = get_node_prop(
        zone_dict,
        d_zonal_prop
    )
    nodes_p["d_prop"] = nodes_p["id"].apply(lambda x: round(d_node_prop[int(x)], 6) if d_node_prop.get(int(x)) else 0.0)

    return nodes_p


def get_weights_from_hour_flows(
    flow: list[float]
) -> list[float]:
    flow_mean = []
    for h in range(24):
        start = (np.array(range(25)) * 12)[h]
        end = (np.array(range(25)) * 12)[h+1]
        flow_mean.append(flow[start:end].mean())
    flow_mean = flow_mean / sum(flow_mean)
    return flow_mean


def main(ctx: dict = context):
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info(f"Reading relevant input data")
        ## Load geospatial dataframes: od zones and ZOI shape
        gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)
        areas_od = data_reader.OD_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            PathRegistry.OD_SHAPE_CENTROID,
        )
        areas_od_zoi = spatial_assignment.OD_to_ZOI(df_od=areas_od, df_zoi=gdf_zoi)

        ## Load the od matrix and preprocess it
        od = data_reader.OD_flows(
            PathRegistry.OD_FLOWS_EXCEL,
            df_shapes=None,
        )
        p2v = pd.read_parquet(PathRegistry.P2V).iloc[0,0]
        od = _preprocess_od(matrice_OD=od, areas_od_zoi=areas_od_zoi, p2v=p2v)

        ## Load the road network
        if ctx["road_network"] == "dsf":
            nodes = gpd.read_parquet(PathRegistry.DSF_NODES)
        elif ctx["road_network"] == "geo":
            nodes = gpd.read_parquet(PathRegistry.GEO_NODES)
        elif ctx["road_network"] == "geo_all":
            nodes = gpd.read_parquet(PathRegistry.GEO_NODES_ALL)

        # Load the vehicle type data per comune, filter and check
        comune_gdf = data_reader.comune_shapes(PathRegistry.COMUNE_SHAPES)
        cars_joint = pd.read_parquet(PathRegistry.CARS_JOINT_BY_COMUNE).sort_values("com_id")
        cars_joint["prop_cum"] = cars_joint.groupby("com_id")["prop_all"].cumsum() ## useful for later
        bo_comune = comune_gdf.loc[cars_joint["com_id"].unique()] ## useful for later

        logging.info(f"Preparing zone-to-comuni dict for vehicle type assignment")
        areas_to_com = spatial_assignment.OD_to_comune( ## TODO Change the function and make it returning a dict(area, list(comuni))
            areas_od=areas_od,
            comune=bo_comune
        )

        logging.info(f"Preparing hourly weights for the demand")
        inflow = data_reader.flow_array(PathRegistry.INFLOW_ESTIMATED, column="Weekday")
        w_inflow = get_weights_from_hour_flows(inflow)

        # reference traffic
        traffic = data_reader.flow_array(PathRegistry.TRAFFIC_ESTIMATED, column="Weekday")
        w_traffic = get_weights_from_hour_flows(traffic)

        names = []
        logging.info(f"Preparing platoons for all seeds")
        list_seeds = [i for i in range(0, ctx['total_seeds'])]
        for i_seed in list_seeds:
            logging.info(f"Preparing zone-to-node dict for flow assignment for seed {i_seed}")
            zone_dict = _create_zone_to_nodes_dict(
                nodes=nodes, id_node_col="id",
                areas=areas_od, id_area_col="id",
                random_seed=int(i_seed * 4 + 2)
            )

            logging.info(f"Preparing platoons for seed {i_seed}")
            vhs, pls = prepare_platoons_and_vehicles(
                zonal_flows=od,
                zone_dict=zone_dict,
                areas_to_com=areas_to_com,
                cars_joint=cars_joint,
                list_inflow_weights=w_inflow,
                list_traffic_weights=w_traffic,
                platoon_size=ctx["platoon_size"],
                seed=i_seed,
            )
            pls.to_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=i_seed))
            vhs.to_parquet(PathRegistry.VEHICLES_SEED.substitute(seed=i_seed))
            names.append((f"platoons_seed_{i_seed}", PathRegistry.PLATOONS_SEED.substitute(seed=i_seed)))
            names.append((f"vehicles_seed_{i_seed}", PathRegistry.VEHICLES_SEED.substitute(seed=i_seed)))

            logging.info(f"Preparing node proportions for seed {i_seed}")
            node_prop = prepare_node_proportions(
                zonal_flows=od,
                zone_dict=zone_dict,
                nodes=nodes
            )
            node_prop = (
                node_prop
                .rename(columns={'id': 'node_id', "o_prop": "O", "d_prop": "D"})
                .melt(
                    id_vars=['node_id'],
                    value_vars=['O', 'D'],
                    var_name='type',
                    value_name='weight'
                )
            )
            node_prop.to_csv(PathRegistry.NODE_ORIG_DEST_PROP.substitute(seed=i_seed), index=False, sep=";")
            names.append((f"node_orig_dest_prop_df_{i_seed}", PathRegistry.NODE_ORIG_DEST_PROP.substitute(seed=i_seed)))

    return names


if __name__ == "__main__":
    main(ctx=context)
