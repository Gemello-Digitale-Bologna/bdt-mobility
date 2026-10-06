from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("geo_edges_all", PathRegistry.GEO_EDGES_ALL, "artifact"),
    PlatformOutput("geo_nodes_all", PathRegistry.GEO_NODES_ALL, "artifact"),
    PlatformOutput("geo_edges", PathRegistry.GEO_EDGES, "artifact"),
    PlatformOutput("geo_nodes", PathRegistry.GEO_NODES, "artifact"),
    PlatformOutput("dsf_edges", PathRegistry.DSF_EDGES, "artifact"),
    PlatformOutput("dsf_nodes", PathRegistry.DSF_NODES, "artifact"),
]

import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
from warnings import catch_warnings, simplefilter
import ast
import dsf


def _simplify_network(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
):
    nodes = nodes.set_index("osmid")
    nodes["y"] = nodes.geometry.y
    nodes["x"] = nodes.geometry.x
    edges = edges.set_index(["u", "v", "key"])

    G = ox.graph_from_gdfs(nodes, edges)
    G = ox.simplification.simplify_graph(G, remove_rings=False)
    nodes, edges = ox.graph_to_gdfs(G)

    nodes = nodes.sort_values("osmid", kind="stable").reset_index()
    edges = edges.sort_values(["u", "v", "key"], kind="stable").reset_index()

    return edges, nodes

def _from_multi_to_digraph(
    edges: gpd.GeoDataFrame,
    id_cols: list,
    value_col: str,
) -> gpd.GeoDataFrame:
    return (
        edges
        .sort_values(value_col, ascending=True, kind="stable")
        .groupby(id_cols)
        .head(1)
        .reset_index(drop=True)
    )

def _get_strongly_connected_component(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    id_col1_edges: str,
    id_col2_edges: str,
    id_col_nodes: str,
):
    edges = edges.copy()
    nodes = nodes.copy()

    G = nx.from_pandas_edgelist(edges, source=id_col1_edges, target=id_col2_edges, create_using=nx.DiGraph())
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    edges = edges[edges[id_col1_edges].isin(largest_scc) & edges[id_col2_edges].isin(largest_scc)]
    nodes = nodes[nodes[id_col_nodes].isin(largest_scc)]

    return edges, nodes

def _fix_osmnx_nodes(
    ox_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    ox_nodes = (
        ox_nodes
        .reset_index()
        .reset_index(drop=True)
        .set_geometry("geometry", crs=CRS_LATLONG)
    )
    return ox_nodes[["osmid", "geometry"]]

def _fix_osmnx_edges(
    ox_edges: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    ox_edges = ox_edges.copy()
    ox_edges = (
        ox_edges
        .reset_index()
        .reset_index(drop=True)
        .set_geometry("geometry", crs=CRS_LATLONG)
    )

    # Highways/Names: from list of strings to string (concatenating all elements)
    ox_edges["highway_list"] = (
        ox_edges["highway"]
        .astype(str)
        .apply(lambda x: _process_columns_concat(x, expected_type="str"))
    )
    ox_edges["name_list"] = (
        ox_edges["name"]        
        .astype(str)
        .apply(lambda x: _process_columns_concat(x, expected_type="str"))
    )

    # Highways/Names: from list of strings to single string (taking the first one in case of multiple values)
    ox_edges["highway"] = (
        ox_edges["highway"]
        .astype(str)
        .apply(lambda x: _process_columns_choose_one(x, expected_type="str"))
    )
    ox_edges["name"] = (
        ox_edges["name"]        
        .astype(str)
        .apply(lambda x: _process_columns_choose_one(x, expected_type="str"))
    )
    ## Highways_ok: if the highway type is in the list of good highways (motorway, primary, secondary, etc.)
    ox_edges["is_relevant"] = _check_highway_ok(ox_edges["highway"])

    # Maxspeed: from list of strings-int to single string (taking the median one in case of multiple values), and fill missing values
    ox_edges["maxspeed"] = _impute_missing_maxspeed(ox_edges)
    ox_edges["maxspeed_mps"] = ox_edges["maxspeed"] / 3.6 # convert from km/h to m/s

    # Lanes: from list of strings-int to single int (taking the median one in case of multiple values), and fill missing values
    ox_edges["lanes"] = _impute_missing_lane(ox_edges)
    ox_edges["lanes"] = ox_edges.apply(lambda x: x["lanes"] if x["oneway"] else max(1, np.ceil(x["lanes"]/2)), axis=1)

    # Free_flow_time:
    ox_edges["free_flow_time"] = ox_edges["length"] / ox_edges["maxspeed_mps"]  # measured in seconds

    # Id: integer id for each edge
    ox_edges["id"] = ox_edges.reset_index(drop=True).index.astype(int)

    # Rounding length to 2 decimals
    ox_edges["length"] = round(ox_edges["length"], 2)
    ox_edges["maxspeed"] = round(ox_edges["maxspeed"], 2)
    ox_edges["maxspeed_mps"] = round(ox_edges["maxspeed_mps"], 2)
    ox_edges["free_flow_time"] = round(ox_edges["free_flow_time"], 2)

    return ox_edges[[
            "u",
            "v",
            "key",
            "highway",
            "highway_list",
            "length",
            "geometry",
            "maxspeed",
            "maxspeed_mps",
            "lanes",
            "free_flow_time",
            "id",
            "oneway",
            "name",
            "name_list",
            "is_relevant"
    ]]

def _impute_missing_maxspeed(
    df: gpd.GeoDataFrame, 
    na_fill: float = 30.0
) -> pd.Series:
    df["maxspeed"] = (
        df["maxspeed"].astype(str).apply(_process_columns_choose_one, expected_type="int")
    )
    mean_maxspeed_by_highway = df.groupby("highway")["maxspeed"].transform("mean")
    values_imp = df["maxspeed"].fillna(mean_maxspeed_by_highway, inplace=False)
    values_imp = values_imp.fillna(na_fill)

    return values_imp

def _impute_missing_lane(
    df: gpd.GeoDataFrame
) -> pd.Series:
    df["lanes"] = df["lanes"].astype(str).apply(_process_columns_choose_one, expected_type="int")
    missing_lanes_mask = df["lanes"].isna()

    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        grouped = df.groupby(["oneway", "highway"])["lanes"]
        group_medians = grouped.median()
        group_medians = group_medians.reindex(
            pd.MultiIndex.from_product([df["oneway"].unique(), df["highway"].unique()]),
            fill_value=np.nan,
        )
        group_medians = group_medians.to_dict()

    group_medians_series = pd.Series(group_medians)

    lanes_imputed = df["lanes"].values
    lanes_imputed[missing_lanes_mask] = df[missing_lanes_mask].apply(
        lambda row: group_medians_series.get(
            (row["oneway"], row["highway"]), 1.0 if row["oneway"] else 2.0
        ),
        axis=1,
    )
    lanes_imputed[np.isnan(lanes_imputed)] = np.where(
        df["oneway"].values[np.isnan(lanes_imputed)], 1.0, 2.0
    )

    return lanes_imputed

def _check_highway_ok(
    highway: list
) -> list:
    good_highways = {
        "motorway",
        "motorway_link",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "trunk",
        "trunk_link",
        "secondary_link",
        "primary_link",
        "tertiary_link",
    }
    return [any(hw in entry for hw in good_highways) for entry in highway]

def _process_columns_choose_one(value: str, expected_type: str = "int"):
    """
    Process mixed integer values:
    - For string integers (e.g., "60"), returns the integer
    - For string lists (e.g., "['70','90']"), returns the average of the integers
    - For other strings (e.g., "signals"), returns np.nan

    Process mixed string values:
    - For signle string (e.g., "motorway"), returns the string
    - For string lists (e.g., "['motorway','trunk']"), returns the first string

    Args:
        value (str or object) : The value to be processed
        expected_type (str) : The expected type of the value, either 'int' or 'str'. Default to 'int'.

    Returns:
        Processed value or np.nan for non-numeric values
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value)

    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            parsed_list = ast.literal_eval(value_str)
            if expected_type == "int":
                integers = [int(x) for x in parsed_list]
                return sum(integers) / len(integers)
            elif expected_type == "str":
                return parsed_list[0]
        except (ValueError, SyntaxError, TypeError):
            return np.nan

    if expected_type == "int" and value_str.isdigit():
        return int(value_str)
    elif expected_type == "str":
        return value_str

    return np.nan

def _process_columns_concat(value: str, expected_type: str = "int"):
    """
    Alternative processor for mixed values.

    For list-like strings such as "['70','90']" or "['motorway','trunk']",
    returns a comma-separated string of the values.
    For scalar values, returns the string representation.

    Args:
        value (str or object): The value to process.
        expected_type (str): The expected type, either 'int' or 'str'.
            Both types are handled by returning string values.

    Returns:
        str or np.nan: A comma-separated string for lists, or the scalar string.
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value)

    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            parsed_list = ast.literal_eval(value_str)
            if isinstance(parsed_list, list):
                return ",".join(str(x) for x in parsed_list)
        except (ValueError, SyntaxError, TypeError):
            return np.nan

    if expected_type in {"int", "str"}:
        return value_str

    return np.nan

def get_all_geo_road_network(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame, 
    e_std_col: dict,
    n_std_col: dict,

):
    edges = edges.copy()
    nodes = nodes.copy()

    logging.debug("---Simplify the network")
    # Simplify the network (removing internal nodes)
    geo_edges_all, geo_nodes_all = _simplify_network(edges, nodes)

    logging.debug("---Removing duplicated edges")
    # Move from a multidigraph to a digraph network
    geo_edges_all = _from_multi_to_digraph(geo_edges_all, id_cols=["u", "v"], value_col="length")

    logging.debug("---Get largest connected component")
    # Get the largest connected component of the graph, if multiple components are present
    geo_edges_all, geo_nodes_all = _get_strongly_connected_component(
        geo_edges_all, geo_nodes_all, id_col1_edges="u", id_col2_edges="v", id_col_nodes="osmid"
    )

    logging.debug("---Fixing the attributes")
    # Fix the attributes
    geo_nodes_all = _fix_osmnx_nodes(geo_nodes_all)
    geo_edges_all = _fix_osmnx_edges(geo_edges_all)

    logging.debug("---Converting to standard format")
    geo_edges_all, geo_nodes_all = from_geo_to_std(geo_edges_all, geo_nodes_all, e_std_col, n_std_col)

    return geo_edges_all, geo_nodes_all

def get_relevant_geo_road_network(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    e_std_col: dict,
    n_std_col: dict,
):
    edges = edges.copy()
    nodes = nodes.copy()

    logging.debug("---Filtering only relevant highways")
    # Remove non-relevant highways
    geo_edges = edges[_check_highway_ok(edges["highway"])]
    geo_nodes = nodes[nodes["osmid"].isin(geo_edges["u"].unique()) | nodes["osmid"].isin(geo_edges["v"].unique())]

    logging.debug("---Simplify the network")
    # Simplify the network (removing internal nodes)
    geo_edges, geo_nodes = _simplify_network(geo_edges, geo_nodes)

    logging.debug("---Removing duplicated edges")
    # Move from a multidigraph to a digraph network
    geo_edges = _from_multi_to_digraph(geo_edges, id_cols=["u", "v"], value_col="length")

    logging.debug("---Get largest connected component")
    # Get the largest connected component of the graph, if multiple components are present
    geo_edges, geo_nodes = _get_strongly_connected_component(
        geo_edges, geo_nodes, id_col1_edges="u", id_col2_edges="v", id_col_nodes="osmid"
    )

    logging.debug("---Fixing the attributes")
    # Fix the attributes
    geo_nodes = _fix_osmnx_nodes(geo_nodes)
    geo_edges = _fix_osmnx_edges(geo_edges)

    logging.debug("---Converting to standard format")
    geo_edges, geo_nodes = from_geo_to_std(geo_edges, geo_nodes, e_std_col, n_std_col)

    return geo_edges, geo_nodes

def get_dsf_road_network(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    e_std_col: dict,
    n_std_col: dict,
):
    edges = edges.copy()
    nodes = nodes.copy()

    logging.debug("---Prepare the graph for dsf-suite")
    # Modify names and format of the data to be compatible with dsf-suite
    nodes = nodes.set_index("osmid")
    nodes["y"] = nodes.geometry.y
    nodes["x"] = nodes.geometry.x
    edges = edges.set_index(["u", "v", "key"])
    # Graph creation
    G = ox.graph_from_gdfs(nodes, edges)

    logging.debug("---Use dsf-suite (cartography) to simplify the graph")
    # Modify names and format of the data to be compatible with dsf-suite
    _, geo_edges_dsf, geo_nodes_dsf = dsf.cartography.process_cartography(
        G=G,
        consolidate_intersections=True,
        dead_ends=False,
        infer_speeds=True,
        scc=True,
    )

    logging.debug("---Get largest connected component")
    # Get the largest connected component of the graph, if multiple components are present
    geo_edges_dsf, geo_nodes_dsf = _get_strongly_connected_component(
        geo_edges_dsf, geo_nodes_dsf, id_col1_edges="source", id_col2_edges="target", id_col_nodes="id"
    )

    logging.debug("---Converting to standard format")
    geo_edges_dsf, geo_nodes_dsf = from_dsf_to_std(geo_edges_dsf, geo_nodes_dsf,  e_std_col, n_std_col)

    logging.debug("---Fixing the attributes")
    # Fix the columns in the gdf_edges_dsf to be compatible with the rest of the code
    if "width" in geo_edges_dsf.columns:
        geo_edges_dsf['width'] = geo_edges_dsf['width'].apply(
                lambda x: float(min(x)) if isinstance(x, list) else float(x) if pd.notna(x) else np.nan
            )
    # Reset the indices
    geo_edges_dsf = geo_edges_dsf.reset_index(drop=True)
    geo_nodes_dsf = geo_nodes_dsf.reset_index(drop=True)

    return geo_edges_dsf, geo_nodes_dsf
    
def from_dsf_to_std(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges_std_columns: dict,
    nodes_std_columns: dict,
):
    nodes = nodes.copy()
    edges = edges.copy()

    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=CRS_LATLONG)
    nodes["coord_long"] = nodes.geometry.x
    nodes["coord_lat"] = nodes.geometry.y
    nodes = nodes.to_crs(CRS_PROJECTED)
    nodes["coord_x"] = nodes.geometry.x
    nodes["coord_y"] = nodes.geometry.y
    nodes = _reorder_columns(nodes, list(nodes_std_columns.keys()))
    nodes = nodes.astype(nodes_std_columns)

    edges = edges.rename(columns={"source": "u", "target": "v", "travel_time": "free_flow_time", "type": "highway", "nlanes": "lanes", "maxspeed": "maxspeed_kph"})
    edges["key"] = 0
    edges["maxspeed_mps"] = round(edges["maxspeed_kph"] / 3.6, 2)
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=CRS_LATLONG).to_crs(CRS_PROJECTED)
    edges["is_relevant"] = _check_highway_ok(edges["highway"])
    edges = _reorder_columns(edges, list(edges_std_columns.keys()))
    edges = edges.astype(edges_std_columns)

    return edges, nodes

def from_geo_to_std(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges_std_columns: dict,
    nodes_std_columns: dict,
):
    nodes = nodes.copy()
    edges = edges.copy()

    nodes = nodes.rename(columns={"osmid": "id"})
    nodes["type"] = "N/A"
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=CRS_LATLONG)
    nodes["coord_long"] = nodes.geometry.x
    nodes["coord_lat"] = nodes.geometry.y
    nodes = nodes.to_crs(CRS_PROJECTED)
    nodes["coord_x"] = nodes.geometry.x
    nodes["coord_y"] = nodes.geometry.y
    nodes = _reorder_columns(nodes, list(nodes_std_columns.keys()))
    nodes = nodes.astype(nodes_std_columns)

    edges = edges.rename(columns={"maxspeed": "maxspeed_kph"})
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=CRS_LATLONG).to_crs(CRS_PROJECTED)
    edges = _reorder_columns(edges, list(edges_std_columns.keys()))
    edges = edges.astype(edges_std_columns)

    return edges, nodes

def _reorder_columns(
    df: gpd.GeoDataFrame,
    columns_order: list
) -> gpd.GeoDataFrame:
    columns_order = columns_order + ["geometry"]

    # Verify whether all the columns in columns_order are present in df.columns
    missing_cols = [col for col in columns_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns expected in the df but not found: {missing_cols}")
    
    # Add the remaining df.columns at the end of the specified order
    extra_cols = [col for col in df.columns if col not in columns_order]
    all_cols = columns_order + extra_cols
    
    return df[all_cols]

def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Loading data")
        # Loading the base network data
        edges, nodes = data_reader.road_data(
            PathRegistry.GEO_EDGES_2024,
            PathRegistry.GEO_NODES_2024,
        )
        # Columns of interest for the edges and nodes, to be used in the from_geo_to_std function
        e_std_col = {
            "id": int,
            "u": int,
            "v": int,
            "key": int,
            "length": float,
            "lanes": int,
            "maxspeed_kph": float,
            "maxspeed_mps": float,
            "free_flow_time": float,
            "highway": str,
            "oneway": bool,
            "is_relevant": bool
        }
        n_std_col = {
            "id": int,
            "type": str,
            "coord_long": float,
            "coord_lat": float,
            "coord_x": float,
            "coord_y": float,
        }

        logging.info("RN with all the roads")
        geo_edges_all, geo_nodes_all = get_all_geo_road_network(edges, nodes, e_std_col, n_std_col)

        logging.debug("Saving!")
        geo_nodes_all.to_parquet(PathRegistry.GEO_NODES_ALL)
        geo_edges_all.to_parquet(PathRegistry.GEO_EDGES_ALL)


        logging.info("RN with only the main roads")
        geo_edges, geo_nodes = get_relevant_geo_road_network(edges, nodes, e_std_col, n_std_col)

        logging.debug("Saving!")
        geo_nodes.to_parquet(PathRegistry.GEO_NODES)
        geo_edges.to_parquet(PathRegistry.GEO_EDGES)


        logging.info("RN simplified with dsf-suite strategies")
        geo_edges_dsf, geo_nodes_dsf = get_dsf_road_network(edges, nodes, e_std_col, n_std_col)

        logging.debug("Saving!")
        geo_nodes_dsf.to_parquet(PathRegistry.DSF_NODES)
        geo_edges_dsf.to_parquet(PathRegistry.DSF_EDGES)


if __name__ == "__main__":
    main()