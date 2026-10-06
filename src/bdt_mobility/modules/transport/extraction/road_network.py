import ast
from warnings import catch_warnings, simplefilter
import logging
from pathlib import Path
from dsf import cartography
import geopandas as gpd
import neatnet
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import polars as pl
import shapely

from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from shapely.geometry import LineString, Point


def create_road_data(
    df_bzoi: gpd.GeoDataFrame,
    relevant_highway: bool,
    connected_network: bool,
    engine: str = "osmnx",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Extracts and preprocesses road data. Optionally filters the largest connected component.

    Args:
        `df_bzoi` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.
        `connected_network` (`bool`): Whether to filter the largest connected network.
        `engine` (`str`, optional): Engine to use for data extraction, either "osmnx" or "dsf". Defaults to "osmnx".

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """
    # TODO: use overpass_settings to load historical data
    if engine == "osmnx":
        edges, nodes = get_osmxn_data(df_bzoi=df_bzoi, relevant_highway=relevant_highway)
        edges = from_multi_to_digraph(edges)
        if connected_network:
            edges, nodes = get_largest_component(edges, nodes, id_start_edge="u", id_end_edge="v", id_node="node_id")

    elif engine == "dsf":
        edges, nodes = get_dsf_data(df_bzoi=df_bzoi, relevant_highway=relevant_highway)
        if connected_network:
            edges, nodes = get_largest_component(edges, nodes, id_start_edge="source", id_end_edge="target", id_node="id")

    else:
        raise ValueError(f"Unsupported engine: {engine}, expected 'osmnx' or 'dsf'")

    return edges, nodes


def save_road_data(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges_namefile: Path | str,
    nodes_namefile: Path | str,
):
    """
    Saves road network edges and nodes to GeoJSON files in a subfolder of the current one named results.

    Args:
        `edges` (`gpd.GeoDataFrame`): GeoDataFrame containing edges data.
        `nodes` (`gpd.GeoDataFrame`): GeoDataFrame containing nodes data.
        `edges_namefile` (`str`): Name of the file to save edges data.
        `nodes_namefile` (`str`): Name of the file to save nodes data.

    Returns:
        `None`
    """

    edges.to_parquet(edges_namefile)
    nodes.to_parquet(nodes_namefile)
    logging.info(f"Saved edges to {edges_namefile} and nodes to {nodes_namefile}")


def get_osmxn_data(
    df_bzoi: gpd.GeoDataFrame,
    relevant_highway: bool,
) -> tuple:
    """
    Extracts road data using OSMNX and then preprocesses the nodes and edges.

    Args:
        `df_bzoi` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """

    edges, nodes = _extract_osmnx_data(df_bzoi=df_bzoi, relevant_highway=relevant_highway)

    nodes = _fix_osmnx_nodes(nodes)
    edges = _fix_osmnx_edges(edges)

    return edges, nodes


def _extract_osmnx_data(
    df_bzoi: gpd.GeoDataFrame,
    relevant_highway: bool,
) -> tuple:
    """
    Extracts road network data from OSMNX using a custom filter.

    Args:
        `df_bzoi` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """
    if relevant_highway:
        highway_ok = [
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
        ]
        pattern = "|".join(highway_ok)
        custom_filter = f"['highway'~'{pattern}']"
    else:
        custom_filter = None

    G = ox.graph_from_polygon(
        polygon=df_bzoi.to_crs(CRS_LATLONG).geometry.iloc[0],
        network_type="drive",
        simplify=True,
        custom_filter=custom_filter,
    )
    nodes, edges = ox.graph_to_gdfs(G)
    return edges, nodes


def _fix_osmnx_nodes(
    ox_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Fixes and preprocesses OSMNX node data.

    Args:
        `ox_nodes` (`gpd.GeoDataFrame`): GeoDataFrame containing raw OSMNX node data.

    Returns:
        `gpd.GeoDataFrame`: Processed GeoDataFrame containing nodes data.
    """
    ox_nodes = (
        ox_nodes.reset_index()
        .rename(columns={"osmid": "node_id"})
        .set_geometry("geometry", crs=CRS_LATLONG)
        .to_crs(CRS_PROJECTED)
    )
    ox_nodes["coord1"] = ox_nodes.geometry.x
    ox_nodes["coord2"] = ox_nodes.geometry.y
    ox_nodes["node_id"] = ox_nodes["node_id"].astype(str)
    return ox_nodes[["node_id", "coord1", "coord2", "geometry"]]


def _fix_osmnx_edges(ox_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fixes and preprocesses OSMNX edge data.

    Args:
        `ox_edges` (`gpd.GeoDataFrame`): GeoDataFrame containing raw OSMNX edge data.

    Returns:
        `gpd.GeoDataFrame`: Processed GeoDataFrame containing edges data.
    """
    ox_edges = (
        ox_edges.reset_index()
        .to_crs(CRS_PROJECTED)
        .astype({"u": str, "v": str})
        .query("u != v")
    )

    ox_edges["highway"] = (
        ox_edges["highway"]
        .astype(str)
        .apply(lambda x: _process_columns(x, expected_type="str"))
    )
    ox_edges["highway_ok"] = _check_highway_ok(ox_edges["highway"])
    ox_edges["maxspeed_imputed"] = (
        _impute_missing_maxspeed(ox_edges) / 3.6
    )  # moving to m/s
    ox_edges["lanes_imputed"] = _impute_missing_lane(ox_edges)
    ox_edges["free_flow_time"] = (
        ox_edges["length"] / ox_edges["maxspeed_imputed"]
    )  # measured in seconds
    ox_edges["link_id"] = ox_edges["u"].astype(str) + "_" + ox_edges["v"].astype(str) + "_" + ox_edges["key"].astype(str)

    return ox_edges[
        [
            "u",
            "v",
            "key",
            "highway",
            "length",
            "geometry",
            "maxspeed_imputed",
            "lanes_imputed",
            "free_flow_time",
            "link_id",
            "highway_ok",
            "oneway",
        ]
    ]


def _impute_missing_maxspeed(df: gpd.GeoDataFrame, na_fill: float = 30):
    """
    Imputes missing maxspeed values for edges.

    Args:
        `df` (`gpd.GeoDataFrame`): GeoDataFrame containing edge data with potential missing maxspeed values.
        `na_fill` (`float`): remaining missing values are filled with this quantity. Default to `30` (for speed in kmh)
    Returns:
        `pd.Series`: Series of imputed maxspeed values.
    """
    df["maxspeed"] = (
        df["maxspeed"].astype(str).apply(_process_columns, expected_type="int")
    )
    mean_maxspeed_by_highway = df.groupby("highway")["maxspeed"].transform("mean")
    values_imp = df["maxspeed"].fillna(mean_maxspeed_by_highway, inplace=False)
    values_imp = values_imp.fillna(na_fill)

    return values_imp


def _impute_missing_lane(df: gpd.GeoDataFrame):
    """
    Imputes missing lane values for edges.

    Args:
        `df` (`gpd.GeoDataFrame`): GeoDataFrame containing edge data with potential missing lane values.

    Returns:
        `np.array`: Array of imputed lane values.
    """
    df["lanes"] = df["lanes"].astype(str).apply(_process_columns, expected_type="int")
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


def _check_highway_ok(highway: list) -> list:
    """
    Checks if the highway type is within the acceptable list.

    Args:
        `highway` (`list`): List of highway types.

    Returns:
        `list`: Boolean list indicating whether each highway type is acceptable.
    """
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


def _add_edge(
    edges_tbf: gpd.GeoDataFrame,
    nodes_tbf: gpd.GeoDataFrame,
    id1: int | str,
    id2: int | str,
    coords1: float = None,
    coords2: float = None,
    speed_mps: float = 30 / 3.6,
    lanes: int = 2,
):
    """
    Adds a new edge to the road network.

    Args:
        `edges_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing existing edges.
        `nodes_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing existing nodes.
        `id1` (`int`|`str`): ID of the first node.
        `id2` (`int`|`str`): ID of the second node.
        `coords1` (`float`, optional): Coordinates of the first node. Defaults to `None`.
        `coords2` (`float`, optional): Coordinates of the second node. Defaults to `None`.
        `speed_mps` (`float`, optional): Speed in meters per second. Defaults to `30/3.6` (for speed in ms).
        `lanes` (`int`, optional): Number of lanes. Defaults to `2`.

    Returns:
        `pd.Series`: Series representing the new edge.
    """
    try:
        p1 = nodes_tbf[nodes_tbf["node_id"] == id1]["geometry"].values[0]
    except IndexError:
        p1 = Point(coords1)
    try:
        p2 = nodes_tbf[nodes_tbf["node_id"] == id2]["geometry"].values[0]
    except IndexError:
        p2 = Point(coords2)
    line = LineString([p1, p2])
    line_gdf = gpd.GeoDataFrame({"geometry": [line]}, crs=CRS_PROJECTED)
    length = line_gdf["geometry"].length.values[0]

    new_edge = [
        id1,
        id2,
        "0",
        "tertiary",
        length,
        line,
        speed_mps,
        lanes,
        length / speed_mps,
        f"{id1}_{id2}_0",
        True,
        True,
    ]
    new_edge_series = pd.Series(new_edge, index=edges_tbf.columns)

    return new_edge_series


def get_largest_component(
    edges_tbf: gpd.GeoDataFrame, 
    nodes_tbf: gpd.GeoDataFrame,
    id_start_edge: str = "u",
    id_end_edge: str = "v",
    id_node: str = "node_id",
) -> tuple:
    """
    Filters the largest connected component of the road network.

    Args:
        `edges_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing edges.
        `nodes_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing nodes.

    Returns:
        `tuple`: Filtered GeoDataFrames for edges and nodes.
    """
    G1 = create_road_graph(df_edges=edges_tbf, df_nodes=nodes_tbf, 
                           id_start_edge=id_start_edge, id_end_edge=id_end_edge, id_node=id_node)
    strong_components = list(nx.strongly_connected_components(G1))
    largest_component = max(strong_components, key=len)
    large_component_nodes = list(largest_component)

    nodes_tbf = nodes_tbf[nodes_tbf[id_node].isin(large_component_nodes)]
    edges_tbf = edges_tbf[
        edges_tbf[id_start_edge].isin(large_component_nodes)
        & edges_tbf[id_end_edge].isin(large_component_nodes)
    ]

    return edges_tbf, nodes_tbf


def create_road_graph(
    df_edges: pd.DataFrame,
    df_nodes: pd.DataFrame,
    weight_col: str | None = None,
    id_start_edge: str = "u",
    id_end_edge: str = "v",
    id_node: str = "id",
    id_key_edge: str | None = None,
) -> nx.MultiDiGraph:
    """
    Creates a directed graph representation of a road network using NetworkX.

    Args:
        `df_edges` (`pd.DataFrame`): DataFrame containing the edges of the road network.
            Must include columns `u`, `v`, and `key`.
        `df_nodes` (`pd.DataFrame`): DataFrame containing the nodes of the road network.
            Must include a column `node_id`.
        `weight_col` (`str`|`None`, optional): Column in `df_edges` to use as edge weights. Defaults to `None`.
        `id_key_edge` (`str`|`None`, optional): Column in `df_edges` holding each edge's MultiDiGraph
            key. If `None` (default), NetworkX auto-assigns keys, which will not necessarily match
            `df_edges`' own key values whenever a node pair's surviving edge has a non-zero key
            (e.g. after some parallel edges were filtered out upstream) - callers that later look up
            edges by `(u, v, key)` against `df_edges` must pass this to keep the two in sync.

    Returns:
        `nx.MultiDiGraph`: A directed graph of the road network with nodes and edges.
    """
    G = nx.MultiDiGraph()
    for idx, row in df_nodes.iterrows():
        G.add_node(row[id_node])
    for idx, row in df_edges.iterrows():
        edge_key = row[id_key_edge] if id_key_edge is not None else None
        if weight_col is None:
            G.add_edge(row[id_start_edge], row[id_end_edge], key=edge_key)
        else:
            G.add_edge(row[id_start_edge], row[id_end_edge], key=edge_key, weight=row[weight_col])
    return G


def _process_columns(value: str, expected_type: str = "int"):
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


def neatify_postprocess(neat_edges, og_edges):
    """Re-introduces road attributes on the output of `neatnet.neatify()`.

    Parameters
    ----------
    neat_edges : GeoDataFrame
        Output of `neatnet.neatify()`.
    og_edges : GeoDataFrame
        Original edges passed to `neatnet.neatify()`.

    Returns
    -------
    GeoDataFrame
        `neat_edges` augmented with imputed road attributes.
    """
    changed_edges = handle_neatified_edges(neat_edges, og_edges)
    og_edges = handle_unneatified_edges(neat_edges)
    new_edges = pd.concat([og_edges[changed_edges.columns], changed_edges]).astype(
        {"oneway": bool}
    )
    return new_edges


def add_reverse_lanes(edges):
    """Add reverse lanes to set of edges.

    Parameters
    ----------
    edges : GeoDataFrame
        Road edges as output by `neatify_postprocess`: edges for which a reversed
        version should be added are marked in the boolean column "to_reverse".

    Returns
    -------
    GeoDataFrame
        The input augmented with the new reversed lanes, and with the correct number of
        lanes in "lanes_imputed".
    """
    reversed_edges = (
        edges.loc[edges["to_reverse"]].copy().assign(u=edges["v"], v=edges["u"])
    )
    reversed_edges.geometry = reversed_edges.geometry.reverse()
    edges = (
        pd.concat([edges, reversed_edges])
        .rename_axis("undir_edge_id")
        .reset_index()
        .rename_axis("edge_id")
    )
    edges.loc[~edges["oneway"], "lanes_imputed"] = np.round(
        edges.loc[~edges["oneway"], "lanes_imputed"] / 2
    )
    edges["lanes_imputed"] = edges["lanes_imputed"].astype(int)
    edges.loc[edges["lanes_imputed"] < 1, "lanes_imputed"] = 1
    return edges


def snap_to_point(edges, p: Point, distance=50):
    to_snap = edges.sjoin(
        gpd.GeoDataFrame(geometry=[p.buffer(distance)], crs=edges.crs)
    )
    new_geoms = gpd.GeoDataFrame(geometry=to_snap.geometry.snap(p, distance))
    edges.update(new_geoms)
    return edges


def snap_specific_points(new_edges, points, distance=50):
    for p in points:
        new_edges = snap_to_point(new_edges, p, distance)
    # Add nodes at intersections if missing:
    new_edges = neatnet.nodes.fix_topology(new_edges)
    new_edges["edge_id"] = new_edges.index
    return new_edges


def handle_unneatified_edges(neat_edges):
    og_edges = neat_edges.loc[neat_edges["_status"] == "original"].copy()
    og_edges["to_reverse"] = ~og_edges["oneway"].astype(bool)
    og_edges.index.name = "new_id"
    return og_edges


def handle_neatified_edges(
    neat_edges, og_edges, default_speed=50 / 3.6, buffer_dist=25
):
    attrs = ["maxspeed_imputed", "lanes_imputed", "oneway"]

    # Take all edges which were modified by `neatnet` and buffer them by `buff_dist`.
    edges_areas = neat_edges.loc[neat_edges["_status"] != "original"].copy()
    edges_areas["new_length"] = edges_areas.geometry.length
    edges_areas.geometry = edges_areas.geometry.buffer(buffer_dist)
    edges_areas.index.name = "new_id"

    # Match the resulting areas with sets of original edges that overlap them.
    matching_og_edges = og_edges.overlay(
        edges_areas[["new_length", "geometry"]].reset_index()
    )
    matching_og_edges["length"] = matching_og_edges.geometry.length

    # Impute attributes `attrs` to new edges that matched original edges using the
    # buffer + overlay above.
    attrs_from_og_df = (
        pl.DataFrame(matching_og_edges.drop(columns="geometry"))
        .group_by("new_id")
        .agg(
            pl.col("length").sum(),
            pl.col("new_length").first(),
            (pl.col(*attrs) * pl.col("length")).sum(),
        )
        # Each attribute takes the average value of the ones of the overlapping original
        # edges, weighted by the length of the overlap.
        .with_columns(pl.col(*attrs) / pl.col("length"), _status=pl.lit("new"))
        .with_columns(
            pl.col("lanes_imputed").round().cast(pl.UInt8()),
            pl.col("oneway").round().cast(pl.Boolean()),
        )
        # We decide that a reversed version of the edge should be created if either the
        # imputed "oneway" is False, or the length of the neatified edge is less than
        # half of the total length of overlapping original edges. The latter happens for
        # road segments with separate edges for two directions (common for highways or
        # big avenues in OSM), that `neatnet` simplifies into a single edge.
        .with_columns(
            to_reverse=(pl.col("length") >= 2 * pl.col("new_length"))
            | (~pl.col("oneway"))
        )
    )

    # When we could not find original edges, put default values.
    attrs_wo_og_df = pl.DataFrame(
        {
            "new_id": edges_areas.index.difference(attrs_from_og_df["new_id"]),
            "maxspeed_imputed": default_speed,
            "lanes_imputed": 2,
            "oneway": False,
            "to_reverse": True,
            "_status": "new",
        },
        schema_overrides={"lanes_imputed": pl.UInt8},
    )

    new_gdf = pl.concat([attrs_from_og_df, attrs_wo_og_df], how="diagonal")
    new_gdf = (
        neat_edges[["geometry"]]
        .join(new_gdf.to_pandas().set_index("new_id"), how="inner")
        .drop(columns=["new_length"])
    )
    new_gdf.index.name = "new_id"
    return new_gdf


def graph_gdfs_from_edges(edges):
    """Get nodes and edges GeoDataFrames from edge geometries.

    Parameters
    ----------
    edges : GeoDataFrame
        GeoDataFrame of edges missing references to node IDs.

    Returns
    -------
    nodes, edges
        GeoDataFrames of nodes and edges with proper ID assignment.
    """
    edgelines = np.array(edges.geometry)
    start_points = shapely.get_point(edgelines, 0)
    end_points = shapely.get_point(edgelines, -1)
    coords = shapely.get_coordinates(np.concatenate([start_points, end_points]))

    # Get the DataFrame of nodes from the endpoints of the edge geometries and assign
    # them a unique ID.
    new_nodes = (
        pl.DataFrame(
            {"edge_id": np.tile(edges.index, 2), "x": coords[:, 0], "y": coords[:, 1]}
        )
        .group_by("x", "y")
        .agg(pl.col("edge_id"), pl.len().alias("degree"))
        .with_row_index("node_id")
    )
    # Get the DataFrame of edges from the nodes, so for every "edge_id", we have the ID
    # of the two end-nodes in "u" and "v".
    new_edges = (
        new_nodes.select("node_id", "edge_id")
        .explode("edge_id")
        .group_by("edge_id")
        .agg(pl.col("node_id"))
        .with_columns(
            pl.col("node_id").list.get(0).alias("u"),
            pl.col("node_id").list.get(1).alias("v"),
            pl.col("node_id").list.len().alias("nr_nodes"),
        )
        .with_columns(key=pl.cum_count("edge_id").over("u", "v") - 1)
        .drop("node_id")
    )
    assert new_edges["nr_nodes"].unique().item() == 2
    # Join back the geometries to make the GeoDataFrames:
    new_edges = gpd.GeoDataFrame(
        new_edges.drop("nr_nodes")
        .to_pandas()
        .set_index("edge_id")
        .join(edges.set_index("edge_id"), how="inner")
    )
    new_nodes = gpd.GeoDataFrame(
        new_nodes.drop("edge_id").to_pandas(),
        geometry=shapely.points(new_nodes.select("x", "y").to_numpy()),
    ).set_index("node_id")
    assert new_edges.shape[0] == edges.shape[0]
    return new_nodes, new_edges


def from_multi_to_digraph(
    edges: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    return (
        edges
        .sort_values("length", ascending=True)
        .groupby(["u", "v"])
        .head(1)
        .reset_index(drop=True)
    )
