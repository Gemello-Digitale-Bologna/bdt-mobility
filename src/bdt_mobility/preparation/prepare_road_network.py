import digitalhub as dh
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from bdt_mobility.utils.constants import CRS_LATLONG
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.config import ROOT_PATH

from bdt_mobility.preparation.config import PROJECT_NAME as PROJECT
from bdt_mobility.preparation.config import YEAR
DATE = f'{YEAR}-01-01T00:00:00Z'

from pathlib import Path
from bdt_mobility.utils.config import DATA_PATH
ROAD_NETWORK_PATH = DATA_PATH / 'preparation' / 'road_network'
ROAD_NETWORK_PATH.mkdir(parents=True, exist_ok=True)


def _extract_osmnx_ltx_data(
    df_bzoi: gpd.GeoDataFrame,
    date: str|None = None
):
    """Extract the limited-traffic-zone boundary from OpenStreetMap.

    Args:
        df_bzoi (gpd.GeoDataFrame): GeoDataFrame representing the area of interest.
        date (str | None): Optional date string to query historical OSM data.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the limited-traffic-zone boundary geometries.
    """
    
    if date:
        __default_settings = ox.settings.overpass_settings
        __default_url = ox.settings.overpass_url
        ox.settings.overpass_url = "https://overpass-api.de/api"
        ox.settings.overpass_settings = '[out:json][timeout:{timeout}][date:"' + date + '"]{maxsize}'

    ltz_gdf = ox.features_from_polygon(
        polygon=df_bzoi.to_crs(CRS_LATLONG).geometry.iloc[0],
        tags={"boundary": "limited_traffic_zone"},
    )

    if date:
        ox.settings.overpass_settings = __default_settings
        ox.settings.overpass_url = __default_url

    return ltz_gdf.reset_index()


def _extract_osmnx_graph(
    df_bzoi: gpd.GeoDataFrame,
    relevant_highway: bool,
    date: str|None = None
) -> nx.MultiDiGraph:
    """
    Extracts road network data from OSMNX using a custom filter.

    Args:
        `df_bzoi` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.
        `date` (`str` | `None`): Optional date string to query historical OSM data.

    Returns:
        Graph: A MultiDiGraph representing the road network extracted from OSMNX.
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

    if date:
        __default_settings = ox.settings.overpass_settings
        __default_url = ox.settings.overpass_url
        ox.settings.overpass_url = "https://overpass-api.de/api"
        ox.settings.overpass_settings = '[out:json][timeout:{timeout}][date:"' + date + '"]{maxsize}'

    G = ox.graph_from_polygon(
        polygon=df_bzoi.to_crs(CRS_LATLONG).geometry.iloc[0],
        network_type="drive",
        simplify=False,
        custom_filter=custom_filter,
    )

    if date:
        ox.settings.overpass_settings = __default_settings
        ox.settings.overpass_url = __default_url

    return G


def main():
    logging.basicConfig(level=logging_level)

    logging.info("Loading data...")
    # Loading data
    filename = dh.get_dataitem(identifier="zoi_manual_border", project=PROJECT).as_file()
    zoi = data_reader.ZOI_shape(filename)
    zoi.geometry = zoi.buffer(3000)

    logging.info("Extracting road network...")
    # Extracting road network
    G_rn = _extract_osmnx_graph(df_bzoi=zoi, relevant_highway=False, date=DATE)

    logging.info("Filtering road network...")
    # Filtering 1: largest component only
    G_rn = ox.truncate.largest_component(G_rn, strongly=True)

    # Filtering 2: from MultiDiGraph to DiGraph
    G_rn = ox.convert.to_digraph(G_rn, weight="length")

    # Getting nodes and edges
    G_rn = nx.MultiDiGraph(G_rn)
    nodes, edges = ox.graph_to_gdfs(G_rn)

    logging.info("Extracting the Limited-Traffic-Zone boundary...")
    # Use OSMNX to extract the LTX boundary
    ltx_gdf = _extract_osmnx_ltx_data(df_bzoi=zoi, date=DATE)

    logging.info("Saving data...")
    
    # Saving
    edges = edges.reset_index()
    geometry_wkt = edges['geometry'].to_wkt()
    edges = pd.DataFrame(edges.drop(columns='geometry'))
    edges['geometry'] = geometry_wkt
    edges.to_parquet(ROAD_NETWORK_PATH / f"geo_edges_{YEAR}.parquet", index=False)
    # dh.log_dataitem(project=PROJECT, name=f"geo_edges_{YEAR}", data=edges, kind="table")

    nodes = nodes.reset_index()
    geometry_wkt = nodes['geometry'].to_wkt()
    nodes = pd.DataFrame(nodes.drop(columns='geometry'))
    nodes['geometry'] = geometry_wkt
    nodes.to_parquet(ROAD_NETWORK_PATH / f"geo_nodes_{YEAR}.parquet", index=False)
    # dh.log_dataitem(project=PROJECT, name=f"geo_nodes_{YEAR}", data=nodes, kind="table")

    geometry_wkt = ltx_gdf['geometry'].to_wkt()
    ltx_gdf = pd.DataFrame(ltx_gdf.drop(columns='geometry'))
    ltx_gdf['geometry'] = geometry_wkt
    ltx_gdf.to_parquet(ROAD_NETWORK_PATH / f"ltz_boundary_{YEAR}.parquet", index=False)
    # dh.log_dataitem(project=PROJECT, name=f"ltz_boundary_{YEAR}", data=ltx_gdf, kind="table")


if __name__ == "__main__":
    main()

# uv run python -m bdt_mobility.preparation.prepare_road_network