import geopandas as gpd

from bdt_mobility.utils import data_reader
from bdt_mobility.modules.transport.extraction import spatial_assignment
from bdt_mobility.utils.constants import CRS_PROJECTED, CRS_LATLONG

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging_level, logging
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("nodes", PathRegistry.SIM_NODES, "table"),
    PlatformOutput("edges", PathRegistry.SIM_EDGES, "table"),
]


def prepare_nodes_and_edges(
    road_network: str,
    activate_ltz: bool,
    set_aoi_type: str,
):
    # Load the shapes
    gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)
    if set_aoi_type == "od":
        aoi = data_reader.AOI_shapes(PathRegistry.OD_SHAPE_ZONE, aoi_type="od", df_around=gdf_zoi, return_type="inside")
    elif set_aoi_type == "census":
        raise NotImplementedError()

    # Load the road network
    if road_network == "dsf":
        edges_path, nodes_path = PathRegistry.DSF_EDGES, PathRegistry.DSF_NODES
    elif road_network == "geo":
        edges_path, nodes_path = PathRegistry.GEO_EDGES, PathRegistry.GEO_NODES
    elif road_network == "geo_all":
        edges_path, nodes_path = PathRegistry.GEO_EDGES_ALL, PathRegistry.GEO_NODES_ALL
    edges = gpd.read_parquet(edges_path)
    nodes = gpd.read_parquet(nodes_path)

    if activate_ltz:
        ltz_zone = data_reader.LTZ_shape(PathRegistry.LTZ_BOUNDARY)
        edges, nodes = spatial_assignment.remove_LTZ_edges_and_nodes(df_edges=edges, df_nodes=nodes, df_ltz=ltz_zone)

    # Assign nodes to in/out ZOI and to the zones of interest
    nodes_zoi = spatial_assignment.nodes_to_ZOI(df_nodes=nodes, df_zoi=gdf_zoi)
    nodes_zoi["is_in_zoi"] = nodes_zoi["zoi_position"] == "inside"
    nodes_zoi = spatial_assignment.nodes_to_AOI(df_nodes=nodes_zoi, df_aoi=aoi)
    nodes_zoi = nodes_zoi.drop(['geometry'], axis=1)

    # Fix nodes types and filter nodes
    column_types = {
        "is_in_zoi": "bool",
        "id_zone": "int64",
    }
    nodes_zoi = nodes_zoi.astype(column_types)

    # Assign edges (i.e., roads) to the area of interest they mostly belong
    edges_aoi = spatial_assignment.roads_to_AOI(df_edges=edges, df_aoi=aoi)
    edges_aoi = edges_aoi.drop(['geometry'], axis=1)

    # Fix edge types
    column_types = {
        'id_zone': "int64"
    }
    edges_aoi = edges_aoi.astype(column_types)

    return nodes_zoi, edges_aoi


def main(ctx: dict = context):
    logging.basicConfig(level=logging_level)
    with timed():
        keys = ["road_network", "activate_ltz", "set_aoi_type"]
        params = {k: ctx[k] for k in keys}
        nds, egs = prepare_nodes_and_edges(**params)

        PathRegistry.TRAFFIC_SIMULATOR.mkdir(parents=True, exist_ok=True)
        nds.to_csv(PathRegistry.SIM_NODES, index=False)
        egs.to_csv(PathRegistry.SIM_EDGES, index=False)


if __name__ == "__main__":
    main(ctx=context)
