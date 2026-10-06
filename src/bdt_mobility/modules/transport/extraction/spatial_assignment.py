import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, box
import networkx as nx

from bdt_mobility.utils.constants import CRS_PROJECTED
from bdt_mobility.utils.spatial_utils import closest_point, is_left, nearest_lines_within_buffer, unique_nearest_lines, cosine_similarity, calculate_direction, alignment_angle


def OD_to_ZOI(
    df_od: gpd.GeoDataFrame, df_zoi: gpd.GeoDataFrame, overlap_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Links the OD zones with the ZOI by calculating the overlap and discovering which are mostly inside.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones with their geometries.
        `df_zoi` (`gpd.GeoDataFrame`): GeoDataFrame of ZOI geometries.
        `overlap_threshold` (`float`): Minimum overlap ratio required for linking. Defaults to `0.5`.

    Returns:
        `gpd.GeoDataFrame`: The updated OD GeoDataFrame with a new column `mostly_within_zoi` indicating overlap.
    """
    df_od["area"] = df_od.area
    areas_intersects = df_od.reset_index(drop=True).overlay(
        df_zoi[["geometry"]], how="intersection"
    )
    ratio_overlap = areas_intersects.area / areas_intersects["area"]

    id_ok = areas_intersects[ratio_overlap > overlap_threshold]["id"].values
    df_od["mostly_within_zoi"] = False
    df_od.loc[df_od["id"].isin(id_ok), "mostly_within_zoi"] = True

    return df_od


def AOI_to_ZOI(
    df_od: gpd.GeoDataFrame, df_zoi: gpd.GeoDataFrame, overlap_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Links the AOI zones with the ZOI by calculating the overlap and discovering which are inside.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of AOI zones with their geometries.
        `df_zoi` (`gpd.GeoDataFrame`): GeoDataFrame of ZOI geometries.
        `overlap_threshold` (`float`): Minimum overlap ratio required for linking. Defaults to `0.5`.

    Returns:
        `gpd.GeoDataFrame`: The updated OD GeoDataFrame with a new column `mostly_within_zoi` indicating overlap.
    """
    df_od["area"] = df_od.area
    areas_intersects = df_od.reset_index(drop=True).overlay(
        df_zoi[["geometry"]], how="intersection"
    )
    ratio_overlap = areas_intersects.area / areas_intersects["area"]

    id_ok = areas_intersects[ratio_overlap > overlap_threshold]["id_zone"].values
    df_od["mostly_within_zoi"] = False
    df_od.loc[df_od["id_zone"].isin(id_ok), "mostly_within_zoi"] = True

    return df_od


def roads_to_AOI(
    df_edges: gpd.GeoDataFrame, df_aoi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns roads (edges) to Areas of Interest (AOI) by checking intersections.

    Args:
        `df_edges` (`gpd.GeoDataFrame`): GeoDataFrame of road network edges.
        `df_aoi` (`gpd.GeoDataFrame`): GeoDataFrame of AOI geometries with `id_zone` attributes.

    Returns:
        `gpd.GeoDataFrame`: Updated GeoDataFrame of edges with AOI zone assignments.
    """
    all_area = gpd.overlay(
        gpd.GeoDataFrame({'geometry' : [Polygon(df_edges.geometry.unary_union.convex_hull)], 
                          'name': ["outside"],
                          'id_zone': ['0']}, crs = df_edges.crs),
        df_aoi,
        how='difference'
    )
    df_aoi = gpd.GeoDataFrame(pd.concat([df_aoi, all_area]), crs=df_aoi.crs, geometry='geometry' )
    joined = gpd.sjoin(df_edges, df_aoi, how="left", predicate="intersects")
    joined["id_zone"] = joined["id_zone"].fillna("0")
    joined = joined.merge(df_aoi[["id_zone", "geometry"]], on="id_zone", how="left")
    joined["intersection_length"] = (
        joined["geometry_x"]
        .intersection(joined["geometry_y"])
        .length
        .fillna(0)          # dove geometry_y è None
    )
    joined_sorted = joined.sort_values(by="intersection_length", ascending=False)
    unique_matches = joined_sorted.drop_duplicates(subset=["geometry_x"], keep="first")

    df_edges = df_edges.merge(
        unique_matches[["u", "v", "key", "id_zone"]], on=["u", "v", "key"]
    )

    return df_edges


def nodes_to_ZOI(
    df_nodes: gpd.GeoDataFrame, df_zoi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns nodes to the ZOI by spatial intersection.

    Args:
        `df_nodes` (`gpd.GeoDataFrame`): GeoDataFrame of road network nodes.
        `df_zoi` (`gpd.GeoDataFrame`): GeoDataFrame of ZOI geometries.

    Returns:
        `gpd.GeoDataFrame`: Updated GeoDataFrame of nodes with their ZOI position (`inside`, `outside`).
    """
    df_nodes = gpd.sjoin(
        df_nodes, df_zoi[["geometry"]], how="left", predicate="intersects"
    )

    df_nodes["index_right"] = df_nodes["index_right"].fillna("outside")
    df_nodes.loc[df_nodes["index_right"] == 0, "index_right"] = "inside"
    df_nodes.rename(columns={"index_right": "zoi_position"}, inplace=True)

    return df_nodes


def nodes_to_AOI(
    df_nodes: gpd.GeoDataFrame, df_aoi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    df_nodes = gpd.sjoin(
        df_nodes, df_aoi[["id_zone", "geometry"]], how="left", predicate="intersects"
    )
    df_nodes = df_nodes.drop_duplicates(
        subset=df_nodes.columns.difference(["index_right", "id_zone"]).tolist(),
        keep="first",
    )
    df_nodes["id_zone"] = df_nodes["id_zone"].fillna(0)
    df_nodes = df_nodes.drop(columns="index_right")

    return df_nodes


def spira_to_road(
    df_spira: gpd.GeoDataFrame, 
    df_edge: gpd.GeoDataFrame,
    assignment_type: str = "physical_pos",
    max_search_distance: int|None = None,
) -> gpd.GeoDataFrame:
    """
    Assigns spira sensors to the closest road segments.

    Args:
        `df_spira` (`gpd.GeoDataFrame`): GeoDataFrame of spira sensor locations.
        `df_edge` (`gpd.GeoDataFrame`): GeoDataFrame of road network edges.
        `assignment_type` (`str`): either "angular_pos" for angular similatrity or 
                                "physical_pos" for relative position to the road (left/right, default)
        `max_search_distance` (`int` or `None`): maximum distance for searching the close roads in the "angular_pos" case.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame of spira sensors with road assignments.
    """
    # Initial checks
    if assignment_type == "angular_pos" and  "angle" not in df_spira.columns:
        ## "Using 'physical_pos' since there is no 'angle' attribute of the spira")
        assignment_type = "physical_pos"

    # Understang the directionality of the spiras and of the roads
    if assignment_type == "angular_pos":

        # Find the closest road to each spira
        spira_close_roads = nearest_lines_within_buffer(df_spira, df_edge, max_search_distance) # nearest lines in a 25m range
        spira_close_roads = spira_close_roads.merge(
            df_edge[["u", "v", "key", "geometry"]],
            on=["u", "v", "key"],
        )

        # Find the relative angle between spira and roads
        spira_close_roads['spira_road_angle'] = spira_close_roads.apply(
            lambda s: alignment_angle(s.geometry_x, s.geometry_y, s.angle), 
            axis=1
        )

        # Filtering: matches with less than 30 degrees of difference are keps
        spira_close_roads = spira_close_roads[abs(spira_close_roads['spira_road_angle'])<30].reset_index(drop=True)

        # Filtering: quality measure based on the distance normalized to 15m and angle normalized to 30 degrees so both are taken into consideration fo the alignment
        spira_close_roads['quality'] = (spira_close_roads['distance_to_road']/15)+(abs(spira_close_roads['spira_road_angle'])/30)
        
        closest_indexes = spira_close_roads.groupby(['spira_unique_id', 'latitudine', 'longitudine'])['quality'].idxmin()
        spira_closest_roads = spira_close_roads.loc[closest_indexes].reset_index(drop=True)
        spira_closest_roads = spira_closest_roads[spira_closest_roads['quality']<2].reset_index(drop=True)

    # Apply the function to understand if the spira is on the left of the road
    if assignment_type == "physical_pos":
        # Find the closest road to each spira
        spira_close_roads = unique_nearest_lines(df_spira, df_edge)
        spira_close_roads = spira_close_roads.merge(
            df_edge[["u", "v", "key", "geometry", "oneway"]],
            on=["u", "v", "key"],
        )

        pt_series = gpd.GeoSeries(spira_close_roads["geometry_x"])
        ln_series = gpd.GeoSeries(spira_close_roads["geometry_y"])
        spira_close_roads["spira_left_location"] = [
            is_left(linestring=ln, point=pt) for ln, pt in zip(ln_series, pt_series)
        ]
        # Assign the real road
        spira_closest_roads = spira_close_roads[
            (spira_close_roads["oneway"] == True)
            | (
                (spira_close_roads["oneway"] == False)
                & (spira_close_roads["spira_left_location"] == False)
            )
        ]
        spira_closest_roads = (
            spira_closest_roads.drop_duplicates(subset="spira_unique_id")
            .reset_index(drop=True)
            .drop(columns=["spira_left_location", "oneway"])
        )

    # Final check (ensure that all the spiras are returned)
    spira_closest_roads = pd.merge(
        df_spira, 
        spira_closest_roads.drop(columns=["geometry_x"]), 
        how="left", 
        on=["spira_unique_id", "longitudine", "latitudine", "angle"])
    spira_closest_roads = spira_closest_roads.rename(columns={'geometry':'geometry_x'}).reset_index(drop=True)
    return spira_closest_roads


def gate_to_road(
    df_gate: gpd.GeoDataFrame, 
    df_edge: gpd.GeoDataFrame,
    fix_manual: bool = False
) -> gpd.GeoDataFrame:
    # Use the spira_to_road function
    df_gate = df_gate.rename(columns={'ID': 'spira_unique_id', 'Latitudine':"latitudine", 'Longitudine':"longitudine"})
    df_gate["angle"] = 0
    df_gate = (
        spira_to_road(df_spira=df_gate, df_edge=df_edge)
        .rename(columns={"spira_unique_id": "ID", 'latitudine':"Latitudine", 'longitudine':"Longitudine",
                         "gate_x": "gate", 'Indirizzo_x':'Indirizzo', 'Settore_x':'Settore', 'Link google maps_x':'Link google maps' }))
    df_gate = df_gate[['ID', 'gate', 'Latitudine', 'Longitudine',  'Indirizzo', 
          'Settore', 'Link google maps', 'geometry_x', 'u', 'v', 'key']]
    return df_gate   


def tomtom_traffic_to_road(
    gdf_osm:gpd.GeoDataFrame,
    gdf_tom:gpd.GeoDataFrame,
    t_buffer:int = 5,
    t_perc_overlap:float = 0.0,
    t_dir_sim:float = -1.0
):
    buffered_gdf1 = gdf_osm.copy()
    buffered_gdf1['geometry'] = buffered_gdf1.geometry.buffer(t_buffer)

    candidates = gpd.sjoin(gdf_tom, buffered_gdf1, how='inner', predicate='intersects')
    results = []
    for idx2, row2 in candidates.iterrows():
        idx1 = row2['index_right']  
        line1 = gdf_osm.loc[idx1].geometry
        buffer1 = buffered_gdf1.loc[idx1].geometry 
        line2 = row2.geometry

        # Overlap
        if line2.length == 0:
            continue
        intersection = buffer1.intersection(line2)
        if intersection.is_empty or not isinstance(intersection, (LineString, MultiLineString)):
            continue

        overlap_length = intersection.length if not intersection.is_empty else 0
        perc_overlap = overlap_length / line2.length

        # Direction
        dir1 = calculate_direction(line1)
        dir2 = calculate_direction(line2)
        dir_sim = cosine_similarity(dir1, dir2)
        
        if perc_overlap >= t_perc_overlap and dir_sim >= t_dir_sim:
            results.append({
                'osm_id': gdf_osm.loc[idx1, 'id'],
                'tomtom_id': row2.loc['segmentId'],
                'overlap_percentage': perc_overlap,
                'direction_similarity': dir_sim
            })

    results = pd.DataFrame(results)
    results = results.loc[results.groupby('tomtom_id')['overlap_percentage'].idxmax()]
    return results


def OD_to_circles(df_od: gpd.GeoDataFrame, df_circles: gpd.GeoDataFrame) -> dict:
    """
    Assigns OD zones to the closest circles (inside or outside).

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones.
        `df_circles` (`gpd.GeoDataFrame`): GeoDataFrame of circles.

    Returns:
        `dict`: A dictionary mapping OD zone IDs to circle IDs.
    """
    out_id_series = df_od["geometry"].apply(
        lambda x: closest_point(
            x,
            df_circles[["id"]]
            .assign(centroid=df_circles.geometry.centroid)
            .set_geometry("centroid")
            .set_crs(CRS_PROJECTED),
        )
    )
    out_id_series[df_od["mostly_within_zoi"]] = df_od.loc[
        df_od["mostly_within_zoi"], "id"
    ]

    id_to_out_id_dict = dict(zip(df_od["id"], out_id_series))
    return id_to_out_id_dict


def OD_to_comune(
    areas_od: gpd.GeoDataFrame, 
    comune: gpd.GeoDataFrame
) -> dict[int, list[int]]:
    areas_gdf = areas_od.dissolve("id", as_index=True)
    areas_gdf = areas_gdf.to_crs(CRS_PROJECTED)
    areas_gdf.columns = areas_gdf.columns.str.lower()
    com_id = comune.index.name
    comune_proj = comune[[com_id, "geometry"]].to_crs(CRS_PROJECTED)

    # Match areas to comune with most spatial overlap
    areas_gdf["area"] = areas_gdf.area
    areas_intersects = (
        areas_gdf[["geometry", "area"]]
        .reset_index()
        .overlay(comune_proj, how="intersection")
    )
    areas_intersects["area_overlap"] = areas_intersects.area
    areas_intersects["ratio_overlap"] = (
        areas_intersects["area_overlap"] / areas_intersects["area"]
    )
    top_intersect = (
        areas_intersects.sort_values(["id", "ratio_overlap"])
        .groupby("id")
        .agg({"ratio_overlap": "last", com_id: "last"})
    )
    areas_gdf = areas_gdf.join(top_intersect[com_id])

    # Handle areas that are external: consider them as the closest comune
    areas_gdf.update(
        areas_gdf.loc[areas_gdf[com_id].isnull()]
        .sjoin_nearest(comune_proj[["geometry"]])["com_id_right"]
        .rename("com_id")
    )

    # Convert to dictionary with id as key and list of comune ids as values
    areas_gdf = pd.DataFrame(areas_gdf.reset_index(names="id")[["id", com_id]])
    areas_gdf[com_id] = areas_gdf[com_id].astype("int")
    areas_gdf["id"] = areas_gdf["id"].astype("int")
    areas_to_com = areas_gdf.groupby("id")[com_id].apply(list).to_dict()
    
    return areas_to_com


def classify_in_out_zones_bbox(
    bbox,
    zones: gpd.GeoDataFrame,
    id_zone_col: str = "id_zone",
):
    '''
    Get the ids of the zones inside a bbox, or outside a bbox.
    '''
#    bbox = nodes.total_bounds
    zones_in = zones[zones.within(gpd.GeoSeries(box(*bbox), crs=zones.crs).iloc[0])]
    id_zones_in = zones_in[id_zone_col].unique()
    zones_in = zones[~zones.within(gpd.GeoSeries(box(*bbox), crs=zones.crs).iloc[0])]
    id_zones_out = zones_in[id_zone_col].unique()

    return id_zones_in, id_zones_out


def get_inside_nodes_per_zone(
    gdf_nodes: gpd.GeoDataFrame, 
    id_node_col: str,
    gdf_zones: gpd.GeoDataFrame, 
    id_zone_col: str, 
    random_seed: int
) -> dict[int, list[int]]:
    '''
    This function takes a geodataframe of nodes and a geodataframe of zones,
    and for each zone it finds the nodes that are within the zone.
    It returns a dictionary with the zone id as key and the list of node ids within the zone as value.
    '''
    
    zone_per_node_dict = get_zone_per_node(gdf_nodes, id_node_col, gdf_zones, id_zone_col, random_seed)

    inside_nodes_dict = {}
    for node_id, zone_id in zone_per_node_dict.items():
        if inside_nodes_dict.get(int(zone_id)):
            inside_nodes_dict[int(zone_id)].append(int(node_id))
        else:
            inside_nodes_dict[int(zone_id)] = [int(node_id)]

    return inside_nodes_dict


def get_zone_per_node(
    gdf_nodes: gpd.GeoDataFrame, 
    id_node_col: str,
    gdf_zones: gpd.GeoDataFrame, 
    id_zone_col: str,
    random_seed: int = 42
) -> dict[int, int]:
    '''
    This function takes a geodataframe of nodes and a geodataframe of zones, 
    and returns a dictionary with the node id as key and the zone id as value for the nodes that are within the zones.
    If nodes are not assigned to any zone, they will not be included in the dictionary.
    If nodes are assigned to multiple zones, only one of the zones will be kept (the one that is randomly sampled).
    '''

    df = gpd.sjoin(
            gdf_nodes, 
            gdf_zones[[id_zone_col, "geometry"]], 
            how="left", 
            predicate="within"
        ).drop(columns=["index_right"])
    # remove the nodes that are not within any zone
    df = df[~df[id_zone_col].isna()]

    df = df.groupby(id_node_col).sample(n=1, random_state=random_seed).reset_index(drop=True)

    return dict(zip(df[id_node_col], df[id_zone_col]))


def get_n_closest_nodes_per_zone(
    gdf_nodes: gpd.GeoDataFrame, 
    id_node_col: str,
    gdf_zones: gpd.GeoDataFrame, 
    id_zone_col: str,
    n_closest: int = 3
) -> dict[int, list[int]]:
    '''
    This function takes a geodataframe of nodes and a geodataframe of zones, 
    and for each zone it finds the n closest nodes. 
    It returns a dictionary with the zone id as key and the list of closest node ids as value.
    '''
    
    closest_nodes_dict = {}
    
    for _, zone in gdf_zones.iterrows():
        zone_id = zone[id_zone_col]
        zone_geometry = zone["geometry"]
        
        # Calculate distances from the zone to all nodes
        gdf_nodes["distance"] = gdf_nodes.geometry.distance(zone_geometry)
        
        # Get the n closest nodes
        closest_nodes = gdf_nodes.nsmallest(n_closest, "distance")[id_node_col].tolist()
        
        # befor this, ensure that all elements are int, otherwise we will have problems with the get_node_prop function
        closest_nodes = [int(node) for node in closest_nodes]
        closest_nodes_dict[int(zone_id)] = closest_nodes
    
    return closest_nodes_dict


def remove_LTZ_edges_and_nodes(
    df_edges: gpd.GeoDataFrame, 
    df_nodes: gpd.GeoDataFrame,
    df_ltz: gpd.GeoDataFrame,
):    
    df_edges = df_edges.copy()
    df_nodes = df_nodes.copy()
          
    # Step 1: Remove nodes that are inside LTZ zones
    mask_outside_ltz = ~df_nodes.geometry.within(df_ltz.unary_union)
    df_nodes = df_nodes[mask_outside_ltz]
    
    # Step 2: Keep only edges where both u and v are in remaining nodes
    remaining_node_ids = set(df_nodes['id'])
    df_edges = df_edges[
        (df_edges['u'].isin(remaining_node_ids)) & 
        (df_edges['v'].isin(remaining_node_ids))
    ]
    
    # Step 3: Build a directed graph and, if not strongly connected, keep only the largest strongly connected component
    G = nx.DiGraph()
    G.add_nodes_from(remaining_node_ids)
    G.add_edges_from(df_edges[['u', 'v']].values)
    
    if not nx.is_strongly_connected(G):
        sccs = list(nx.strongly_connected_components(G))
        largest_scc = max(sccs, key=len)
        
        df_nodes = df_nodes[df_nodes['id'].isin(largest_scc)]
        df_edges = df_edges[
            (df_edges['u'].isin(largest_scc)) & 
            (df_edges['v'].isin(largest_scc))
        ]

    df_edges = df_edges.reset_index(drop=True)
    df_nodes = df_nodes.reset_index(drop=True)

    return df_edges, df_nodes