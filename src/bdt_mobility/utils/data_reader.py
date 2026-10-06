from pathlib import Path
from . import data_cleaner
from shapely.geometry import LineString
from shapely import wkt
import geopandas as gpd
import json
import pandas as pd
import numpy as np
from .constants import CRS_LATLONG, CRS_PROJECTED


def OD_shapes(
    path_polygons: Path | str,
    path_centers: Path | str | None = None,
) -> gpd.GeoDataFrame:
    """Read OD (origin-destination) polygon shapes from a local file.

    The GeoDataFrame CRS is set to ``CRS_PROJECTED``. If ``path_centers`` is
    provided, centroids from that file are buffered by 1000 meters and appended
    as external zones. This function does not currently dissolve or otherwise
    modify polygon geometries beyond CRS conversion and simple concatenation.

    Args:
        path_polygons (Path | str): Full path to the polygon shapefile.
        path_centers (Path | str | None): Full path to centroid points to
            include as external zones. Defaults to ``None``.

    Returns:
        gpd.GeoDataFrame: Processed OD shapes with lowercase column names and
            an added ``type2`` column indicating "internal" or "external".
    """
    df = (
        gpd.read_file(path_polygons)
        .set_crs("EPSG:23032")
        .to_crs(CRS_PROJECTED)
        .rename(columns={"NO": "id"})
        .astype({"id": int, "TYPENO": int, "NAME": pd.StringDtype()})
    )
    df.columns = df.columns.str.lower()
    # df['geometry'] = df.geometry.buffer(0)
    # df = df.dissolve(by='id').reset_index() #fix the multiple polygons to single multipolygon
    df["type2"] = "internal"

    if path_centers is not None:
        df_centers = OD_centers(path_centers)
        df_centers = df_centers[~(df_centers["id"].isin(df["id"]))]
        df_centers["geometry"] = df_centers.geometry.buffer(1000)
        df_centers["type2"] = "external"
        df = pd.concat([df, df_centers], ignore_index=True)
    return df


def OD_centers(
    path: Path | str,
) -> gpd.GeoDataFrame:
    """Read OD centroid points from a local file and set CRS to ``CRS_PROJECTED``.

    Args:
        path (Path | str): Full path to the centroid shapefile.

    Returns:
        gpd.GeoDataFrame: Processed centroid GeoDataFrame with lowercase
            column names and typed columns.
    """
    df_centers = (
        gpd.read_file(path)
        .set_crs("EPSG:23032")
        .to_crs(CRS_PROJECTED)
        .rename(columns={"NO": "id"})
        .astype({"id": int, "TYPENO": int, "NAME": pd.StringDtype()})
    )
    df_centers.columns = df_centers.columns.str.lower()
    return df_centers


def ZOI_shape(
    path: Path | str,
) -> gpd.GeoDataFrame:
    """Read the ZOI (Zone of Interest) shape and set CRS to ``CRS_PROJECTED``.

    Args:
        path (Path | str): Full path to the GeoJSON (or other spatial) file.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the processed ZOI shape.
    """
    df = (
        gpd.read_file(path)
        .set_crs(CRS_LATLONG)
        .to_crs(CRS_PROJECTED)
    )
    return df


def AOI_shapes(
    path: Path | str,
    aoi_type: str,
    df_around: gpd.GeoDataFrame,
    return_type: str = "inside"
):
    """Load Area-Of-Interest shapes and return zones inside/outside a given area.

    The function supports three ``aoi_type`` values: "od", "census", or "zoi".
    All spatial outputs use ``CRS_PROJECTED``. The function requires a
    GeoDataFrame ``df_around`` to compute intersection/difference and returns
    either the inside, outside, or both results depending on ``return_type``.

    Args:
        path (Path | str): Full path to the file to load (polygons, census,
            or ZOI boundary depending on ``aoi_type``).
        aoi_type (str): One of "od", "census", or "zoi".
        df_around (gpd.GeoDataFrame): GeoDataFrame used to crop/select zones.
        return_type (str): One of "inside", "outside", or any other value to
            receive a tuple ``(inside, outside)``. Defaults to "inside".

    Returns:
        gpd.GeoDataFrame or tuple: Depending on ``return_type``, returns the
            inside GeoDataFrame, the outside GeoDataFrame, or a tuple
            ``(inside, outside)``.
    """
    if aoi_type == "od":
        df = OD_shapes(path_polygons=path)
        df_inside, df_outside = data_cleaner.AOI_OD_shapes(df, df_around)
        df = gpd.GeoDataFrame(
            pd.concat([df_inside, df_outside], axis=0)[["id_zone", "name", "geometry"]], 
            geometry="geometry", 
            crs=df_inside.crs)

    elif aoi_type == "census":
        df = (
            gpd.read_file(path)
            .set_crs(CRS_LATLONG)
            .to_crs(CRS_PROJECTED)
            .dissolve(by="zona")
            .reset_index()
            .rename({"zona": "id_zone"}, axis=1)[["id_zone", "geometry"]]
            .drop_duplicates()
        )
        df['name'] = df["id_zone"]
    elif aoi_type == "zoi":
        df = ZOI_shape(path)
        df["id_zone"] = "1"
        df["name"] = "ZOI"

    df["id_zone"] = df["id_zone"].astype(str)
    df1 = gpd.overlay(df, df_around, how="intersection", keep_geom_type=True)
    df2 = gpd.overlay(df, df_around, how="difference", keep_geom_type=True)
    
    # Filtering of non-relevant overlaps
    zonal_areas1 = df1.geometry.area
    df1 =  df1[zonal_areas1 > 1] # only significant areas (> 1 m^2) are kept
    zonal_areas2 = df2.geometry.area
    df2 =  df2[zonal_areas2 > 1] # only significant areas (> 1 m^2) are kept

    # If multiple overlaps are still present, join them in 1 element with union geometries
    df1 = df1.dissolve(by="id_zone").reset_index()
    df2 = df2.dissolve(by="id_zone").reset_index()

    if return_type == "inside":
        return df1
    elif return_type == "outside":
        return df2
    else:
        return df1, df2


def OD_flows(
    path: Path | str,
    df_shapes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load OD matrix of traffic flows [unit: people].

    Filters flows from/to selected shapes if ``df_shapes`` is provided.

    Args:
        path (Path | str): Full path to the OD Excel file.
        df_shapes (pd.DataFrame | None): DataFrame for filtering flows.
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the OD matrix flows.
    """
    df_od = pd.read_excel(path).rename(
        columns={"NumDaZona": "from", "NumAZona": "to", "ValMatr(1 Totale_H24)": "flow"}
    )[
        ["from", "to", "flow"]
    ]

    if df_shapes is None or df_shapes.empty:
        return df_od
    else:
        return data_cleaner.OD_flows(df_od, df_shapes)


def road_data(
    edges_path: Path | str,
    nodes_path: Path | str,
) -> list:
    """Load road network edges and nodes from parquet files.

    Args:
        edges_path (Path | str): Full path to the edges parquet file.
        nodes_path (Path | str): Full path to the nodes parquet file.

    Returns:
        list[gpd.GeoDataFrame]: ``[edges, nodes]`` GeoDataFrames with CRS set
            to the stored CRS (no conversion is performed here).
    """
    edges = pd.read_parquet(edges_path)
    edges["geometry"] = edges["geometry"].apply(wkt.loads)
    edges = gpd.GeoDataFrame(edges, crs=CRS_LATLONG, geometry="geometry")
    edges = edges.sort_values(by=['u','v','key'])

    nodes = pd.read_parquet(nodes_path)
    nodes["geometry"] = nodes["geometry"].apply(wkt.loads)
    nodes = gpd.GeoDataFrame(nodes, crs=CRS_LATLONG, geometry="geometry")
    nodes = nodes.sort_values(by='osmid')
    
    return [edges, nodes]


def road_neat_data(
    edges_path: Path | str,
    nodes_path: Path | str,
) -> list:
    """Load road network data from neatnet as edges and nodes GeoDataFrames.

    Args:
        edges_path (Path | str): Full path to the edges parquet file.
        nodes_path (Path | str): Full path to the nodes parquet file.

    Returns:
        list[gpd.GeoDataFrame]: ``[edges, nodes]`` GeoDataFrames.
    """
    edges = gpd.read_parquet(edges_path)
    edges.loc[edges['length'].isna(), 'length'] = edges.loc[edges['length'].isna(), 'geometry'].length
    edges['free_flow_time'] = edges['length'] / edges['maxspeed_imputed'] #measured in seconds
    edges['link_id'] = edges.index
    edges['highway'] = None
    edges['highway_ok'] = True
    edges = edges.reset_index(drop=True).set_geometry("geometry").set_crs(CRS_PROJECTED)

    edges['length'] = edges['length'].astype(float)
    edges['maxspeed_imputed'] = edges['maxspeed_imputed'].astype(float)
    edges['lanes_imputed'] = edges['lanes_imputed'].astype(float)
    edges['link_id'] = edges['link_id'].astype(str)
    edges['u'] = edges['u'].astype(str)
    edges['v'] = edges['v'].astype(str)

    nodes = gpd.read_parquet(Path(datapath) / nodes_namefile)
    nodes = nodes.rename(columns={'x':'coordx', 'y':'coordy'})[['node_id', 'coordx', 'coordy', 'geometry']]
    nodes = nodes.set_geometry("geometry").set_crs(CRS_PROJECTED)

    nodes['node_id'] = nodes['node_id'].astype(str)
    nodes['coordx'] = nodes['coordx'].astype(float)
    nodes['coordy'] = nodes['coordy'].astype(float)

    return [edges, nodes]


def spira_shapes(
    path: Path | str,
    df_around: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Load Spira point shapes from parquet and optionally crop to ``df_around``.

    Args:
        path (Path | str): Full path to the parquet file containing Spira points.
        df_around (gpd.GeoDataFrame | None): If provided, only keep points
            that intersect ``df_around``.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with columns ``spira_unique_id``,
            ``longitudine``, ``latitudine``, ``angle``, and ``geometry`` in
            ``CRS_PROJECTED``.
    """
    spira = pd.read_parquet(path)
    spira = (
        spira[["spira_unique_id", "longitudine", "latitudine", "angle"]]
        .drop_duplicates()
        .groupby("spira_unique_id")
        .first()
        .reset_index()
    )
    spira = gpd.GeoDataFrame(
        spira,
        geometry=gpd.points_from_xy(spira.longitudine, spira.latitudine),
        crs=CRS_LATLONG,
    ).to_crs(CRS_PROJECTED)

    if df_around is not None and not df_around.empty:
        spira = gpd.overlay(
            spira.set_crs(CRS_PROJECTED),
            df_around.set_crs(CRS_PROJECTED),
            how="intersection",
        )

    return spira[["spira_unique_id", "longitudine", "latitudine", "angle", "geometry"]]


def spira_codes(
    path: Path | str,
) -> gpd.GeoDataFrame:
    """Read Spira codes from a parquet file.

    Args:
        path (Path | str): Full path to the parquet file.

    Returns:
        gpd.GeoDataFrame: DataFrame with Spira unique IDs ``spira_unique_id``
            and sensor codes ``spira_code``.
    """
    spira = pd.read_parquet(path)

    return spira[["spira_unique_id", "spira_code"]].drop_duplicates()


def spira_flows(
    path: Path | str,
    filters: list | None = None,
    clean: bool = False,
) -> pd.DataFrame:
    """Read Spira flows data from a parquet file.

    Args:
        path (Path | str): Full path to the parquet file.
        filters (list | None): List of filtering conditions for reading.
            Defaults to ``None``.
        clean (bool): Whether to apply a cleaning filter to the data.
            Defaults to ``False``.

    Returns:
        pd.DataFrame: DataFrame with Spira sensor flow data.
    """
    spira = pd.read_parquet(path, filters=filters, engine="pyarrow")

    spira = spira[["sensor_id", "value", "start"]].rename(
        columns={"value": "count", "start": "DateTime"}
    )
    spira["DateTime"] = pd.to_datetime(spira["DateTime"], format="%Y-%m-%d %H:%M")
    spira = spira.sort_values("DateTime")

    if clean:
        spira = data_cleaner.spira_flows(df_spiras=spira)

    return spira


def spira_accuracy(
    path: Path | str,
    filters: list | None = None,
) -> pd.DataFrame:
    """Read Spira accuracy parquet data and parse the ``DateTime`` column.

    Args:
        path (Path | str): Full path to the parquet file.
        filters (list | None): Optional parquet filters passed to
            ``read_parquet``.

    Returns:
        pd.DataFrame: DataFrame with a parsed ``DateTime`` column.
    """
    spira = pd.read_parquet(path, filters=filters)
    spira["DateTime"] = pd.to_datetime(spira["DateTime"])
    return spira


def comune_shapes(
    path: Path | str,
):
    """Load municipality shapes and normalize columns.

    Args:
        path (Path | str): Full path to the shapefile.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame indexed by ``com_id`` with original
            geometry and lowercase columns.
    """
    comune_gdf = gpd.read_file(path)
    comune_gdf.columns = comune_gdf.columns.str.lower()
    comune_gdf = comune_gdf.rename(columns={"pro_com": "com_id"}).set_index(
        "com_id", drop=False
    )
    return comune_gdf


def flow_array(
    path: Path | str,
    column: str,
):
    """Return a numpy array extracted from a parquet column of lists.

    Args:
        path (Path | str): Full path to the parquet file.
        column (str): Column name containing iterable values to convert.

    Returns:
        np.ndarray: Numpy array built from the column values.
    """
    flow = pd.read_parquet(path)
    flow = np.array(flow[column].to_list())
    return flow


def tomtom_traffic_shape(
    datapath: str, 
    namefile: str
):
    """Parse TomTom traffic JSON and return segment geometries as GeoDataFrame.

    Args:
        datapath (str): Directory where the JSON file is located.
        namefile (str): JSON filename.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with TomTom segment geometries in
            ``CRS_PROJECTED``.
    """
    with open(f"{datapath}/{namefile}", 'r', encoding='utf-8') as f:
        traffic_data = json.load(f)

    segment_shape = pd.DataFrame(traffic_data['network']['segmentResults'])
    segment_shape['geometry'] = segment_shape['shape'].apply(
        lambda coords: LineString([(point['longitude'], point['latitude']) for point in coords])
    )
    segment_shape = gpd.GeoDataFrame(
        segment_shape, crs=CRS_LATLONG, geometry="geometry"
    ).to_crs(CRS_PROJECTED)
    segment_shape = segment_shape.drop(columns=['shape', 'segmentTimeResults'])

    return segment_shape

    
def tomtom_traffic_measures(
    datapath: str, 
    namefile: str
):
    """Parse TomTom traffic JSON and return per-segment time measures.

    Args:
        datapath (str): Directory where the JSON file is located.
        namefile (str): JSON filename.

    Returns:
        pd.DataFrame: Exploded DataFrame of segment time measures with human
            readable ``timeSet_name`` and ``dateRange_name`` columns.
    """
    with open(f"{datapath}/{namefile}", 'r', encoding='utf-8') as f:
        traffic_data = json.load(f)

    # Create segment measure data
    segment_data = pd.DataFrame(traffic_data['network']['segmentResults'])
    segment_data = segment_data.explode('segmentTimeResults', ignore_index=True)
    segment_time_df = pd.json_normalize(segment_data['segmentTimeResults'])
    segment_data = pd.concat([segment_data[['segmentId']], segment_time_df], axis=1)
    
    #Explicit time variables from id to names
    timeSets_dict = {}
    dateRanges_dict = {}
    for item in traffic_data['timeSets']:
        timeSets_dict[item['@id']] = item
    for item in traffic_data['dateRanges']:
        dateRanges_dict[item['@id']] = item
    segment_data['timeSet_name'] = segment_data['timeSet'].apply(lambda x: timeSets_dict[x]['name'])
    segment_data['dateRange_name'] = segment_data['dateRange'].apply(lambda x: dateRanges_dict[x]['name'])

    return segment_data

def open_data(
    path: Path | str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Standardize column names
    df = df.rename(columns={"yyyy": "anno","codareast": "codice_area_statistica"})
    if "codice_area_statistica" in df.columns:
        df["codice_area_statistica"] = df["codice_area_statistica"].astype("int64")
    # Clean year
    df["anno"] = (df["anno"].astype(str).str[:4])
    # Filter years
    df = data_cleaner.get_filtered_years(df,year_column ="anno", min_year=min_year, max_year=max_year)
    # Remove excluded areas
    if "aree_escl" in df.columns:
        df = df[df["aree_escl"] != 1]  
    return df

def statistical_areas(
    path: Path | str,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Standardize column names
    df = df.rename(columns={"codareast": "codice_area_statistica"})
     # Clean area names
    if "zona" in df.columns:
        df = data_cleaner.clean_name_as(df, colum_name="zona")
    return df
    

def LTZ_shape(
    path: Path | str,
) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, crs=CRS_LATLONG, geometry="geometry").to_crs(CRS_PROJECTED)
    return df