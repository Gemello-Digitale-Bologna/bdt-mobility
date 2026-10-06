"""Crosswalk between statistical areas and PUMS OD zones, weighted by overlap area."""
import geopandas as gpd
import pandas as pd
from shapely import wkb

from bdt_mobility.utils import data_reader
from bdt_mobility.utils import data_cleaner
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils import constants
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("map_as_zonepums", PathRegistry.MAP_AS_ZONEPUMS, "table"),
]


def _decode_statistical_areas_geometry(df_statistical_areas: pd.DataFrame) -> gpd.GeoDataFrame:

    # convert from raw geometru column
    geo_shape = df_statistical_areas["geo_shape"]
    geometry = geo_shape.map(wkb.loads)
    # copy the table
    df_attributes = df_statistical_areas.drop(columns=["geo_shape"]).copy()
    # attach decoded geometries and state CRS
    gdf = gpd.GeoDataFrame(df_attributes,geometry=geometry,crs=constants.CRS_LATLONG,)
    # projected CRS
    gdf = gdf.to_crs(constants.CRS_PROJECTED)
    # compute area_statistica area
    gdf["source_area_geometry_area"] = gdf.geometry.area
    return gdf


def _format_geo_point_2d(value) -> str:

    # hlper to just return string from many formats
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        point = wkb.loads(value)
        lon = point.x
        lat = point.y
        return f'{{ "lon": {lon}, "lat": {lat} }}'
    return str(value)


def build_zone_pums_crosswalk_df() -> gpd.GeoDataFrame:

    # load statistical areas
    df_statistical_areas = data_reader.statistical_areas(PathRegistry.FRAGILITIES_STATISTICAL_AREAS)
    # produce the geometry
    gdf_statistical_areas = _decode_statistical_areas_geometry(df_statistical_areas)
    # cast to string
    geo_points = gdf_statistical_areas["geo_point_2d"]
    formatted_geo_points = geo_points.map(_format_geo_point_2d)
    gdf_statistical_areas["geo_point_2d"] = formatted_geo_points

    # load ZOI
    gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)
    # load od zones
    gdf_od = data_reader.OD_shapes(PathRegistry.OD_SHAPE_ZONE)
    # (copyed, clip to gdf inside)
    gdf_od_inside, _ = data_cleaner.AOI_OD_shapes(gdf_od, gdf_zoi)

    od_cols = ["id_zone", "code", "name", "typeno", "type2", "geometry"]
    area_cols = [
        "codice_area_statistica",
        "geo_point_2d",
        "area_statistica",
        "cod_quar",
        "quartiere",
        "cod_zona",
        "zona",
        "source_area_geometry_area",
        "geometry",
    ]

    # actual overlap computation
    od_subset = gdf_od_inside[od_cols]
    statistical_areas_subset = gdf_statistical_areas[area_cols]
    crosswalk = gpd.overlay(od_subset,statistical_areas_subset,how="intersection",keep_geom_type=True,)
    # weight computation
    crosswalk["perc_area_intersect"] = crosswalk.geometry.area / crosswalk["source_area_geometry_area"]

    # final product
    crosswalk = crosswalk[
        [
            "id_zone",
            "code",
            "name",
            "typeno",
            "type2",
            "codice_area_statistica",
            "geo_point_2d",
            "area_statistica",
            "cod_quar",
            "quartiere",
            "cod_zona",
            "zona",
            "geometry",
            "perc_area_intersect",
        ]
    ].sort_values(["id_zone", "codice_area_statistica"]).reset_index(drop=True)

    return crosswalk

def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Building the statistical-area / PUMS-zone crosswalk")
        build_zone_pums_crosswalk_df().to_parquet(PathRegistry.MAP_AS_ZONEPUMS)
        logging.info("Crosswalk written")


if __name__ == "__main__":
    main()
