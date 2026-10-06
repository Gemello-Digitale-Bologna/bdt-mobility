import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import json

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.constants import CRS_LATLONG
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("poi_h3_assigned", PathRegistry.POI_H3_ASSIGNED, "table"),
    PlatformOutput("hex_poi_stat", PathRegistry.HEX_POI_STAT, "table"),

]

def create_h3_grid(gdf_boundary : gpd.GeoDataFrame, resolution: int = 9)-> gpd.GeoDataFrame:
  """
  Generates an H3 hexagonal grid clipped to the boundary of the input GeoDataFrame.

  Only hexagons whose centroid falls within the boundary polygon are retained,
  ensuring the grid matches the study area without including edge hexagons
  whose centre lies outside.

  Parameters
  ----------
  gdf_boundary : gpd.GeoDataFrame
      GeoDataFrame defining the study area boundary.
      Multiple geometries are merged into a single polygon via unary_union.
  resolution : int, optional
      H3 resolution level (default: 9, ~200m side length).

  Returns
  -------
  gpd.GeoDataFrame
      GeoDataFrame with columns: hex_id, centroid_lat, centroid_lon, geometry.
      CRS: EPSG:4326.
  """
  
  boundary_polygon = unary_union(gdf_boundary.geometry)
 
  # Serialize polygon to GeoJSON format for h3
  boundary_geojson = json.loads(gpd.GeoSeries([boundary_polygon]).to_json())
  boundary_geojson_geom = boundary_geojson["features"][0]["geometry"]
  
  hexagons = list(h3.geo_to_cells(boundary_geojson_geom, res=resolution))
  
  hex_data = []
  for hex_id in hexagons:
    lat, lon = h3.cell_to_latlng(hex_id)
    centroid = Point(lon, lat)
    if boundary_polygon.contains(centroid):
        # Build polygon from the hexagons
        boundary_coords = h3.cell_to_boundary(hex_id)
        hex_polygon = Polygon([(lon, lat) for lat, lon in boundary_coords])
        hex_data.append({
            "hex_id": hex_id,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "geometry": hex_polygon
        })
        
  gdf_hex = gpd.GeoDataFrame(hex_data, crs="EPSG:4326")
  
  return gdf_hex


def poiAssignment(gdf_poi: gpd.GeoDataFrame, h3_resolution : int)-> gpd.GeoDataFrame:
  """
  Assigns each POI to the H3 hexagon that contains it.

  The input GeoDataFrame is reprojected to EPSG:4326 if needed.
  POIs that cannot be assigned to a hexagon (e.g. invalid geometries)
  are dropped from the output.

  Parameters
  ----------
  gdf_poi : gpd.GeoDataFrame
      GeoDataFrame of points of interest. Geometries must be Point type.
  h3_resolution : int
      H3 resolution level used to identify the containing hexagon.

  Returns
  -------
  gpd.GeoDataFrame
      Input GeoDataFrame with an additional column ``hex_id`` containing
      the H3 index of the hexagon each POI belongs to.
      Rows with no valid assignment are removed.
  """
  
  if gdf_poi.crs.to_epsg() != 4326:
    gdf_poi = gdf_poi.to_crs(epsg=4326)

  def assign_hex(row):
    try:
        return h3.latlng_to_cell(row.geometry.y, row.geometry.x, res=h3_resolution)
    except Exception:
        return None
      
  gdf_poi["hex_id"] = gdf_poi.apply(assign_hex, axis=1)
  gdf_poi = gdf_poi.dropna(subset=["hex_id"])
  
  return gdf_poi


def main(
  ctx: dict         
):
  """
  Builds the H3 grid, assigns POIs to hexagons, and saves all outputs.

  Workflow:
    1. Load the study area boundary and POI dataset.
    2. Generate the H3 hexagonal grid clipped to the boundary.
    3. Assign each POI to its containing hexagon.
    4. Aggregate POI counts per hexagon and merge onto the grid.
    5. Save grid, POI assignments and centroids to TRANSPORT_EFFICIENCY_PATH.

  Parameters
  ----------
  ctx : dict
      Runtime context. Expected keys:
        - ``h3_resolution`` (int): H3 resolution level for the grid.

  Outputs
  -------
  poi_h3_assigned.geojson  : POIs with hex_id column.
  hex_grid.geojson         : H3 grid with poi_count column.
  hex_poi_stats.geojson    : Alias of hex_grid.geojson (used downstream).
  hex_centroids.csv        : Hexagon centroids with poi_count.
  """

  logging.basicConfig(level=logging_level)
  with timed():
    logging.info(f"Reading relevant input data")
    zoi_shp = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER).to_crs(CRS_LATLONG)
    gdf_poi =  pd.read_parquet(PathRegistry.POI_AMENITIES)
    geometry = gpd.GeoSeries.from_wkt(gdf_poi['geometry'])
    gdf_poi = gpd.GeoDataFrame(gdf_poi, geometry=geometry, crs=CRS_LATLONG)

    logging.info(f"Creating H3 grid with resolution {ctx['h3_resolution']}")
    gdf_hex = create_h3_grid(zoi_shp, ctx['h3_resolution'])

    logging.info(f"Assigning POIs to hexagons")
    gdf_poi_assigned = poiAssignment(gdf_poi, ctx['h3_resolution'])

    # Save the assigned POIs to a file
    PathRegistry.TRANSPORT_EFFICIENCY.mkdir(parents=True, exist_ok=True) # NB Non so se serve o se è già stato fatto in un passaggio precedente, ma per sicurezza creo la cartella di output
    gdf_poi_assigned.to_parquet(PathRegistry.POI_H3_ASSIGNED)

    # POIs count per hexagon
    poi_per_hex = gdf_poi_assigned.groupby("hex_id").size().reset_index(name="poi_count")
    gdf_hex = gdf_hex.merge(poi_per_hex, on="hex_id", how="left")
    gdf_hex["poi_count"] = gdf_hex["poi_count"].fillna(0).astype(int)


    # Save POI Statistics per hexagon
    logging.info(f"Saving hex grid and POI stats to file")
    hex_stats_output_file = PathRegistry.HEX_POI_STAT
    gdf_hex.to_parquet(hex_stats_output_file)

    ## Save also hexagons centroids
    #centroids_df = gdf_hex[["hex_id", "centroid_lat", "centroid_lon", "poi_count"]].copy()
    #centroids_df.to_parquet(PathRegistry.HEX_CENTROIDS)

  
if __name__ == "__main__":
  
  main(ctx=context)
  