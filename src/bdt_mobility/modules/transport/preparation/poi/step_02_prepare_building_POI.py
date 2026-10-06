import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.ops import unary_union
import osmnx as ox
import time
from datetime import datetime
import digitalhub as dh

from bdt_mobility.utils.constants import CRS_LATLONG
from bdt_mobility.utils.config import logging, logging_level

from bdt_mobility.preparation.config import PROJECT_NAME as PROJECT
from bdt_mobility.utils.config import DATA_PATH
POI_PATH = DATA_PATH / 'preparation' / 'poi_amenities'
POI_PATH.mkdir(parents=True, exist_ok=True)

from bdt_mobility.preparation.config import YEAR
DATE = f'{YEAR}-01-01T00:00:00Z'

def main():
  print(PROJECT)
  logging.basicConfig(level=logging_level)
  start_time = time.time()
  start_dt = datetime.now()
  
  logging.info("Donwloading buildings data from OSM")
  # First, load the shape of the aree statistiche
  logging.info("Loading the aree statistiche")
  aree_statistiche_df = dh.get_dataitem("aree_statistiche", PROJECT).as_df()

  logging.info("Preparing the aree statistiche")
  # Drop centroid column
  aree_statistiche_df.drop(columns=['geo_point_2d'], inplace=True)
  
  # Convert WKB to Shapely geometries and create GeoDataFrame
  aree_statistiche_df["geometry"] = (
    aree_statistiche_df["geo_shape"]
    .apply(lambda x: wkb.loads(x) if x is not None else None)
    .drop(columns=['geo_shape'])
  )
  gdf_areas = gpd.GeoDataFrame(
    aree_statistiche_df,
    geometry='geometry',
    crs=CRS_LATLONG
  )

  # Compute the union of all statistical areas to use as OSM query boundary
  boundary = unary_union(gdf_areas.geometry)

  logging.info("osmnx setting")
  # Change the osmnx setting to get historical data
  __default_settings = ox.settings.overpass_settings
  __default_url = ox.settings.overpass_url
  __default_timeout = ox.settings.requests_timeout
  ox.settings.overpass_url = "https://overpass-api.de/api"
  ox.settings.overpass_settings = '[out:json][timeout:{timeout}][date:"' + DATE + '"]{maxsize}'
  ox.settings.requests_timeout = 1200
  
  logging.info("Downloading data")
  try:
    gdf_buildings = ox.features_from_polygon(boundary, tags={"building": True})

    # Keep only polygon geometries
    gdf_buildings = gdf_buildings[
      gdf_buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].reset_index()

    # Saving
    geometry_wkt = gdf_buildings['geometry'].to_wkt()
    gdf_buildings = pd.DataFrame(gdf_buildings.drop(columns='geometry'))
    gdf_buildings['geometry'] = geometry_wkt
    gdf_buildings.to_parquet(POI_PATH / f"buildings_bologna_{YEAR}.parquet", index=False)
  # dh.log_dataitem(project=PROJECT, name=f"buildings_bologna_{YEAR}", data=gdf_buildings, kind="table")

  except Exception as e:
    gdf_buildings = None
    print(f"   OSM download failed: {e}")
    print("   → Fallback: uniform distribution by geographic area")

  
  end_time = time.time()
  logging.info(f"Starting time: {start_dt}; Ending time: {datetime.now()}; Duration: {end_time - start_time:.2f}")

  # Change back the osmnx setting to default
  ox.settings.overpass_url = __default_url
  ox.settings.overpass_settings = __default_settings
  ox.settings.requests_timeout = __default_timeout
  
if __name__ == "__main__":
  main()