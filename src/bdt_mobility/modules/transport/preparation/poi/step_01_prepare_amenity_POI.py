import pandas as pd
import geopandas as gpd
from shapely import wkb
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


POI_TAGS = {
    # Servizi e strutture
    "amenity": True,          # ristoranti, bar, scuole, ospedali, banche, ecc.
    "shop": True,             # negozi di ogni tipo
    "tourism": True,          # hotel, musei, attrazioni, ecc.
    "leisure": True,          # parchi, palestre, piscine, ecc.
    "healthcare": True,       # strutture sanitarie
    "office": True,           # uffici pubblici e privati
    "public_transport": True, # fermate bus, metro, treni
    "historic": True,         # monumenti, edifici storici
    "sport": True,            # impianti sportivi
    "craft": True,            # artigiani
    "emergency": True,        # pronto soccorso, polizia, ecc.
}

from bdt_mobility.preparation.config import YEAR
DATE = f'{YEAR}-01-01T00:00:00Z'

def main():
  print(PROJECT)
  logging.basicConfig(level=logging_level)
  start_time = time.time()
  start_dt = datetime.now()
  
  logging.info("Donwloa ding POI data from OSM")
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

  logging.info("osmnx setting")
  # Change the osmnx setting to get historical data
  __default_settings = ox.settings.overpass_settings
  __default_url = ox.settings.overpass_url
  __default_timeout = ox.settings.requests_timeout
  ox.settings.overpass_url = "https://overpass-api.de/api"
  ox.settings.overpass_settings = '[out:json][timeout:{timeout}][date:"' + DATE + '"]{maxsize}'
  ox.settings.requests_timeout = 1200
  
  logging.info("Downloading data per each area statistica")
  # Iterating over statistical areas
  gdfs = []
  for _, row in gdf_areas.iterrows():
    name = row['codice_area_statistica']
    logging.info(f"Downloading data per area statistica {name}")
    geom = row.geometry
    try:
      gdf_tmp = ox.features_from_polygon(geom, tags = POI_TAGS)
      logging.debug(f"Downloaded data per area statistica {name}")
      gdf_tmp['codice_area_statistica'] = name
      gdfs.append(gdf_tmp)
    except Exception as e:
      logging.info(f"Errore nel download per {row['area_statistica']}: {e}")
      
  gdf_poi = pd.concat(gdfs).reset_index().drop_duplicates(subset='id')
  logging.info(f"Downloaded {len(gdf_poi):,} row elements from OSM")

  # Saving
  geometry_wkt = gdf_poi['geometry'].to_wkt()
  gdf_poi = pd.DataFrame(gdf_poi.drop(columns='geometry'))
  gdf_poi['geometry'] = geometry_wkt
  gdf_poi.to_parquet(POI_PATH / f"poi_bologna_{YEAR}.parquet", index=False)
  # dh.log_dataitem(project=PROJECT, name=f"poi_bologna_{YEAR}", data=gdf_poi, kind="table")
  
  end_time = time.time()
  logging.info(f"Starting time: {start_dt}; Ending time: {datetime.now()}; Duration: {end_time - start_time:.2f}")

  # Change back the osmnx setting to default
  ox.settings.overpass_url = __default_url
  ox.settings.overpass_settings = __default_settings
  ox.settings.requests_timeout = __default_timeout
  
if __name__ == "__main__":
  main()