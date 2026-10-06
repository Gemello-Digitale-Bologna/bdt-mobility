import time
from datetime import datetime

import pandas as pd
import geopandas as gpd
import json

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.modules.transport.extraction.transport_efficiency.parameters import context

MAIN_OUTPUTS = [
    PlatformOutput("poi_classified", PathRegistry.POI_MUNICIPALITY_TAXONOMY, "artifact"),
]
# ── Lookup table ──────────────────────────────────────────────
# Format: (osm_tag_key, osm_tag_value) -> (macro_categoria, categoria, sottocategoria)
# When osm_tag_value is None it matches any value for that key (wildcard).


# Priority order when a POI row has multiple tag columns populated
TAG_PRIORITY = [
    "amenity", "shop", "leisure", "healthcare",
    "historic", "tourism", "office", "craft", "emergency", "landuse"
]


def classify_poi(row: pd.Series, osm_to_municipal: dict) -> tuple:
  """
  Returns (macro_categoria, categoria, sottocategoria) for a POI row,
  or (None, None, None) if no match is found.

  Uses TAG_PRIORITY to resolve ambiguity when multiple tags are present.
  """
  for tag_key in TAG_PRIORITY:
    tag_val = row.get(tag_key)
    if pd.isna(tag_val) or tag_val in (True, "yes", ""):
      continue
    tag_val = str(tag_val).strip().lower()
    result  = osm_to_municipal.get((tag_key, tag_val))
    if result:
      return result
  return (None, None, None)


def main(
  ctx: dict
):
  logging.basicConfig(level=logging_level)
  start_time = time.time()
  start_dt = datetime.now()
    
  logging.info(f"Loading POIs")
  gdf_poi =  pd.read_parquet(PathRegistry.POI_AMENITIES)
  geometry = gpd.GeoSeries.from_wkt(gdf_poi['geometry'])
  gdf = gpd.GeoDataFrame(gdf_poi, geometry=geometry, crs=CRS_LATLONG)
  
  logging.info("Loading dictionary for OSM to municipal taxonomy mapping")
  with open(PathRegistry.OSM_TO_MUNICIPAL_TAXONOMY, "r") as f:
    osm_to_municipal_taxonoy = json.load(f)
    
  osm_to_municipal = {
    (item['osm_key'], item['osm_value']): (
      item['macro'],
      item['micro'],
      item['dettaglio']
    )
    for item in osm_to_municipal_taxonoy
  }
  
  # Apply classification
  logging.info("Classifying POIs against municipal taxonomy...")
  taxonomy = gdf.apply(classify_poi, axis=1, args=(osm_to_municipal,), result_type="expand")
  taxonomy.columns = ["macro_categoria", "categoria", "sottocategoria"]
  gdf = pd.concat([gdf, taxonomy], axis=1)

  # Coverage report
  matched   = gdf["sottocategoria"].notna().sum()
  unmatched = gdf["sottocategoria"].isna().sum()
  logging.info(f"Matched:   {matched:,} ({matched/len(gdf):.1%})")
  logging.info(f"Unmatched: {unmatched:,} ({unmatched/len(gdf):.1%})")

  if unmatched > 0:
    logging.info("Top unmatched tag combinations:")
    unmatched_df = gdf[gdf["sottocategoria"].isna()]
    for tag in TAG_PRIORITY:
      if tag in unmatched_df.columns:
        counts = (
          unmatched_df[tag]
          .dropna()
          .value_counts()
          .head(5)
        )
        if len(counts):
          logging.info(f"  [{tag}]\n{counts.to_string()}")

  # Save
  
  logging.info(f"Saving POIs dataframe with municipality taxonomy")
  gdf.to_parquet(PathRegistry.POI_MUNICIPALITY_TAXONOMY)

  end_time = time.time()
  logging.info(f"Starting time: {start_dt}; Ending time: {datetime.now()}; Duration: {end_time - start_time:.2f}")

if __name__ == "__main__":
  
    main(ctx = context)