"""
Geocode localities from zona500.pdf using Nominatim (OpenStreetMap).

Pipeline:
    1. Extract (locality, zone, fare) from the PDF via extract_pdf()
    2. Scrape (locality, zone_code, zone_name) from the TPER website
       via scrape_zone_table()
    3. Geocode each unique zone
    4. Geocode each unique locality with a 2-level fallback strategy:
           a. "<locality>, <zone_name>, Emilia Romagna, Italy"
           b. "<locality>, Emilia Romagna, Italy"          (if a fails)
    5. Postprocess the locality geocoding results:
           c. "<zone_name>, Emilia Romagna, Italy"         (if single-locality is far from zone)

Requirements: pdfplumber, pandas, geopy, requests, lxml
"""

import re
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy import distance
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


from .prepare_pts_cost_fare_table import extract_pdf, download_pdf
from .prepare_pts_cost_zone_table import scrape_zone_table
from bdt_mobility.utils.config import logging, logging_level

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Nominatim requires a unique user-agent string to identify your application
USER_AGENT = "zona500_geocoder"

# Geographic bias: bounding box around the Bologna metropolitan area
# (south-west, north-east)
BOLOGNA_VIEWBOX = ((44.067211, 10.659937), (44.9706, 12.301203))

# Seconds to wait between requests (Nominatim policy: max 1 request/second)
REQUEST_DELAY = 1.1


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

# Abbreviations to expand before querying Nominatim.
_ABBREVIATIONS = [
    (r"Z\.I\.", "zona industriale "),   # trailing space handles attached words
    (r"BV\.",   "bivio "),
]

# Suffixes to remove entirely (no meaningful expansion for Nominatim).
_SUFFIXES_TO_REMOVE = [
    r"\+intersc\.bigl\.",
    r"\+intersc\.abb\.",
]

# Manual overrides for names that Nominatim cannot resolve even after
# abbreviation expansion. Keys must be in uppercase, post-normalisation
# form (i.e. after parentheses have been stripped).
_MANUAL_FIXES = {
    "INTERPORTO":              "INTERPORTO BOLOGNA",
    "AEROPORTO G.MARCONI 54":  "AEROPORTO GUGLIELMO MARCONI",
    "AEROPORTO G.MARCONI BLQ": "AEROPORTO GUGLIELMO MARCONI",
}

# Manual overrides for zone names scraped from the HTML table.
# Keys must be in uppercase. Add entries here whenever a zone name
# is abbreviated or ambiguous for Nominatim.
_ZONE_NAME_FIXES = {
    "S.BENEDETTO V.S.": "SAN BENEDETTO VAL DI SAMBRO",
    "S.BENEDETTO V.S. STAZIONE": "SAN BENEDETTO VAL DI SAMBRO STAZIONE",
    "CORTICELLA": "CORTICELLA BOLOGNA",
    "POGGIORENATICO": "POGGIO RENATICO",
    "POGGIO" : "POGGIO PICCOLO",
    "GRANAROLO": "GRANAROLO DELL'EMILIA",
    "MASSALOMBARDA": "MASSA LOMBARDA",
    "BOTTEGHINO": "BOTTEGHINO DI ZOCCA",
    "SAN NICOLO'": "SAN NICOLO' FERRARA",
    "VILLANOVA": "VILLANOVA DI CASTENASO",
}


def normalize_zone_name(name: str) -> str:
    """Normalise a zone name from the HTML table before using it in a Nominatim query.

    Currently applies manual fixes only; extend _ZONE_NAME_FIXES as needed.

    Examples:
        'S.BENEDETTO V.S.'  -> 'SAN BENEDETTO VAL DI SAMBRO'
        'BOLOGNA'           -> 'BOLOGNA'
    """
    if not name or not isinstance(name, str):
        return name
    return _ZONE_NAME_FIXES.get(name.upper().strip(), name)


def normalize_locality(name: str) -> str:
    """
    Expand abbreviations to their full form and remove parentheses,
    so that Nominatim receives a clean, readable place name.

    Examples:
        'CADRIANO BV. (Via Ferrarese)'  -> 'CADRIANO BIVIO VIA FERRARESE'
        'CASTEL MAGGIORE Z.I.SALICETO'  -> 'CASTEL MAGGIORE ZONA INDUSTRIALE SALICETO'
        'ARGELATO Z.I.'                 -> 'ARGELATO ZONA INDUSTRIALE'
        'BOLOGNA'                       -> 'BOLOGNA'
    """
    result = name
    # remove +intersc.* suffixes before anything else
    for pattern in _SUFFIXES_TO_REMOVE:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # expand abbreviations
    for pattern, expansion in _ABBREVIATIONS:
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    # remove parentheses and collapse spaces
    result = re.sub(r"[()]", "", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    # apply manual fixes after all other normalisation is done
    result = _MANUAL_FIXES.get(result.upper(), result)
    return result


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _nominatim_query(
    geolocator: Nominatim,
    query: str,
) -> tuple[float | None, float | None]:
    """
    Run a single Nominatim query and return (lat, lon) or (None, None).
    Handles timeout with one automatic retry.
    """
    try:
        result = geolocator.geocode(
            query,
            viewbox=BOLOGNA_VIEWBOX,
            bounded=True,
            language="it",
            timeout=10,
        )
        if result:
            return result.latitude, result.longitude
        return None, None

    except GeocoderTimedOut:
        logging.critical(f"    [timeout] retrying: {query!r}")
        time.sleep(2)
        return _nominatim_query(geolocator, query)

    except GeocoderServiceError as e:
        logging.critical(f"    [error] {query!r}: {e}")
        return None, None


def geocode_locality(
    geolocator: Nominatim,
    locality: str,
    zone_name: str | None,
) -> tuple[float | None, float | None, str]:
    """
    Geocode a single locality with a 3-level fallback strategy.

    Level 1: "<locality>, <zone_name>, Italy"
    Level 2: "<locality>,  Italy"

    Returns (latitude, longitude, query_used).
    If all levels fail, returns (None, None, last_query_tried).
    """

    has_zone = zone_name and pd.notna(zone_name)

    # Level 1 -- locality + zone_name
    if has_zone:
        q1 = f"{locality}, {zone_name}, Italy"
        lat, lon = _nominatim_query(geolocator, q1)
        if lat is not None:
            return lat, lon, q1
        time.sleep(REQUEST_DELAY)

    # Level 2 -- locality only
    q3 = f"{locality}, Italy"
    lat, lon = _nominatim_query(geolocator, q3)
    if lat is not None:
        return lat, lon, q3
    
    return None, None, q3   # all levels failed


def geocode_zone_name(
    geolocator: Nominatim,
    zone_name: str
) -> tuple[float | None, float | None, str]:
    """
    Geocode a single zone name (without locality) with a single query.

    Returns (latitude, longitude, query_used).
    If all levels fail, returns (None, None, last_query_tried).
    """
    q3 = f"{zone_name}, Italy"
    lat, lon = _nominatim_query(geolocator, q3)
    if lat is not None:
        return lat, lon, q3

    return None, None, q3   # all levels failed


def geocode_all_zone_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode all unique zones in the DataFrame.

    Expects columns: zone_name
    Returns a DataFrame with columns:
        zone_name | query_name | lat | lon | geocoded
    """
    # Deduplicate: geocode each unique zone once

    geolocator = Nominatim(user_agent=USER_AGENT)
    results = []

    for i, row in df.iterrows():

        zone_name = row["zone_name"]
        zone_name = normalize_zone_name(zone_name) 

        logging.debug(f"[{i+1}/{len(df)}] {zone_name!r}")
        lat, lon, query_used = geocode_zone_name(geolocator, zone_name)

        results.append({
            "zone_name":  row["zone_name"],
            "zone_code":  row["zone_code"],
            "query_name": query_used,
            "lat":        lat,
            "lon":        lon,
            "geocoded":   lat is not None,
        })
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(results).sort_values("zone_name").reset_index(drop=True)


def geocode_all_localities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode all unique localities in the merged DataFrame.

    Expects columns: locality, zone_name (may be NaN).
    Returns a DataFrame with columns:
        locality | zone_name | zone_code | query_locality_name | lat | lon | geocoded | far_from_zone | dist_km_from_zone
    """
    
    geolocator = Nominatim(user_agent=USER_AGENT)
    results = []

    for i, row in df.iterrows():

        locality = row["locality"]
        locality = normalize_locality(locality)
        zone_name = row["zone_name"]
        zone_name = normalize_zone_name(zone_name) if pd.notna(zone_name) else None
        has_zone = zone_name and pd.notna(zone_name)
    
        logging.debug(f"[{i+1}/{len(df)}] {locality!r} (zone: {zone_name!r})")
        
        lat, lon, query_used = geocode_locality(geolocator, locality, zone_name)
        modified = False
        if lat is None and has_zone:
            lat, lon, query_used = row["lat_zone"], row["lon_zone"], row["query_name_zone"]
            modified = True

        results.append({
            "locality":   row["locality"],
            "zone_name":  row["zone_name"],
            "zone_code":  row["zone_code"],
            "query_name": query_used,
            "lat":        lat,
            "lon":        lon,
            "geocoded":   lat is not None,
            "modified":   modified
        })
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(results).sort_values(by=["zone_name","locality"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def post_geocoding_localities(localities: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
        
    def compute_distance(row):
        if pd.notna(row["lat_zone"]) and pd.notna(row["lat_loc"]):
            return distance.distance((row["lat_loc"], row["lon_loc"]), (row["lat_zone"], row["lon_zone"])).km
        else:
            return None

    def adjust_far_locality(row):
        if row["far_locality"] and not row["far_zone"]:
            row["lat_loc"] = row["lat_zone"]
            row["lon_loc"] = row["lon_zone"]
            row["query_name_loc"] = row["query_name_zone"]
            row["geocoded_loc"] = True
            row["modified"] = True
        return row

    df = pd.merge(
        localities.rename(columns={"lat": "lat_loc", "lon": "lon_loc", "query_name": "query_name_loc", "geocoded": "geocoded_loc"}), 
        zones.rename(columns={"lat": "lat_zone", "lon": "lon_zone", "query_name": "query_name_zone", "geocoded": "geocoded_zone"}), 
        on=["zone_name", "zone_code"],
        how="left"
    )

    # Now, create column "far_locality" saying if the locality is more than 20 km from the zone
    df["distance"] = df.apply(lambda row: compute_distance(row), axis=1)
    df["far_locality"] = df["distance"].apply(lambda x: x > 20 if pd.notna(x) else None)

    # Now, create the column "far_zone" saying if the majority of localities in the zone are set as far localities
    zone_far_stats = df.groupby(["zone_code", "zone_name"])["far_locality"].agg(["mean"]).reset_index()
    zone_far_stats["far_zone"] = zone_far_stats["mean"] > 0.5
    df = df.merge(zone_far_stats.drop(columns=["mean"]), how="left")

    # For the cases of far_locality==True and far_zone==True, do nothing
    # For the cases of far_locality==True and far_zone==False, set the lat/lon of the locality to the lat/lon of the zone
    df = df.apply(lambda row: adjust_far_locality(row), axis=1)
    return (
        df[["locality", "zone_name", "zone_code", "lat_loc", "lon_loc", "query_name_loc", "geocoded_loc", "modified"]]
        .rename(columns={"lat_loc":"lat", "lon_loc": "lon", "query_name_loc":"query_name", "geocoded_loc":"geocoded"})
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def geocode(df_pdf: pd.DataFrame, df_html: pd.DataFrame) -> pd.DataFrame:
    
    # Load and preprocess data
    logging.info("Loading and preparing the data for zone geocoding")

    df_html = df_html.assign(
        _key1=df_html["locality"].apply(normalize_locality),
        _key2=df_html["zone_code"].apply(normalize_zone_name)
    )
    df_pdf = df_pdf.assign(
        _key1=df_pdf["locality"].apply(normalize_locality),
        _key2=df_pdf["zone_code"].apply(normalize_zone_name)
    )
    df = pd.merge(df_pdf, df_html[["_key1", "_key2", "zone_name"]], on=["_key1", "_key2"], how="left")
    df = df.drop(columns=["_key1", "_key2"])
    df = df.sort_values(by=["zone_name", "locality"]).reset_index(drop=True)
    
    # Geocode the zones
    logging.info("Geocoding zone names (without locality)")
    df_red = df[["zone_name", "zone_code"]].drop_duplicates().reset_index(drop=True).dropna(subset=["zone_name"])
    geo_df_zone = geocode_all_zone_names(df_red).astype({"zone_code": int})

    # Checks
    n_found  = geo_df_zone["geocoded"].sum()
    n_missed = (~geo_df_zone["geocoded"]).sum()
    logging.info(f"Results: {n_found} found, {n_missed} not found out of {len(geo_df_zone)}")

    if n_missed:
        logging.info("Not geocoded:")
        logging.info(geo_df_zone[~geo_df_zone["geocoded"]]["zone_name"].to_string())

    # Load and preprocess data
    logging.info("Loading and preparing the data for locality geocoding")

    geo_df_zone = geo_df_zone.rename(columns={"query_name": "query_name_zone", "lat": "lat_zone", "lon": "lon_zone", "geocoded": "geocoded_zone"})
    df = df.merge(geo_df_zone, on=["zone_name", "zone_code"], how="left")

    # Geocode the localities
    logging.info("Geocoding locality names")
    logging.debug(f"Total localities to geocode: {len(df)}")
    geo_df_locality = geocode_all_localities(df).astype({"zone_code": int})
    logging.debug(f"Total localities geocoded: {len(geo_df_locality)}")
    geo_df_locality = post_geocoding_localities(localities=geo_df_locality, zones=geo_df_zone)
    logging.debug(f"Total localities after postprocessing: {len(geo_df_locality)}")

    # Checks
    n_found  = geo_df_locality["geocoded"].sum()
    n_missed = (~geo_df_locality["geocoded"]).sum()
    logging.info(f"Results: {n_found} found, {n_missed} not found out of {len(geo_df_locality)}")

    if n_missed:
        logging.info("Not geocoded:")
        logging.info(geo_df_locality[~geo_df_locality["geocoded"]]["zone_name"].to_string())

    logging.info("Finalizing and returning results")
    geo_df_locality = geo_df_locality.merge(df[["locality", "zone_name", "zone_code", "fare"]], on=["locality", "zone_name", "zone_code"], how="left")
    return geo_df_locality


def main():
    logging.basicConfig(level=logging_level)

    # Loading data
    url_locality_fare = "https://t.tper.it/zone/zona500.pdf"
    df_pdf = extract_pdf(download_pdf(url_locality_fare)).rename(columns={"zone": "zone_code"}).astype({"zone_code": int})

    url_locality_zone = "https://www.tper.it/biglietti-abbonamenti/distanze-zonali"
    df_html = scrape_zone_table(url=url_locality_zone).astype({"zone_code": int})

    # Executing main
    geo_df_locality = geocode(df_pdf=df_pdf, df_html=df_html)
    
    # Saving
    geo_df_locality.to_parquet("zona500_geocoded_localities_fares.parquet")
    

if __name__ == "__main__":
    main()