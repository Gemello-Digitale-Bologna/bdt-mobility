"""
Scrape the locality-to-zone table from the TPER website and return a DataFrame.

Source: https://www.tper.it/biglietti-abbonamenti/distanze-zonali

The page contains an HTML table with columns:
    Località | Codice zona | Nome zona | (link column — discarded)

Requirements: pandas, requests, lxml
    pip install pandas requests lxml

Note: lxml is used as the HTML parser backend for pandas.read_html().
      If unavailable, install html5lib as an alternative:
          pip install html5lib
      and replace flavor="lxml" with flavor="html5lib" below.
"""

import pandas as pd
import requests
from io import StringIO


from bdt_mobility.utils.config import logging, logging_level


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Column names as they appear in the raw HTML table
RAW_LOCALITY_COL  = "Località"
RAW_ZONE_CODE_COL = "Codice zona"
RAW_ZONE_NAME_COL = "Nome zona"

# Renamed output columns (English)
OUT_LOCALITY_COL  = "locality"
OUT_ZONE_CODE_COL = "zone_code"
OUT_ZONE_NAME_COL = "zone_name"


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_zone_table(url: str) -> pd.DataFrame:
    """
    Download the TPER zone page and extract the locality table.

    Returns a DataFrame with columns:
        locality  | zone_code | zone_name

    Raises
    ------
    ValueError  if no table is found on the page
    requests.HTTPError  if the HTTP request fails
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; tper-zone-scraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # pandas.read_html parses all <table> elements on the page
    tables = pd.read_html(StringIO(response.text), flavor="lxml")

    if not tables:
        raise ValueError(f"No HTML tables found at {url}")

    # The locality table is the first (and only) data table on the page
    df = tables[0]

    # Drop the last column (it contains the PDF link text, not useful data)
    df = df.iloc[:, :3]
    df.columns = [OUT_LOCALITY_COL, OUT_ZONE_CODE_COL, OUT_ZONE_NAME_COL]

    # Ensure zone_code is stored as int
    df[OUT_ZONE_CODE_COL] = df[OUT_ZONE_CODE_COL].astype(str)

    # Strip leading/trailing whitespace from text columns
    for col in [OUT_LOCALITY_COL, OUT_ZONE_NAME_COL]:
        df[col] = df[col].str.strip()

    return df.sort_values(by=[OUT_ZONE_NAME_COL, OUT_LOCALITY_COL]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging_level)

    logging.info("Scraping the zone table")
    URL = "https://www.tper.it/biglietti-abbonamenti/distanze-zonali"
    df = scrape_zone_table(url=URL)

    logging.info("Scraping complete")
    logging.debug(f"Rows scraped:        {len(df)}")
    logging.debug(f"Unique zone codes:   {df['zone_code'].nunique()}")
    logging.debug(f"Unique zone names:   {df['zone_name'].nunique()}")
    logging.debug(f"First 10 rows:\n{df.head(10).to_string(index=False)}")
    logging.debug(f"zone_code sample: {sorted(df['zone_code'].unique())[:10]}")