"""
Extract table data from zona500.pdf (Tper - Principali Destinazioni)
Requirements: pdfplumber, pandas
"""

import pdfplumber
import pandas as pd
from io import BytesIO
import requests


from bdt_mobility.utils.config import logging, logging_level


# ---------------------------------------------------------------------------
# Table structure
# ---------------------------------------------------------------------------
# The PDF has 3 pages, each containing 5 identical horizontal blocks.
# Each block has 3 columns:
#   - LOCALITY    (width ~95.9 pt)
#   - ZONE        (width ~23.1 pt)
#   - FARE        (width ~34.7 pt — contains the number of zones or a special
#                  fare code such as "urb.BO", "BLQ", "5+int.FE")
#
# Note: although the column header reads "TARIFFA  N° zone" (two lines),
# FARE and N° zones share the same physical cell in the PDF.
#
# X coordinates (in PDF points) for the 5 blocks:
BLOCKS = [
    (25,  121, 144, 179),   # block 1
    (185, 280, 304, 338),   # block 2
    (344, 440, 463, 498),   # block 3
    (504, 600, 623, 657),   # block 4
    (663, 759, 782, 817),   # block 5
]
# Each tuple: (x_start_locality, x_start_zone, x_start_fare, x_end_block)

# RGB color of data cells (light blue — all data rows share this fill color)
DATA_COLOR = (0.8, 0.8, 1.0)

# Approximate height of each data row in PDF points
ROW_HEIGHT = 11

# The legend box starts at this vertical position (PDF points) on every page.
# All content at or below this threshold is excluded from extraction.
LEGEND_TOP = 546.0


def extract_page(page: "pdfplumber.page.Page") -> list[dict]:
    """Extract all data rows from a single PDF page."""

    words = page.extract_words()

    # Identify data rows: rectangles in the first column (x0 ≈ 25)
    # that have the DATA_COLOR fill and are above the legend area
    data_rects = [
        r for r in page.rects
        if abs(r["x0"] - 25) < 1
        and 6 < r["height"] < 12
        and r["non_stroking_color"] == DATA_COLOR
        and r["top"] < LEGEND_TOP          # exclude legend
    ]

    # Vertical position (top) of each data row
    row_tops = sorted(set(round(r["top"], 1) for r in data_rects))

    def get_text(top: float, x0: float, x1: float) -> str:
        """Return the concatenated text of all words within the given area."""
        bottom = top + ROW_HEIGHT
        matching = [
            w["text"] for w in words
            if x0 - 1 <= w["x0"] < x1 and top <= w["top"] < bottom
        ]
        return " ".join(matching).strip()

    rows = []
    for top in row_tops:
        for (x_loc, x_zone, x_fare, x_end) in BLOCKS:
            locality = get_text(top, x_loc,   x_zone)
            zone     = get_text(top, x_zone,  x_fare)
            fare     = get_text(top, x_fare,  x_end + 2)

            if locality and zone:  # skip empty padding rows and alphabet divider rows (A, B, C...)
                rows.append({
                    "locality": locality,
                    "zone":     zone,
                    "fare":     fare,
                })

    return rows


def extract_pdf(source: str | BytesIO) -> pd.DataFrame:
    """
    Open a PDF from a file path or an in-memory BytesIO buffer and return
    a DataFrame with columns:
        locality | zone | fare
 
    Parameters
    ----------
    source : str or io.BytesIO
        Either a local file path (str) or an in-memory buffer returned by
        download_pdf().
 
    'fare' may contain:
        - an integer string (e.g. "3")  → number of zones
        - a special fare code (e.g. "urb.BO", "BLQ", "5+int.FE")
        - an empty string               → data not present
    """
    all_rows = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            all_rows.extend(extract_page(page))
    return pd.DataFrame(all_rows)


def download_pdf(url: str) -> BytesIO:
    """
    Download a PDF from a URL and return it as an in-memory BytesIO object.
    The file is never written to disk.
 
    Parameters
    ----------
    url : str
        Direct URL to the PDF file.
 
    Returns
    -------
    io.BytesIO
        In-memory buffer containing the raw PDF bytes, ready to be passed
        to pdfplumber.open().
 
    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status code.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; tper-pdf-extractor/1.0)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return BytesIO(response.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging_level)

    PDF_URL  = "https://t.tper.it/zone/zona500.pdf"

    logging.info("Extracting PTS cost table")
    df = extract_pdf(download_pdf(PDF_URL))

    logging.info("Extraction complete")
    logging.debug(f"Rows extracted: {len(df)}")
    logging.debug(f"Unique fare values: {sorted(df['fare'].unique())}")
    logging.debug(df.to_string())