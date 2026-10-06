"""
Two analyses on the geocoded locality dataset:

1. Max intra-zone distance
   For each zone, compute the maximum great-circle distance (km) between all
   pairs of geocoded localities in that zone. Saves a histogram as PDF.

2. Interactive map
   Plot all geocoded localities on a Folium map. Hovering over a marker shows
   the locality name and zone. Saves the map as an HTML file.

Input:  zona500_geocoded.csv
        columns: locality, zone, fare, lat, lon, geocoded, normalized_locality

Outputs:
    zone_max_distances.pdf   -- histogram of max intra-zone distances
    zone_map.html            -- interactive Folium map

Requirements: pandas, numpy, matplotlib, folium
    pip install pandas numpy matplotlib folium
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import folium
from itertools import combinations


from .prepare_pts_cost_geocoding import BOLOGNA_VIEWBOX

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMPUT_FILE   = "zona500_geocoded_localities_fares.parquet"
OUTPUT_PDF  = "zone_max_distances.pdf"

# Earth radius in kilometres
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance in kilometres between two points
    given as (latitude, longitude) in decimal degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def max_distance_in_zone(points: list[tuple[float, float]]) -> float:
    """
    Given a list of (lat, lon) tuples, return the maximum pairwise
    great-circle distance in kilometres.
    Returns 0.0 if fewer than 2 points are provided.
    """
    if len(points) < 2:
        return 0.0
    return max(
        haversine(lat1, lon1, lat2, lon2)
        for (lat1, lon1), (lat2, lon2) in combinations(points, 2)
    )


# ---------------------------------------------------------------------------
# Interactive map
# ---------------------------------------------------------------------------

OUTPUT_HTML = "zone_map.html"

# Assign a distinct colour to each unique zone using a fixed palette.
# Folium marker colours (named): red, blue, green, purple, orange, darkred,
# lightred, beige, darkblue, darkgreen, cadetblue, darkpurple, white, pink,
# lightblue, lightgreen, gray, black, lightgray.
_PALETTE = [
    "blue", "green", "red", "purple", "orange", "darkred", "darkblue",
    "darkgreen", "cadetblue", "darkpurple", "pink", "lightblue",
    "lightgreen", "gray", "black",
]


def build_map(df_geo: pd.DataFrame, output_html: str = OUTPUT_HTML) -> folium.Map:
    """
    Build an interactive Folium map of all geocoded localities.

    Each marker is coloured by zone. Hovering over a marker shows a tooltip
    with the locality name and zone number. Clicking shows a popup with the
    same information plus the fare value.
    The geocoding bounding box (BOLOGNA_VIEWBOX) is drawn as a dashed rectangle.

    Parameters
    ----------
    df_geo      : DataFrame filtered to geocoded rows only
                  (must have columns: locality, zone, fare, lat, lon)
    output_html : path where the HTML map is saved

    Returns
    -------
    folium.Map object (also saved to disk)
    """
    # Centre the map on the mean coordinates of all points
    center_lat = df_geo["lat"].mean()
    center_lon = df_geo["lon"].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                      tiles="OpenStreetMap")

    # Draw the geocoding bounding box as a dashed rectangle
    # BOLOGNA_VIEWBOX = ((south, west), (north, east))
    folium.Rectangle(
        bounds=list(BOLOGNA_VIEWBOX),
        color="#E85D04",
        weight=2,
        dash_array="8 4",
        fill=False,
        tooltip="Geocoding bounding box (Nominatim viewbox)",
    ).add_to(fmap)

    # Build a colour mapping: zone -> colour
    unique_zones = df_geo["fare"].unique()
    zone_colour  = {z: _PALETTE[i % len(_PALETTE)] for i, z in enumerate(unique_zones)}

    # Add one marker per locality
    for _, row in df_geo.iterrows():
        colour = zone_colour[row["fare"]]

        tooltip = f"<b>{row['locality']}</b><br>Zone: {row['zone_name']}</b><br>Fare:</b> {row['fare']}"
        popup   = folium.Popup(
            f"<b>Locality:</b> {row['locality']}<br>"
            f"<b>Zone:</b> {row['zone_name']}<br>"
            f"<b>Fare:</b> {row['fare']}",
            max_width=220,
        )

        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=tooltip,
            popup=popup,
            icon=folium.Icon(color=colour, icon="circle", prefix="fa"),
        ).add_to(fmap)

    fmap.save(output_html)
    print(f"Interactive map saved to: {output_html}  ({len(df_geo)} markers)")
    return fmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Load data
    df = pd.read_parquet(IMPUT_FILE)
    print(f"Total rows loaded: {len(df)}")

    # Keep only successfully geocoded rows
    df_geo = df[df["geocoded"] == True].copy()
    print(f"Geocoded rows:     {len(df_geo)}")

    # 2. Compute max distance per zone
    records = []
    for zone, group in df_geo.groupby("zone_name"):
        points = list(zip(group["lat"], group["lon"]))
        if len(points) < 2:
            continue                          # skip zones with a single point
        max_dist = max_distance_in_zone(points)
        records.append({
            "zone":        zone,
            "n_points":    len(points),
            "max_dist_km": round(max_dist, 3),
            "localities":  ", ".join(sorted(group["locality"].tolist())),
        })

    result_df = pd.DataFrame(records).sort_values("max_dist_km", ascending=False)
    print(f"\nZones with \u2265 2 geocoded points: {len(result_df)}")
    print(f"Max distance overall:  {result_df['max_dist_km'].max():.2f} km  "
          f"(zone {result_df.iloc[0]['zone']})")
    print(f"Mean max distance:     {result_df['max_dist_km'].mean():.2f} km")
    print(f"Median max distance:   {result_df['max_dist_km'].median():.2f} km")

    # Top 10 — wrap locality names for readability
    pd.set_option("display.max_colwidth", 80)
    print(f"\nTop 10 zones by max distance:")
    print(result_df.head(10)[["zone", "n_points", "max_dist_km", "localities"]].to_string(index=False))

    # 3. Plot histogram
    distances = result_df["max_dist_km"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        distances,
        bins=30,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
    )

    # Vertical lines for mean and median
    mean_val   = distances.mean()
    median_val = distances.median()
    ax.axvline(mean_val,   color="#DD4444", linestyle="--", linewidth=1.4,
               label=f"Mean   {mean_val:.1f} km")
    ax.axvline(median_val, color="#44AA44", linestyle=":",  linewidth=1.4,
               label=f"Median {median_val:.1f} km")

    ax.set_xlabel("Max intra-zone distance (km)", fontsize=12)
    ax.set_ylabel("Number of zones",              fontsize=12)
    ax.set_title("Distribution of maximum intra-zone distances\n"
                 "(each zone: max great-circle distance between geocoded localities)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, format="pdf")
    print(f"\nHistogram saved to: {OUTPUT_PDF}")
    plt.close(fig)

    # 4. Build interactive map
    build_map(df_geo)