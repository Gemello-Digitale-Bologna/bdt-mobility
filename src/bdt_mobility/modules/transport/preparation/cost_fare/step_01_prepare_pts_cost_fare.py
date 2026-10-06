
from bdt_mobility.utils.config import logging, logging_level
from .pts_cost.prepare_pts_cost_geocoding import geocode
from .pts_cost.prepare_pts_cost_fare_table import extract_pdf, download_pdf
from .pts_cost.prepare_pts_cost_zone_table import scrape_zone_table


from bdt_mobility.utils.config import DATA_PATH

COST_FARE_PATH = DATA_PATH / 'preparation' / 'cost_fare'
COST_FARE_PATH.mkdir(parents=True, exist_ok=True)

# Map tarifs to cost-zones
dict_fare_cost ={
    "urb.BO": 2.3,
    "2": 2.9,
    "3": 3.7,
    "4": 4.5,
    "5": 5.4,
    "5+int.FE": 7.3,
    "6": 6.2,
    "7": 6.8,
    "8": 7.4,
    "BLQ": 12.8,
}


def main():
    logging.basicConfig(level=logging_level)

    # Loading data
    url_locality_fare = "https://t.tper.it/zone/zona500.pdf"
    df_pdf  = extract_pdf(download_pdf(url_locality_fare)).rename(columns={"zone": "zone_code"}).astype({"zone_code": int})

    url_locality_zone = "https://www.tper.it/biglietti-abbonamenti/distanze-zonali"
    df_html = scrape_zone_table(url=url_locality_zone).astype({"zone_code": int})

    # Geocoding
    geo_df_locality = geocode(df_pdf=df_pdf, df_html=df_html)
    
    # Map fare to cost zones
    geo_df_locality["cost"] = geo_df_locality["fare"].map(dict_fare_cost)

    # Saving
    geo_df_locality.to_parquet(COST_FARE_PATH / "zona500_geocoded_localities_costs.parquet")


if __name__ == "__main__":
    main()

# uv run python -m bdt_mobility.modules.transport.preparation.cost_fare.step_01_prepare_pts_cost_fare