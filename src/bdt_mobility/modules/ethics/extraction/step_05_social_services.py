"""Social-service users over their denominator, per category.

Published by quartiere and fanned out to the statistical areas inside it. Elderly is a
share of residents 65+, minors a share of families with a minor.
"""
import pandas as pd

from bdt_mobility.utils import data_reader
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities
from bdt_mobility.modules.ethics.extraction.common import map_to_as
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("social_services_by_area", PathRegistry.SOCIAL_SERVICES_BY_AREA, "table"),
]

ELDERLY_AGE_GROUP = "65 e oltre"

# Rounded as step_01_fragilities_preprocess rounded them.
ELDERLY_DECIMALS = 2
MINORS_DECIMALS = 4


def elderly_share(df_services: pd.DataFrame, df_population: pd.DataFrame) -> pd.DataFrame:
    """Social-service users aged 65+ over residents aged 65+, by quartiere."""
    df_over65 = df_population[df_population["eta_grandi"] == ELDERLY_AGE_GROUP]
    df_over65 = df_over65.groupby(["anno", "quartiere"], as_index=False)["residenti"].sum()

    df = pd.merge(df_services[["anno", "quartiere", "anziani"]], df_over65, on=["anno", "quartiere"])
    df["value"] = round(df["anziani"] / df["residenti"], ELDERLY_DECIMALS)
    df["category"] = "elderly"
    return df[["anno", "quartiere", "category", "value"]]


def minors_share(df_services: pd.DataFrame, df_families: pd.DataFrame) -> pd.DataFrame:
    """Families served over families with a minor, by quartiere."""
    df_totals = df_families.groupby(["anno", "quartiere"], as_index=False)["numero_famiglie"].sum()

    df = pd.merge(df_services[["anno", "quartiere", "famiglie_e_minori"]], df_totals, on=["anno", "quartiere"])
    df["value"] = round(df["famiglie_e_minori"] / df["numero_famiglie"], MINORS_DECIMALS)
    df["category"] = "minors"
    return df[["anno", "quartiere", "category", "value"]]


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        min_year = fragilities["temporal"]["min_year"]
        max_year = fragilities["temporal"]["max_year"]

        logging.info("Loading social services and their denominators")
        df_services = data_reader.open_data(PathRegistry.FRAGILITIES_SOCIAL_SERVICES, min_year=min_year, max_year=max_year)
        df_population = data_reader.open_data(PathRegistry.FRAGILITIES_RESIDENT_POPULATION, min_year=min_year, max_year=max_year)
        df_families = data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_MINORS_18, min_year=min_year, max_year=max_year)
        df_areas = data_reader.statistical_areas(PathRegistry.FRAGILITIES_STATISTICAL_AREAS)

        logging.info("Computing shares per category")
        df_shares = pd.concat(
            [elderly_share(df_services, df_population), minors_share(df_services, df_families)],
            ignore_index=True,
        )

        logging.info("Mapping quartiere values onto statistical areas")
        df_out = map_to_as(df_shares, df_areas, target_columns=["category", "value"], geo_column="quartiere")

        df_out.to_parquet(PathRegistry.SOCIAL_SERVICES_BY_AREA)
        logging.info(f"Wrote {len(df_out)} rows of social-service shares")


if __name__ == "__main__":
    main()
