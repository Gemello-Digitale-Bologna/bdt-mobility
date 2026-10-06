"""Median and mean declared income per statistical area, by citizenship, sex and age band.

Built by modules/ethics/preparation/step_01_redditi_mediani and downloaded back by
step_00. Not read by any indicator yet.
"""
import pandas as pd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("redditi_by_area", PathRegistry.REDDITI_BY_AREA, "table"),
]

# The full citizenship x sex x age-band grid is kept.
COLUMNS = [
    "codice_area_statistica",
    "anno",
    "cittadinanza",
    "sesso",
    "fascia_anni",
    "reddito_mediano",
    "reddito_medio",
    "contribuenti",
    # Which territorial level and demographic category the value fell back to.
    "fonte_reddito_mediano",
    "livello_fallback_categoria",
]


def prepare_redditi(df: pd.DataFrame, min_year: str, max_year: str) -> pd.DataFrame:
    df = df[df["anno"].between(int(min_year), int(max_year))].copy()
    df["anno"] = df["anno"].astype(str)
    return df[COLUMNS].reset_index(drop=True)


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Loading median incomes")
        df = pd.read_parquet(PathRegistry.FRAGILITIES_REDDITI)

        df = prepare_redditi(
            df,
            min_year=fragilities["temporal"]["min_year"],
            max_year=fragilities["temporal"]["max_year"],
        )

        df.to_parquet(PathRegistry.REDDITI_BY_AREA)
        logging.info(f"Wrote {len(df)} rows of median income by area/citizenship/sex/age")


if __name__ == "__main__":
    main()
