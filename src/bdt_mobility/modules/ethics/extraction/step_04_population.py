"""Resident population per statistical area, by sex, citizenship and age band.

Weights the normalization reference values in step_09.
"""
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("population_by_area", PathRegistry.POPULATION_BY_AREA, "table"),
]

GROUP_COLUMNS = ["codice_area_statistica", "anno", "sesso", "cittadinanza", "eta_grandi"]


def prepare_population(df):
    """Collapse the single-year-of-age detail."""
    return df.groupby(GROUP_COLUMNS, as_index=False)["residenti"].sum()


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Loading the resident population")
        df = data_reader.open_data(
            PathRegistry.FRAGILITIES_RESIDENT_POPULATION,
            min_year=fragilities["temporal"]["min_year"],
            max_year=fragilities["temporal"]["max_year"],
        )

        df = prepare_population(df)

        df.to_parquet(PathRegistry.POPULATION_BY_AREA)
        logging.info(f"Wrote {len(df)} rows of population by area/sex/citizenship")


if __name__ == "__main__":
    main()
