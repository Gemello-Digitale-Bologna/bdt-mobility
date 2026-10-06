"""Fragility index per statistical area, as published.

Unit fixes and the percent-to-ratio conversion only.
"""
from bdt_mobility.utils import data_reader
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities
from bdt_mobility.modules.ethics.extraction.common import convert_percent_to_ratio
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("fragility_index_by_area", PathRegistry.FRAGILITY_INDEX_BY_AREA, "table"),
]

# Published as 0-100, consumed as 0-1 ratios. The other columns are index numbers or counts.
PERCENT_COLUMNS = (
    "ultra80",
    "soli_65",
    "perfragsan",
    "minori1gen",
    "peraffit",
    "perfam_r60",
    "bil",
    # Unused so far; converted to keep the table consistent.
    "stra_0_19",
    "imem_stra",
)

# Below this the source reports thousands of euro.
RMPE_FAM_THOUSANDS_THRESHOLD = 1000


def prepare_fragility_index(df):
    df = df.copy()
    df.loc[df["rmpe_fam"] < RMPE_FAM_THOUSANDS_THRESHOLD, "rmpe_fam"] = df["rmpe_fam"] * 1000
    present = [c for c in PERCENT_COLUMNS if c in df.columns]
    df[present] = convert_percent_to_ratio(df[present], exclude=())
    return df


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Loading the fragility index")
        df = data_reader.open_data(
            PathRegistry.FRAGILITIES_STRUC_FRAGILITY,
            min_year=fragilities["temporal"]["min_year"],
            max_year=fragilities["temporal"]["max_year"],
        )

        logging.info("Applying unit fixes")
        df = prepare_fragility_index(df)

        df.to_parquet(PathRegistry.FRAGILITY_INDEX_BY_AREA)
        logging.info(f"Wrote {len(df)} rows of raw fragility index values")


if __name__ == "__main__":
    main()
