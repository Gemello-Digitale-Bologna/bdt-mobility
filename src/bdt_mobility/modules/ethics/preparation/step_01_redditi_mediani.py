"""Median and mean IRPEF income per statistical area, by citizenship, sex and age band.

The three sources are already aggregated at their own territorial level; areas
missing a value fall back to their zona and then to their quartiere, and within
each territory to progressively more general demographic categories.

Column names stay Italian: they are the `redditi_mediani_aree` schema read by
modules/ethics/extraction/step_07_redditi.
"""
import pandas as pd
import digitalhub as dh

from bdt_mobility.utils.config import DATA_PATH, logging, logging_level
from bdt_mobility.preparation.config import PROJECT_NAME as PROJECT

REDDITI_PATH = DATA_PATH / 'preparation' / 'redditi'
REDDITI_PATH.mkdir(parents=True, exist_ok=True)

DATAITEM_NAME = "redditi_mediani_aree"

NAME = f"{DATAITEM_NAME}.parquet"

TOTAL_VALUE = "__TOTALE__"

# The publisher withholds territories below this; a median over fewer
# contributors is neither disclosable nor meaningful.
MIN_SAMPLE = 150

DIMENSIONS = ["cittadinanza", "sesso", "fascia_anni"]

AGE_BANDS = ["fino a 29", "30-44", "45-59", "60-74", "75 e oltre"]

# Fallback ladder over territories, most specific first: the area's own code,
# then the code of the zona and of the quartiere it belongs to.
LEVELS = [
    ("codice_area_statistica", "area_statistica"),
    ("cod_zona", "zona"),
    ("cod_quar", "quartiere"),
]

# Which of DIMENSIONS to keep, from the most specific category to the total.
GENERALISATIONS = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (False, False, False),
]

SOURCE_COLUMNS = {
    "cittad_f": "cittadinanza",
    "eta_fgc": "fascia_anni",
    "reddito_irpv_Median": "reddito_mediano",
    "reddito_irpv_Mean": "reddito_medio",
    "reddito_irpv_N": "contribuenti",
}

LOOKUP_KEYS = ["anno", "livello", "territorio"] + DIMENSIONS

VALUE_COLUMNS = ["reddito_mediano", "reddito_medio", "contribuenti"]

CATEGORIES = pd.DataFrame(
    [
        (citizenship, sex, age_band)
        for citizenship in [TOTAL_VALUE, "Italiani", "Stranieri"]
        for sex in [TOTAL_VALUE, "Maschile", "Femminile"]
        for age_band in [TOTAL_VALUE, *AGE_BANDS]
    ],
    columns=DIMENSIONS,
)


def prepare_incomes(df, code_column, level):
    """One row per territory, year and category, above the sample threshold."""
    result = df.rename(columns=SOURCE_COLUMNS)

    result = result[result["contribuenti"] >= MIN_SAMPLE].copy()

    result["territorio"] = result[code_column].astype(str)
    result["livello"] = level
    result["anno"] = result["anno"].astype(int)

    for dimension in DIMENSIONS:
        result[dimension] = result[dimension].fillna(TOTAL_VALUE)

    result = result[LOOKUP_KEYS + VALUE_COLUMNS]

    duplicated = int(result.duplicated(LOOKUP_KEYS).sum())
    if duplicated:
        raise ValueError(
            f"{level}: {duplicated} rows share a lookup key, so the value picked "
            f"would depend on the input order. The source is expected to hold one "
            f"row per territory, year and category."
        )

    return result


def apply_fallback(grid, incomes):
    """Fill each grid row from the closest territory and category available."""
    result = grid.copy()

    for column in VALUE_COLUMNS:
        result[column] = pd.NA
    result["fonte_reddito_mediano"] = pd.NA
    result["livello_fallback_categoria"] = pd.NA

    lookup = incomes.set_index(LOOKUP_KEYS)

    for category_priority, keep in enumerate(GENERALISATIONS):
        for code_column, level in LEVELS:
            missing = result["reddito_mediano"].isna()
            if not missing.any():
                return result

            probe = result.loc[missing, [code_column, "anno"] + DIMENSIONS].copy()
            probe = probe.rename(columns={code_column: "territorio"})
            probe["territorio"] = probe["territorio"].astype(str)
            probe["livello"] = level

            for dimension, keep_dimension in zip(DIMENSIONS, keep):
                if not keep_dimension:
                    probe[dimension] = TOTAL_VALUE

            found = probe.join(lookup, on=LOOKUP_KEYS)
            hit = found["reddito_mediano"].notna()
            index = probe.index[hit]

            for column in VALUE_COLUMNS:
                result.loc[index, column] = found.loc[hit, column].to_numpy()
            result.loc[index, "fonte_reddito_mediano"] = level
            result.loc[index, "livello_fallback_categoria"] = category_priority

    return result


def build_grid(df_statistical_areas, years):
    areas = (
        df_statistical_areas[
            [
                "codice_area_statistica",
                "area_statistica",
                "cod_zona",
                "zona",
                "cod_quar",
                "quartiere",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return (
        areas
        .merge(pd.DataFrame({"anno": sorted(years)}), how="cross")
        .merge(CATEGORIES, how="cross")
        .reset_index(drop=True)
    )


def demographic_category(df):
    return (
        df["cittadinanza"].replace({TOTAL_VALUE: "Totale"})
        + " - "
        + df["sesso"].replace({
            "Maschile": "Maschi",
            "Femminile": "Femmine",
            TOTAL_VALUE: "Totale",
        })
        + " - "
        + df["fascia_anni"].replace({TOTAL_VALUE: "Totale"})
    )


def process_incomes(df_incomes_area, df_incomes_zone, df_incomes_district, df_statistical_areas):
    incomes = pd.concat(
        [
            prepare_incomes(df_incomes_area, "COD_AREAST", "area_statistica"),
            prepare_incomes(df_incomes_zone, "CODZONA", "zona"),
            prepare_incomes(df_incomes_district, "NUMQUART", "quartiere"),
        ],
        ignore_index=True,
    )

    grid = build_grid(df_statistical_areas, incomes["anno"].unique())

    result = apply_fallback(grid, incomes)

    result["categoria_demografica"] = demographic_category(result)
    result["fonte_reddito_mediano"] = result["fonte_reddito_mediano"].fillna("non_disponibile")

    # The fallback fills object columns seeded with pd.NA; give them the dtypes
    # the parquet round-trip would infer anyway.
    result["reddito_mediano"] = pd.to_numeric(result["reddito_mediano"])
    result["reddito_medio"] = pd.to_numeric(result["reddito_medio"])
    result["contribuenti"] = pd.to_numeric(result["contribuenti"]).astype("Int64")
    result["livello_fallback_categoria"] = (
        pd.to_numeric(result["livello_fallback_categoria"]).astype("Int64")
    )

    return (
        result[
            [
                "codice_area_statistica",
                "area_statistica",
                "quartiere",
                "zona",
                "anno",
                "cittadinanza",
                "sesso",
                "fascia_anni",
                "categoria_demografica",
                "reddito_mediano",
                "reddito_medio",
                "contribuenti",
                "fonte_reddito_mediano",
                "livello_fallback_categoria",
            ]
        ]
        .sort_values(["codice_area_statistica", "anno", "categoria_demografica"])
        .reset_index(drop=True)
    )


def main():
    logging.basicConfig(level=logging_level)

    logging.info("Downloading data items")

    statistical_areas = dh.get_dataitem("aree_statistiche", PROJECT).as_df()

    incomes_area = dh.get_dataitem("redditi_area", PROJECT).as_df()
    incomes_zone = dh.get_dataitem("redditi_zona", PROJECT).as_df()
    incomes_district = dh.get_dataitem("redditi_quartiere", PROJECT).as_df()

    logging.info("Processing data (minimum sample: %d contributors)", MIN_SAMPLE)

    df = process_incomes(incomes_area, incomes_zone, incomes_district, statistical_areas)

    missing = int(df["fonte_reddito_mediano"].eq("non_disponibile").sum())
    logging.info(
        "%d rows, %d without any income (%.1f%%); sources: %s",
        len(df), missing, 100 * missing / len(df),
        df["fonte_reddito_mediano"].value_counts().to_dict(),
    )

    logging.info("Saving processed data in %s", REDDITI_PATH / NAME)

    df.to_parquet(REDDITI_PATH / NAME)
    # dh.log_dataitem(project=PROJECT, name=DATAITEM_NAME, data=df, kind="table")

    logging.info("Done")


if __name__ == '__main__':
    main()

# uv run python -m bdt_mobility.modules.ethics.preparation.step_01_redditi_mediani
