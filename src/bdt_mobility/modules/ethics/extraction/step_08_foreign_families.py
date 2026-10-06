"""Lone-parent families by the citizenship of the head of household and of the minors.

Not read by any indicator yet. Households with no foreign member come from the 18y
source, which does not record the citizenship of the head of household.
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
    PlatformOutput("foreign_families_percentages", PathRegistry.FOREIGN_FAMILIES_PERCENTAGES, "table"),
    PlatformOutput("foreign_families_percentages_city", PathRegistry.FOREIGN_FAMILIES_PERCENTAGES_CITY, "table"),
]

# Columns shared by the foreign-headed and italian-headed family datasets once aligned.
FAMILY_COLUMNS = [
    "anno",
    "zona",
    "quartiere",
    "cittadinanza_capofamiglia",
    "componenti",
    "minori_stranieri",
    "numero_famiglie",
    "flg_only_foreigners",
    "flg_single_mother",
    "flg_single_father",
]

CATEGORY_NAMES = [
    "padri_stranieri_soli_con_minori_stranieri",
    "madri_straniere_sole_con_minori_stranieri",
    "padri_italiani_soli_con_minori_stranieri",
    "madri_italiane_sole_con_minori_stranieri",
    "madri_sole_con_minori_italiani",
    "padri_soli_con_minori_italiani",
]

# Ratio used where a zone has no lone parents of either kind.
GENDER_PARITY = 0.5

# Mother/father column pairs compared within the same family concept.
GENDER_PAIRS = {
    "stranieri_soli_con_minori_stranieri": (
        "madri_straniere_sole_con_minori_stranieri",
        "padri_stranieri_soli_con_minori_stranieri",
    ),
    "italiani_soli_con_minori_stranieri": (
        "madri_italiane_sole_con_minori_stranieri",
        "padri_italiani_soli_con_minori_stranieri",
    ),
    "soli_con_minori_italiani": (
        "madri_sole_con_minori_italiani",
        "padri_soli_con_minori_italiani",
    ),
}


def _cast_component_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Numbers of family members are capped/censored as '6 e oltre' / 'Oltre 6' in the source data."""
    df = df.copy()
    for column in columns:
        df[column] = df[column].replace("6 e oltre", 6).replace("Oltre 6", 6).astype(int)
    return df


def _add_single_parent_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tipologia = df["tipologia_famiglia"]
    df["flg_single_mother"] = tipologia.str.contains(r"Madre|Sola", case=False, na=False).astype(int)
    df["flg_single_father"] = tipologia.str.contains(r"Padre|Solo", case=False, na=False).astype(int)
    return df


def prepare_foreign_families(df_famiglie_straniere: pd.DataFrame) -> pd.DataFrame:
    """Families with at least one foreign member, restricted to those with a foreign minor."""
    df = _cast_component_counts(df_famiglie_straniere, ["componenti", "componenti_stranieri", "minori_stranieri"])
    df["flg_only_foreigners"] = (df["cittadinanza_componenti"] == "solo componenti stranieri").astype(int)
    df = _add_single_parent_flags(df)
    df = df[df["minori_stranieri"] > 0]
    return df[FAMILY_COLUMNS]


def prepare_italian_families(df_minori_18: pd.DataFrame) -> pd.DataFrame:
    """Families with at least one minor and no foreign minors (head of household citizenship unknown)."""
    df = _cast_component_counts(df_minori_18, ["num_comp_minorenni", "num_comp_minorenni_stranieri", "dimensione_familiare"])
    df = _add_single_parent_flags(df)
    mask = (df["num_comp_minorenni_stranieri"] == 0) & (df["num_comp_minorenni"] > 0)
    df = df[mask].rename(columns={
        "dimensione_familiare": "componenti",
        "num_comp_minorenni_stranieri": "minori_stranieri",
    })
    df["cittadinanza_capofamiglia"] = "Unknown"
    df["flg_only_foreigners"] = 0
    return df[FAMILY_COLUMNS]


def build_family_categories(df_famiglie_minori: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks (over df_famiglie_minori) for each of CATEGORY_NAMES."""
    is_father = df_famiglie_minori["flg_single_father"] == 1
    is_mother = df_famiglie_minori["flg_single_mother"] == 1
    has_foreign_minors = df_famiglie_minori["minori_stranieri"] > 0
    is_foreign = df_famiglie_minori["cittadinanza_capofamiglia"] == "Straniera"
    is_italian = df_famiglie_minori["cittadinanza_capofamiglia"] == "Italiana"

    return {
        "padri_stranieri_soli_con_minori_stranieri": is_father & has_foreign_minors & is_foreign,
        "madri_straniere_sole_con_minori_stranieri": is_mother & has_foreign_minors & is_foreign,
        "padri_italiani_soli_con_minori_stranieri": is_father & has_foreign_minors & is_italian,
        "madri_italiane_sole_con_minori_stranieri": is_mother & has_foreign_minors & is_italian,
        "madri_sole_con_minori_italiani": is_mother & ~has_foreign_minors,
        "padri_soli_con_minori_italiani": is_father & ~has_foreign_minors,
    }


def _mother_share(madri: pd.Series, padri: pd.Series) -> pd.Series:
    """Share of this family concept's lone parents that are mothers, in [0, 1].

    Both-zero zones get GENDER_PARITY: the concept is absent there, not balanced
    towards fathers, and 0.0 would wrongly read as "all fathers".
    """
    total = madri + padri
    return (madri / total).where(total > 0, GENDER_PARITY)


def compute_family_percentages(df_famiglie_minori: pd.DataFrame, df_minori_18: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Percentage of each family category over all families with a minor, grouped by group_cols."""
    totali = df_minori_18.groupby(group_cols)["numero_famiglie"].sum()
    categorie = build_family_categories(df_famiglie_minori)

    percentuali = []
    for nome, mask in categorie.items():
        numeratore = df_famiglie_minori.loc[mask].groupby(group_cols)["numero_famiglie"].sum()
        percentuali.append((100 * numeratore / totali).rename(nome))
    df_percentuali = pd.concat(percentuali, axis=1).fillna(0).reset_index()

    for nome, (col_madri, col_padri) in GENDER_PAIRS.items():
        df_percentuali[f"gender_ratio_{nome}"] = _mother_share(
            df_percentuali[col_madri], df_percentuali[col_padri]
        )

    return df_percentuali


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        min_year = fragilities["temporal"]["min_year"]
        max_year = fragilities["temporal"]["max_year"]

        logging.info("Loading foreign and minors family data")
        df_famiglie_straniere = data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_FOREIGNERS, min_year=min_year, max_year=max_year)
        df_minori_18 = data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_MINORS_18, min_year=min_year, max_year=max_year)
        df_areas = data_reader.statistical_areas(PathRegistry.FRAGILITIES_STATISTICAL_AREAS)

        logging.info("Aligning foreign-headed and italian-headed family datasets")
        df_famiglie_minori = pd.concat(
            [prepare_foreign_families(df_famiglie_straniere), prepare_italian_families(df_minori_18)],
            ignore_index=True,
        )

        logging.info("Computing family category percentages by zone")
        df_percentuali_zona = compute_family_percentages(df_famiglie_minori, df_minori_18, group_cols=["anno", "zona"])

        logging.info("Mapping percentages to statistical areas")
        target_columns = CATEGORY_NAMES + [f"gender_ratio_{nome}" for nome in GENDER_PAIRS]
        df_percentuali_as = map_to_as(df_percentuali_zona, df_areas, target_columns=target_columns, geo_column="zona")
        df_percentuali_as.to_parquet(PathRegistry.FOREIGN_FAMILIES_PERCENTAGES)

        logging.info("Computing city-wide family category percentages by year")
        df_percentuali_citta = compute_family_percentages(df_famiglie_minori, df_minori_18, group_cols=["anno"])
        df_percentuali_citta.to_parquet(PathRegistry.FOREIGN_FAMILIES_PERCENTAGES_CITY)

        logging.info("Analysis completed and output saved")


if __name__ == "__main__":
    main()
