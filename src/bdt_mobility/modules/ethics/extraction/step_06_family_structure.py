"""Lone-parent shares and the incidence of families with a minor, per statistical area.

`process_family_data` is moved from step_01_fragilities_preprocess; only its signature
changed, to take the two frames directly.
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
    PlatformOutput("family_structure_by_area", PathRegistry.FAMILY_STRUCTURE_BY_AREA, "table"),
]

AGE_SUFFIXES = ("14y", "18y")


def target_columns(age_suffix: str) -> list[str]:
    return [
        f"perc_family_minors_{age_suffix}",
        f"perc_single_female_family_{age_suffix}",
        f"perc_single_male_family_{age_suffix}",
        f"perc_single_parents_{age_suffix}",
    ]


##MINORS
def process_family_data(df, df_families_total, age_suffix):
    """
    Sintetize statistics for single-parent families.
    
    Args:
        df: Original DataFrame (e.g., df_family_minors_14_years)
        df_families_total: DataFrame with the total number of families
        age_suffix: String for the suffixes (e.g., '14y' or '18y')
    """
    # 1. Flag e Split per genere
    df = df.copy() # Evitiamo SettingWithCopyWarning
    df["flg_single_mother"] = df["tipologia_famiglia"].str.contains(r"Madre|Sola", case=False, na=False).astype(int)
    df["flg_single_father"] = df["tipologia_famiglia"].str.contains(r"Padre|Solo", case=False, na=False).astype(int)
    
    df_mothers = df[df["flg_single_mother"] == 1]
    df_fathers = df[df["flg_single_father"] == 1]
    
    # 2. Raggruppamento (Somma numero famiglie)
    group_cols = ['zona', 'anno']
    df_grouped = df.groupby(group_cols)['numero_famiglie'].sum().reset_index()
    df_mothers_grouped = df_mothers.groupby(group_cols)['numero_famiglie'].sum().reset_index()
    df_fathers_grouped = df_fathers.groupby(group_cols)['numero_famiglie'].sum().reset_index()
    
    # 3. Merge e Calcolo Percentuali Genere
    # Iniziamo dal totale parziale del dataset in ingresso
    df_res = pd.merge(df_grouped, df_mothers_grouped, on=group_cols, suffixes=('_total', f'_female_{age_suffix}'))
    df_res = pd.merge(df_res, df_fathers_grouped, on=group_cols)
    df_res = df_res.rename(columns={"numero_famiglie": f"numero_family_male_{age_suffix}"})
    
    # Calcolo percentuali interne al dataset (es. madri single su tot famiglie con minori)
    df_res[f"perc_single_female_family_{age_suffix}"] = round(df_res[f"numero_famiglie_female_{age_suffix}"] / df_res["numero_famiglie_total"], 4)
    df_res[f"perc_single_male_family_{age_suffix}"] = round(df_res[f"numero_family_male_{age_suffix}"] / df_res["numero_famiglie_total"], 4)
    df_res[f"perc_single_parents_{age_suffix}"] = round((df_res[f"perc_single_female_family_{age_suffix}"] + df_res[f"perc_single_male_family_{age_suffix}"]), 4)

    # Calcolo famiglie per zone
    df_families_total = df_families_total.groupby(['zona', 'anno'])['numero_famiglie'].sum().reset_index()
    
    # 4. Merge finale con il totale generale delle zone
    df_res = pd.merge(df_res, df_families_total, on=group_cols, suffixes=(f'_minors_{age_suffix}', '_total'))
    df_res = df_res.rename(columns={"numero_famiglie_total": f'numero_famiglie_minors_{age_suffix}'})
    
    # Calcolo incidenza sul totale generale
    df_res[f'perc_family_minors_{age_suffix}'] = round(df_res[f'numero_famiglie_minors_{age_suffix}'] / df_res['numero_famiglie'], 4)
    
    return df_res

def main():
    logging.basicConfig(level=logging_level)
    with timed():
        min_year = fragilities["temporal"]["min_year"]
        max_year = fragilities["temporal"]["max_year"]

        logging.info("Loading family data")
        sources = {
            "14y": data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_MINORS_14, min_year=min_year, max_year=max_year),
            "18y": data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_MINORS_18, min_year=min_year, max_year=max_year),
        }
        df_families_total = data_reader.open_data(PathRegistry.FRAGILITIES_FAMILY_BY_ZONE, min_year=min_year, max_year=max_year)
        df_areas = data_reader.statistical_areas(PathRegistry.FRAGILITIES_STATISTICAL_AREAS)

        logging.info("Computing single-parent shares and family incidence by zone")
        df_out = None
        for age_suffix in AGE_SUFFIXES:
            df_age = process_family_data(sources[age_suffix], df_families_total, age_suffix)
            df_age = map_to_as(
                df_age, df_areas,
                target_columns=target_columns(age_suffix),
                geo_column="zona",
            )
            df_out = df_age if df_out is None else df_out.merge(df_age, on=["anno", "codice_area_statistica"])

        df_out.to_parquet(PathRegistry.FAMILY_STRUCTURE_BY_AREA)
        logging.info(f"Wrote {len(df_out)} rows of family structure by area")


if __name__ == "__main__":
    main()
