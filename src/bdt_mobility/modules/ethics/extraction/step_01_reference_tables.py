"""Survey values and weight tables for the ethics module.

Parameter material only: no source data is read here.
"""
import pandas as pd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("quality_of_life_survey", PathRegistry.QUALITY_OF_LIFE_SURVEY, "table"),
    PlatformOutput("MPI_weights", PathRegistry.MPI_WEIGHTS, "table"),
    PlatformOutput("income_indicators_directions", PathRegistry.INCOME_INDICATORS_DIRECTIONS, "table"),
    PlatformOutput("vulnerability_weights", PathRegistry.VULNERABILITY_WEIGHTS, "table"),
    PlatformOutput("sensibility_weights", PathRegistry.SENSIBILITY_WEIGHTS, "table"),
]

# TODO: Import of quality of life survey directly from the source and insert it in the get_dataframe
dict_quality_of_life_survey = {
    "elderly":{
    "care_load": {
        "male": 0.062, #percentuale di persone che si prende cura di anziani
        "female": 0.098
    },
    "care_difficulty": {
        "male": 0.574, #percentuale di persone che ha risposto che la cura "pesante + molto pensante"
        "female": 0.665
    },
    "care_time": {
        "male": 0.36,   #percentuale di persone che se ne occupa quotidianamente
        "female": 0.558
    },
    "care_on_their_own": {
        "male": 0.255,   # percentuale persone che se ne occupano da sole
        "female": 0.423
    }},
    "minors_under6" : {"care_load": {
        "male": 0.069, #percentuale di persone che si prende cura di anziani
        "female": 0.092
    },
    "care_difficulty": {
        "male": 0.643, #percentuale di persone che ha risposto che la cura "pesante + molto pensante"
        "female": 0.666
    },
    "care_on_their_own": {
        "male": 0.239,   # percentuale persone che se ne occupano da da sola o per la maggior parte da sola
        "female": 0.573
    }},
    "minors_6_17" :{"care_load": {
        "male": 0.115, #percentuale di persone che si prende cura di anziani
        "female": 0.126
    },
    "care_difficulty": {
        "male": 0.606, #percentuale di persone che ha risposto che la cura "pesante + molto pensante"
        "female": 0.592
    },
    "care_on_their_own": {
        "male": 0.239,   # percentuale persone che se ne occupano da da sola o per la maggior parte da sola
        "female": 0.573
    }},
    "economic":{"struggle": {
        "male": 0.26, 
        "female": 0.328
    },
    "last_year": {
        "male": 0.338, 
        "female": 0.381
    },
    "at_least_one_difficulty": {
        "male": 0.361,   
        "female": 0.396
    }}}

def get_df_from_dict(data_dict: dict, column_names: list[str]) -> pd.DataFrame:
    """
    Convert a nested dictionary into a long-format DataFrame.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary to flatten.
    column_names : list[str]
        Names of the hierarchical columns (excluding the value column).

    Returns
    -------
    pd.DataFrame
    """

    rows = []

    def unpack(d, keys):
        for k, v in d.items():
            if isinstance(v, dict):
                unpack(v, keys + [k])
            else:
                rows.append(keys + [k, v])

    unpack(data_dict, [])

    df = pd.DataFrame(rows)

    # assegna nomi colonne
    df.columns = column_names + ["value"]

    return df

def get_quality_of_life_survey_df(
    data_dict: dict
):
    df_quality_of_life = get_df_from_dict(
        data_dict,
        column_names=["category", "indicator", "gender"])
    df_quality_of_life['anno'] = '2023'
    return df_quality_of_life

def get_mpi_weights_df(
    data_dict: dict
):
    df_MPI_weights = get_df_from_dict(
        data_dict,
        column_names=["category", "indicator"])
    return df_MPI_weights

def get_economic_direction_df(
    data_dict: dict
):
    df_economic_direction = get_df_from_dict(
        data_dict,
        column_names=["indicator"])
    df_economic_direction.columns = ["indicator", "direction"]
    return df_economic_direction


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Load and saving the data from dicts")
        df_quality_of_life_survey = get_quality_of_life_survey_df(data_dict=dict_quality_of_life_survey)
        df_quality_of_life_survey.to_parquet(PathRegistry.QUALITY_OF_LIFE_SURVEY)
        df_MPI_weights = get_mpi_weights_df(fragilities["indicators"]["mpi_weights"])
        df_MPI_weights.to_parquet(PathRegistry.MPI_WEIGHTS)
        df_economic_direction = get_economic_direction_df(fragilities["economics"]["directions"])
        df_economic_direction.to_parquet(PathRegistry.INCOME_INDICATORS_DIRECTIONS)
        df_vulnerability_weights = fragilities["indicators"]["vulnerability_df"]
        df_vulnerability_weights.to_parquet(PathRegistry.VULNERABILITY_WEIGHTS)
        df_sensibility_weights = fragilities["indicators"]["sensibility_df"]
        df_sensibility_weights.to_parquet(PathRegistry.SENSIBILITY_WEIGHTS)
        logging.info("Reference tables written")


if __name__ == "__main__":
    main()
