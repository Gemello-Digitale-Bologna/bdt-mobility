"""TEMPORARY: rebuilds the legacy df_{category}_indicators_{gender} and MPI tables.

Reads only the outputs of step_01 and step_03..step_06, never a raw source. Everything
below `scale_by_indicator_each_year` is the tail of step_01_fragilities_preprocess, moved
unchanged; the assembly at the top replaces get_indicators_by_category.

Delete once the model assembles the partials itself.
"""
import re

import pandas as pd

from bdt_mobility.modules.ethics import mpi_toolkit as MPI_toolkit
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.modules.ethics.extraction.parameters import fragilities

# df_{category}_indicators_{gender}.parquet and df_{category}_MPI_{gender}.parquet are dynamic
# outputs -- enumeration requires runtime knowledge of categories/genders; not yet logged to platform.
MAIN_OUTPUTS = []

# The column order the legacy tables carry.
KEY_COLUMNS = fragilities["indicators"]["by_category"]["base"]


def _read(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _fragility(category: str) -> pd.DataFrame:
    """Base + category columns of the fragility index."""
    columns = fragilities["indicators"]["by_category"][category]
    return _read(PathRegistry.FRAGILITY_INDEX_BY_AREA)[KEY_COLUMNS + columns]


def _social_service(category: str) -> pd.DataFrame:
    df = _read(PathRegistry.SOCIAL_SERVICES_BY_AREA)
    df = df[df["category"] == category]
    return df[KEY_COLUMNS + ["value"]].rename(columns={"value": f"perc_{category}_soc_service"})


def _population() -> pd.DataFrame:
    """Residents per area, for the reference weighting."""
    return _read(PathRegistry.POPULATION_BY_AREA)


def build_category_frame(category: str) -> pd.DataFrame:
    """Wide indicator frame of one category, before the survey join."""
    df = _fragility(category)

    if category == "elderly":
        return df.merge(_social_service("elderly"), on=KEY_COLUMNS)

    if category == "minors":
        df = df.merge(_read(PathRegistry.FAMILY_STRUCTURE_BY_AREA), on=KEY_COLUMNS)
        return df.merge(_social_service("minors"), on=KEY_COLUMNS)

    # economic: split rmpe_fam by a constant gender gap.
    df = df.copy()
    df["male_income"] = df["rmpe_fam"] / fragilities["economics"]["male_factor"]
    df["female_income"] = df["male_income"] * (1 - fragilities["economics"]["gender_pay_gap"])
    return df


# Indicator driving the survey scaling; None leaves the values unscaled.
SCALE_INDICATOR = {"elderly": "care_load", "minors": "care_load", "economic": None}


def get_indicators_by_category(df_quality_of_life_survey, category_name, use_year_match, gender=None):
    """Category frame plus the survey constants."""
    df = build_category_frame(category_name)
    df_survey_temp = aggregate_mean_scaled(
        df_quality_of_life_survey, category=category_name, gender=gender,
        scale_indicator=SCALE_INDICATOR[category_name],
    )
    columns_to_exclude = df_survey_temp['indicator'].tolist()
    df_survey_temp_pivot = df_survey_temp.pivot(index='anno', columns='indicator', values='value').reset_index()
    df = join_indicator_dfs(df, df_survey_temp_pivot, use_year_match, year_col='anno')
    if use_year_match==False:
        df[columns_to_exclude] = df[columns_to_exclude] * 100
    return columns_to_exclude, df


# --- moved from step_01_fragilities_preprocess ---

def scale_by_indicator_each_year(df, indicator_name):
    """
    For each year:
        - find the value of the indicator `indicator_name`
        - multiply all values of that year by that value

    Parameters
    ---------
    df : DataFrame with columns [indicator, anno, value]
    indicator_name : str, name of the indicator to use as a scaling factor

    Returns
    -------
    df_scaled : DataFrame
    """

    df_scaled = df.copy()

    # obtain all the years present
    years = df_scaled["anno"].unique()

    for year in years:
        # select the value of the given indicator in the given year
        row = df_scaled[(df_scaled["indicator"] == indicator_name) &
                        (df_scaled["anno"] == year)]

        if row.empty:
            print(f"[WARN] No value found for '{indicator_name}' in year {year}. Skipping.")
            continue

        scale_value = row["value"].iloc[0]

        # multiply all rows of that year
        df_scaled.loc[(df_scaled["anno"] == year) & (df_scaled["indicator"] != indicator_name), "value"] *= scale_value

    return df_scaled

def aggregate_mean_scaled(df, category, gender=None, scale_indicator="care_load"):
    # 1. Category filter
    if category == "minors":
        subcategories = ["minors_under6", "minors_6_17"]
        df_work = df[df["category"].isin(subcategories)].copy()
    else:
        df_work = df[df["category"] == category].copy()

    # 2. Gender filter (if explicitly requested)
    if gender is not None:
        df_work = df_work[df_work["gender"].str.lower() == gender.lower()]

    # 3. SCALE LOGIC (Only if scale_indicator is not None)
    if scale_indicator is not None:
        genders_to_process = df_work["gender"].unique()
        final_scaled_list = []

        for g in genders_to_process:
            df_gen = df_work[df_work["gender"] == g].copy()
            
            if category == "minors":
                for subcat in ["minors_under6", "minors_6_17"]:
                    temp_sub = df_gen[df_gen["category"] == subcat].copy()
                    if not temp_sub.empty:
                        # Different scale logic for each gender and age group
                        temp_sub = scale_by_indicator_each_year(temp_sub, scale_indicator)
                        final_scaled_list.append(temp_sub)
            else:
                # Scaling for each gender for the other categories (e.g., elderly)
                df_gen = scale_by_indicator_each_year(df_gen, scale_indicator)
                final_scaled_list.append(df_gen)

        df_work = pd.concat(final_scaled_list)
    #else:
    #    # If scale_indicator is None, we don't do anything and df_work remains as is
    #    print(f"[INFO] Scaled per category {category}.")

    # 4. FINAL AGGREGATION
    if category == "minors":
        group_keys = ["indicator", "anno"]
        if gender is None:
            group_keys.append("gender")
        # Sum of age groups (under6 + 6_17) maintaining genders separated
        df_work = df_work.groupby(group_keys, as_index=False)["value"].sum()

    # If gender is None, we make the average between Male and Female to get a unique indicator for area/anno
    if gender is None:
        df_agg = df_work.groupby(["indicator", "anno"], as_index=False)["value"].mean()
    else:
        df_agg = df_work.groupby(["indicator", "anno"], as_index=False)["value"].mean()
        
    return df_agg

def join_indicator_dfs(df_main, df_to_add, use_year_match=True, year_col='anno'):
    """
    Joins two indicator DataFrames.
    
    Parameters:
    -----------
    df_main : Main DataFrame (e.g., the one with statistical area and year)
    df_to_add : DataFrame to add (e.g., the one obtained from the survey pivot)
    use_year_match : bool, if True executes a merge on 'year'. If False, concatenates the columns.
    year_col : str, name of the year column.
    """
    
    if use_year_match:
        df_final = pd.merge(df_main, df_to_add, on=year_col, how='left')
    else:
        df_final= df_main.copy()
        survey_values = df_to_add.iloc[0].to_dict()
        # Creation of columns in the main DataFrame using the values from the dictionary
        for col, val in survey_values.items():
            if col != 'anno': # Avoid overwriting the year if not needed
                df_final[col] = val
        
    return df_final

def get_fixed_columns(df, category_name, indicators_dict):
    soc_service_cols = [c for c in df.columns if "soc_service" in c]
    fixed_cols = indicators_dict.get("base", []) + indicators_dict.get(category_name, [])
    fixed_cols = list(set(fixed_cols + soc_service_cols))
    return fixed_cols

def filter_columns_smart(df, gender_type, category_name, indicators_dict, quality_of_life_indicators):
    """
    Goal:
        Dynamically filters the DataFrame to retain only relevant columns based on the specific 
        category, gender filter, and family-related structural indicators.

    Input:
        df (pd.DataFrame): The input dataset containing all available indicators.
        gender_type (str|None): Filter for gender-specific columns ('male', 'female') or None for the total population.
        category_name (str): Analysis category used to identify fixed indicators.
        indicators_dict (dict): Mapping of categories to their corresponding structural indicators.
        quality_of_life_indicators (list): List of columns related to Quality of Life survey data.

    Output:
        pd.DataFrame: A filtered copy of the DataFrame containing the selected subset of columns.
    """

    # 1. Build the list of "fixed" columns (always present)
    # Take 'base' (e.g., year, statistical area code) + the specific category (e.g., 'elderly')
    # fixed_cols = indicators_dict.get("base", []) + indicators_dict.get(category, []) + quality_of_life_indicators
    
    # 2. Add columns dynamically that contain "soc_service"
    # soc_service_cols = [c for c in df.columns if "soc_service" in c]
    # fixed_cols = list(set(fixed_cols + soc_service_cols)) # set per evitare duplicati

    fixed_cols = get_fixed_columns(df, category_name=category_name, indicators_dict=indicators_dict)
    fixed_cols = fixed_cols + quality_of_life_indicators
    # 3. Identify the columns that contain the gender (e.g., "donne")
    # Exclude those already identified as fixed to avoid duplicates

    if gender_type is not None:
        pattern = rf"(^|_){gender_type.lower()}(_|$)"
        gender_cols = [
            c for c in df.columns 
            if re.search(pattern, c.lower()) and c not in fixed_cols
        ]
        gender_cols = gender_cols + ['perc_family_minors_14y', 'perc_family_minors_18y']
    else:
        gender_cols = ['perc_family_minors_14y','perc_family_minors_18y','perc_single_parents_18y', 'perc_single_parents_14y' ]
    # 4. FFinal list of columns to keep
    # Verify that they actually exist in the DataFrame
    cols_to_keep = [c for c in (fixed_cols + gender_cols) if c in df.columns]
    
    return df[cols_to_keep].copy()

def keep_suffixed_columns(df: pd.DataFrame, suffix: str = "norm") -> pd.DataFrame:
    """
    If in df there are duplicate columns of the type:
        colname           (base version)
        colname_suffix    (normalized version)
    then:
      - keeps only colname_suffix
      - renames it to colname
    """   
    # Regex to obtain columns that end with _suffix
    pattern = re.compile(rf"(.+?)_{suffix}$")
    rename_map = {}
    cols_to_drop = []
    # For each column, check if it has the base+suffix form
    for col in df.columns:
        match = pattern.match(col)
        if match:
            base_name = match.group(1)   
            # If the base column exists, drop it
            if base_name in df.columns:
                cols_to_drop.append(base_name)            
            # Rename col_suffix → col
            rename_map[col] = base_name
    # Apply drop of the base columns
    df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")
    # Rename the columns with suffix
    df_cleaned = df_cleaned.rename(columns=rename_map)
    return df_cleaned

def normalize_indicators(df, columns_to_exclude=['anno', 'codice_area_statistica', 'zona_pross','care_difficulty',  'care_load',  'care_on_their_own',  'care_time'], ref_year="2021"):
    """
    Goal:
        Standardizes indicators using a reference year and constrained min-max scaling 
        across the project's temporal range.

    Input:
        df (pd.DataFrame): Dataset containing the indicators to be normalized.
        columns_to_exclude (list): Metadata and specific indicators to skip during processing.
        ref_year (str): Baseline year used to establish reference values.

    Output:
        pd.DataFrame: A DataFrame containing only the normalized columns (suffixed with "norm").
    """
    df_temp = df.copy()
    df_ref = MPI_toolkit.get_reference_df(df_temp, _population(), ref_year =ref_year, columns_to_exclude =columns_to_exclude, year_col = "anno", geo_col = "codice_area_statistica", people_col = "residenti")
    df_norm = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp, df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=columns_to_exclude, year_col ="anno", geo_col="codice_area_statistica", column_for_norm=None)
    df_norm = keep_suffixed_columns(df_norm, suffix="norm")
    return df_norm


def normalize_indicators_gender(df, columns_to_exclude, category_name, ref_year, indicators_dict):
    """
    Goal:
        Performs gender-aware normalization for 'minors' and 'economic' categories, 
        adjusting reference values and handling gender-based population splits.

    Input:
        df (pd.DataFrame): Raw indicator dataset.
        columns_to_exclude (list): ID, year, and metadata columns to skip.
        category_name (str): Target category ('minors' or 'economic').
        ref_year (str): Baseline year for reference calculations.
        indicators_dict (dict): Mapping of indicators associated with each category.

    Output:
        pd.DataFrame: Normalized dataset containing only processed columns (suffixed with "norm").
    """
    df_temp = df.copy()
    df_ref = MPI_toolkit.get_reference_df(
        df_temp, 
        _population(), 
        ref_year=ref_year, 
        columns_to_exclude=columns_to_exclude, 
        year_col="anno", 
        geo_col="codice_area_statistica", 
        people_col="residenti"
    )
    # Gender and minors logic
    fixed_col = get_fixed_columns(df_temp, category_name=category_name, indicators_dict=indicators_dict) + columns_to_exclude
    if category_name == "minors":
        targets = ['perc_single_parents_14y', 'perc_single_parents_18y']
        fixed_col = list(set(fixed_col + ['perc_family_minors_14y', 'perc_family_minors_18y']))
        df_minors_norm = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp[fixed_col], df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=columns_to_exclude, year_col ="anno", geo_col="codice_area_statistica", column_for_norm=None)
        for target in targets:
            mask = df_ref['indicator'] == target
            df_temp[target] = df_temp[target] / 2  
            if not df_ref[mask].empty:
                # halve the reference value for each gender
                valore_dimezzato = df_ref.loc[mask, 'ref_value'] / 2
                df_ref.loc[mask, 'ref_value'] = round(valore_dimezzato, 4)
            logging.debug(f"Mask: {mask}, Target: {df_temp[target]}")
        df_minors_gender_norm_14y = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp[['codice_area_statistica', 'anno', 'perc_single_female_family_14y', 'perc_single_male_family_14y','perc_single_parents_14y']], df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=['anno', 'codice_area_statistica', 'zona'], year_col ="anno", geo_col="codice_area_statistica", column_for_norm=['perc_single_female_family_14y','perc_single_male_family_14y', 'perc_single_parents_14y'], ref_value="perc_single_parents_14y")
        df_minors_gender_norm_18y = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp[['codice_area_statistica', 'anno', 'perc_single_female_family_18y', 'perc_single_male_family_18y','perc_single_parents_18y']], df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=['anno', 'codice_area_statistica', 'zona'], year_col ="anno", geo_col="codice_area_statistica", column_for_norm=['perc_single_female_family_18y','perc_single_male_family_18y', 'perc_single_parents_18y'], ref_value="perc_single_parents_18y")
        df_gender_merged = pd.merge(df_minors_gender_norm_14y, df_minors_gender_norm_18y, on=['anno', 'codice_area_statistica'])
        df_gender_final = pd.merge(df_minors_norm, df_gender_merged, on=['anno', 'codice_area_statistica'])
        df_norm = keep_suffixed_columns(df_gender_final, suffix="norm")
    if category_name == 'economic':
        fixed_col = list(set(fixed_col) - {'rmpe_fam'})
        df_economic_norm = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp[fixed_col], df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=columns_to_exclude, year_col ="anno", geo_col="codice_area_statistica", column_for_norm=None)
        df_economic_gender_norm = MPI_toolkit.normalize_indicators_vincolato_wide(df_temp[['codice_area_statistica', 'anno','female_income', 'male_income', 'rmpe_fam']], df_ref, minmax_years=(fragilities['temporal']['min_year'], fragilities['temporal']['max_year']), columns_to_exclude=columns_to_exclude, year_col ="anno", geo_col="codice_area_statistica", column_for_norm=['female_income','male_income', 'rmpe_fam'], ref_value="rmpe_fam")
        df_economic_final = pd.merge(df_economic_norm, df_economic_gender_norm, on=['anno', 'codice_area_statistica'])
        df_norm = keep_suffixed_columns(df_economic_final, suffix="norm")
    return df_norm


def get_multidimensional_index_category(
    df_weights: pd.DataFrame,
    df_economic_direction: pd.DataFrame|None,
    df_quality_of_life_survey: pd.DataFrame,
    category_name: str, 
    use_year_match: bool, 
    gender:str|None):
    """
    Goal:
        Calculates the Multidimensional Poverty Index (MPI) for a specific social category 
        (elderly, minors, or economic) and gender.

    Input:
        df_weights (pd.DataFrame): Indicator weights per category.
        category_name (str): Target category ('elderly', 'minors', 'economic').
        use_year_match (bool): Enable temporal alignment for indicators.
        gender (str|None): Filter by 'male', 'female', or None for total population.

    Output:
        tuple (pd.DataFrame, pd.DataFrame): 
            - df_temp_indicators: The processed NOT normalized indicators.
            - df_MPI: The resulting MPI values per year and area NOT normalized.
    """
    results = []
    indicators_by_category = fragilities['indicators']['by_category']
    ref_year=fragilities['temporal']['ref_year']
    quality_of_life_indicators, df_temp_indicators = get_indicators_by_category(df_quality_of_life_survey=df_quality_of_life_survey, category_name=category_name, use_year_match=use_year_match, gender=gender)
    columns_to_exclude = ['anno', 'codice_area_statistica'] + quality_of_life_indicators
    if category_name != 'elderly':# and gender != None:
        df_temp = normalize_indicators_gender(df_temp_indicators, columns_to_exclude=columns_to_exclude, category_name=category_name, ref_year=ref_year, indicators_dict=indicators_by_category)
    else:
        df_temp = normalize_indicators(df_temp_indicators, columns_to_exclude=columns_to_exclude, ref_year=ref_year)
    df_temp = filter_columns_smart(df_temp, gender_type=gender, category_name=category_name, indicators_dict=indicators_by_category, quality_of_life_indicators=quality_of_life_indicators)
    df_weights_temp = df_weights[df_weights["category"]==category_name].copy()
    if category_name == "economic":
        df_directions = df_economic_direction
    else:
        df_directions = None
    for id, year in df_temp[["codice_area_statistica", "anno"]].drop_duplicates().itertuples(index=False):
        indice = MPI_toolkit.get_MPI_index_as(df_temp, id, year, df_weights_temp, df_directions=df_directions)
        col_name = f"MPI_{category_name}" if gender is None else f"MPI_{category_name}_{gender}"
        results.append({
            "anno": year, 
            "codice_area_statistica": id, 
            col_name: indice
        })
        #results.append({"anno": year, "codice_area_statistica": id, "MPI_" + category_name + "_" + gender: indice})
    df_MPI = pd.DataFrame(results)
    return df_temp_indicators, df_MPI


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        df_quality_of_life_survey = _read(PathRegistry.QUALITY_OF_LIFE_SURVEY)
        df_MPI_weights = _read(PathRegistry.MPI_WEIGHTS)
        df_economic_direction = _read(PathRegistry.INCOME_INDICATORS_DIRECTIONS)

        logging.info("Computing MPIs per category")
        for category in ["elderly", "minors", 'economic']:
            for gender in [None, "female", "male"]:
                df_temp_indicators, df_MPI = get_multidimensional_index_category(
                    df_weights=df_MPI_weights,
                    df_quality_of_life_survey=df_quality_of_life_survey,
                    df_economic_direction=df_economic_direction,
                    category_name=category,
                    use_year_match=False,
                    gender=gender)

                logging.debug(f"Category: {category}, Gender: {gender}")
                logging.debug(f"Temp Indicators:\n{df_temp_indicators}")
                logging.debug(f"MPI:\n{df_MPI}")

                df_temp_indicators.to_parquet(PathRegistry.FRAGILITY_INDICATORS.substitute(category=category, gender=gender))
                df_MPI.to_parquet(PathRegistry.FRAGILITY_MPI.substitute(category=category, gender=gender))

        logging.info("Analysis completed and output saved")


if __name__ == "__main__":
    main()
