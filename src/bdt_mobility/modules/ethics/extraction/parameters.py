import pandas as pd
from bdt_mobility.utils.paths import PathRegistry

# 1. Definition of flexible "pieces" (DataFrame and complex dictionaries)
_minors_shared_indicators = ["perc_family_minors_18y", "perc_minors_soc_service", "perc_family_minors_14y", "minori1gen"]
_minors_shared_weights    = [0.1,                      0.05,                      0.15,                     0.3]

df_1_vulnerability = pd.concat([
    pd.DataFrame({
        "sector": "minors",
        "gender": "None",
        "indicators": ["perc_single_parents_18y", "perc_single_parents_14y"] + _minors_shared_indicators,
        "weight":     [0.1,                       0.1                      ] + _minors_shared_weights,
    }),
    pd.DataFrame({
        "sector": "minors",
        "gender": "female",
        "indicators": ["perc_single_parents_18y", "perc_single_parents_14y"] + _minors_shared_indicators,
        "weight":     [0.1,                       0.3                      ] + _minors_shared_weights,
    }),
    pd.DataFrame({
        "sector": "minors",
        "gender": "male",
        "indicators": ["perc_single_parents_18y", "perc_single_parents_14y"] + _minors_shared_indicators,
        "weight":     [0.1,                       0.3                      ] + _minors_shared_weights,
    }),
]).reset_index(drop=True)

_elderly_indicators = ["ultra80", "soli_65", "perfragsan", "perc_elderly_soc_service"]
_elderly_weights    = [0.2,       0.15,      0.5,          0.15]

df_2_vulnerability = pd.concat([
    pd.DataFrame({"sector": "elderly", "gender": g, "indicators": _elderly_indicators, "weight": _elderly_weights})
    for g in ("None", "female", "male")
]).reset_index(drop=True)

df_1_sensibility = pd.DataFrame({
    "sector": "elderly",
    "indicators": ["care_load", "care_difficulty", "care_time", "care_on_their_own"],
    "weight": [1.0, 0.8, 0.6, 0.9]
})

df_2_sensibility = pd.DataFrame({
    "sector": "minors",
    "indicators": ["care_load", "care_difficulty", "care_on_their_own"],
    "weight": [1.0, 0.9, 0.8]
})

df_3_sensibility = pd.DataFrame({
    "sector": "economic",
    "indicators": ["struggle", "last_year", "at_least_one_difficulty"],
    "weight": [0.40, 0.30, 0.25]
})


# 2. Creation of the main "fragilities" dictionary
fragilities = {
    
    # Temporal Parameters
    "temporal": {
        "min_year": "2021",
        "max_year": "2023",
        "ref_year": "2021"
    },

    # Economic Parameters
    "economics": {
        "gender_pay_gap": 0.056,
        "male_factor": 0.972,
        "directions": {
            "male_income": "inverse", 
            "female_income": "inverse",
            "rmpe_fam": "inverse", 
            "rmpe_fam_new": "inverse",
            "male_income_new": "inverse", 
            "female_income_new": "inverse"
        }
    },

    # Weights and Logic for Indicators
    "indicators": {
        "mpi_weights": {
            'elderly': {
                "ultra80": 0.125, 
                "soli_65": 0.125, 
                "perfragsan": 0.125,
                "perc_elderly_soc_service": 0.125, 
                "care_load": 0.125,
                "care_difficulty": 0.125, 
                "care_time": 0.125, 
                "care_on_their_own": 0.125
            },
            'minors': {
                "minori1gen": 0.125, 
                "perc_single_parents_14y": 0.125,
                "perc_single_male_family_14y": 0.125, 
                "perc_single_female_family_14y": 0.125,
                "perc_single_parents_18y": 0.125, 
                "perc_single_male_family_18y": 0.125,
                "perc_single_female_family_18y": 0.125, 
                "perc_family_minors_14y": 0.03125,
                "perc_family_minors_18y": 0.03125, 
                "perc_minors_soc_service": 0.0625,
                "care_load": 0.16, 
                "care_difficulty": 0.16, 
                "care_on_their_own": 0.16
            },
            'economic': {
                "peraffit": 0.125, 
                "perfam_r60": 0.125, 
                "bil": 0.125,
                "rmpe_fam": 0.125, 
                "struggle": 0.25, 
                "last_year": 0.125,
                "at_least_one_difficulty": 0.125
            }
        },
        "by_category": {
            "base": ["codice_area_statistica", "anno"],
            "elderly": ["ultra80", "soli_65", "perfragsan"],
            "minors": ["minori1gen"],
            "economic": ["peraffit", "rmpe_fam", "perfam_r60", "bil"]
        },
        "vulnerability_df": pd.concat([df_1_vulnerability, df_2_vulnerability]).reset_index(drop=True),
        "sensibility_df": pd.concat([df_1_sensibility, df_2_sensibility, df_3_sensibility]).reset_index(drop=True),

    },
    # Source files mapping
    "files": {
        "df_statistical_areas": PathRegistry.FRAGILITIES_STATISTICAL_AREAS,
        "df_struc_fragility": PathRegistry.FRAGILITIES_STRUC_FRAGILITY,
        "df_struc_social_services": PathRegistry.FRAGILITIES_SOCIAL_SERVICES,
        "df_struc_family_minors_14": PathRegistry.FRAGILITIES_FAMILY_MINORS_14,
        "df_struc_family_minors_18": PathRegistry.FRAGILITIES_FAMILY_MINORS_18,
        "df_struc_resident_population": PathRegistry.FRAGILITIES_RESIDENT_POPULATION,
        "df_struc_family_by_zone": PathRegistry.FRAGILITIES_FAMILY_BY_ZONE,
    },
    "SCHOOL_SLOTS" : [
            (7, 9),   # ingresso scuola
            (12, 13), # uscita pranzo
            (15, 16)  # uscita pomeriggio
        ]
}