from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bdt_mobility.utils.paths import PathRegistry

from .policy import CareTimePolicy, EconomicBurdenPolicy, Policy


GenderKey = Literal["None", "male", "female"]


@dataclass(frozen=True)
class EthicalCaseSpec:
    case_id: str
    sector_key: str
    mpi_sector_key: str
    dataset_prefix: str
    mpi_value_prefix: str
    policy: Policy
    survey_cols: tuple[str, ...] = ()
    ref_year: str = "2021"
    minmax_years: tuple[str, str] = ("2021", "2023")
    inverse_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedEthicalSliceSpec:
    slice_id: str
    case_spec: EthicalCaseSpec
    gender: GenderKey
    indicators_path: Path
    indicators_schema: str
    baseline_mpi_path: Path
    baseline_mpi_value_col: str
    baseline_mpi_name: str
    modified_mpi_name: str
    delta_mpi_name: str


CASE_SPECS = {
    "elderly": EthicalCaseSpec(
        case_id="elderly",
        sector_key="elderly",
        mpi_sector_key="elderly",
        dataset_prefix="elderly",
        mpi_value_prefix="elderly",
        policy=CareTimePolicy(
            care_intervals=(),
        ),
        survey_cols=("care_load", "care_difficulty", "care_on_their_own", "care_time"),
    ),
    "minors": EthicalCaseSpec(
        case_id="minors",
        sector_key="minors",
        mpi_sector_key="minors",
        dataset_prefix="minors",
        mpi_value_prefix="minors",
        policy=CareTimePolicy(
            care_intervals=((7.0, 9.0), (12.0, 13.0), (15.0, 16.0)),
        ),
        survey_cols=("care_load", "care_difficulty", "care_on_their_own"),
    ),
    "income": EthicalCaseSpec(
        case_id="income",
        sector_key="economic",
        mpi_sector_key="economic",
        dataset_prefix="economic",
        mpi_value_prefix="economic",
        policy=EconomicBurdenPolicy(
            income_threshold=13000.0,
        ),
        survey_cols=("struggle", "last_year", "at_least_one_difficulty"),
        inverse_cols=("rmpe_fam", "male_income", "female_income"),
    ),
}


def get_case_spec(case_id: str) -> EthicalCaseSpec:
    try:
        return CASE_SPECS[case_id]
    except KeyError as exc:
        known = ", ".join(sorted(CASE_SPECS))
        raise ValueError(f"Unknown ethical case {case_id!r}. Expected one of: {known}.") from exc


def resolve_case_slice(case_id: str, gender: GenderKey = "None") -> ResolvedEthicalSliceSpec:
    case_spec = get_case_spec(case_id)
    if gender not in {"None", "male", "female"}:
        raise ValueError(f"Unknown gender {gender!r}. Expected one of: 'None', 'male', 'female'.")

    schema_gender = "None" if gender == "None" else gender.capitalize()
    gender_suffix = "" if gender == "None" else f"_{gender}"

    return ResolvedEthicalSliceSpec(
        slice_id=f"{case_id}/{gender}",
        case_spec=case_spec,
        gender=gender,
        indicators_path=PathRegistry.FRAGILITY_INDICATORS.substitute(category=case_spec.dataset_prefix, gender=gender),
        indicators_schema=f"{case_spec.dataset_prefix}Indicators{schema_gender}",
        baseline_mpi_path=PathRegistry.FRAGILITY_MPI.substitute(category=case_spec.dataset_prefix, gender=gender),
        baseline_mpi_value_col=f"MPI_{case_spec.mpi_value_prefix}{gender_suffix}",
        baseline_mpi_name=f"I_baseline_MPI_{case_id}{gender_suffix}",
        modified_mpi_name=f"I_modified_MPI_{case_id}{gender_suffix}",
        delta_mpi_name=f"I_delta_MPI_{case_id}{gender_suffix}",
    )
