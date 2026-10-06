from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from civic_digital_twins.dt_model import Index

from bdt_mobility.modules.ethics.model.bundle import EthicalModelBundle
from bdt_mobility.modules.ethics.model.cases import resolve_case_slice
from bdt_mobility.modules.ethics.model.loading.loaders import CaseLoader
from bdt_mobility.modules.ethics.model.model import EthicalModel

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
GENDERS = ("None", "female", "male")

EXPECTED_CANONICAL_COLS = [
    "minori1gen",
    "perc_family_minors_14y",
    "perc_single_parents_14y",
    "perc_family_minors_18y",
    "perc_single_parents_18y",
    "perc_minors_soc_service",
    "care_difficulty",
    "care_load",
    "care_on_their_own",
]


def _load_raw(gender: str) -> pd.DataFrame:
    slice_spec = resolve_case_slice("minors", gender)
    return pd.read_parquet(slice_spec.indicators_path).fillna(0)


def _load_arranged(gender: str) -> pd.DataFrame:
    slice_spec = resolve_case_slice("minors", gender)
    return CaseLoader(slice_spec, base_dir=REPO_ROOT).load().dfs["indicators"]


@pytest.mark.parametrize(
    "gender, raw_parent_14y",
    [
        ("None", "perc_single_parents_14y"),
        ("female", "perc_single_female_family_14y"),
        ("male", "perc_single_male_family_14y"),
    ],
)
def test_canonical_parent_mapping_by_slice(gender, raw_parent_14y):
    _load_raw(gender)
    df_arranged = _load_arranged(gender)

    assert "perc_single_parents_14y" in df_arranged.columns
    assert "perc_single_parents_18y" in df_arranged.columns
    assert (raw_parent_14y if gender != "None" else "SKIP") not in df_arranged.columns


@pytest.mark.parametrize("gender", GENDERS)
def test_case_loader_exposes_canonical_columns_for_all_slices(gender):
    expected_columns = ["codice_area_statistica", "anno", *EXPECTED_CANONICAL_COLS]

    loaded = CaseLoader(resolve_case_slice("minors", gender), base_dir=REPO_ROOT).load()
    arranged = loaded.dfs["indicators"]
    assert list(arranged.columns) == expected_columns

    indexed = loaded.indicators
    sample_area = next(iter(indexed.values()))
    sample_year = next(iter(sample_area.values()))
    assert set(sample_year.keys()) == set(EXPECTED_CANONICAL_COLS)
    assert set(loaded.vulnerability_weights.keys()) == {
        "minori1gen",
        "perc_family_minors_14y",
        "perc_single_parents_14y",
        "perc_family_minors_18y",
        "perc_single_parents_18y",
        "perc_minors_soc_service",
    }
    assert set(loaded.sensibility_weights.keys()) == {
        "care_difficulty",
        "care_load",
        "care_on_their_own",
    }
    assert set(loaded.mpi_weights.keys()) == set(EXPECTED_CANONICAL_COLS)


def test_parent_indicators_differ_by_slice_after_normalization():
    arranged_none = _load_arranged("None")
    arranged_female = _load_arranged("female")
    arranged_male = _load_arranged("male")

    for col in ("perc_single_parents_14y", "perc_single_parents_18y"):
        assert not arranged_none[col].equals(arranged_female[col])
        assert not arranged_none[col].equals(arranged_male[col])


@pytest.mark.parametrize("col", ["care_difficulty", "care_load", "care_on_their_own"])
def test_survey_values_remain_slice_specific(col):
    raw_female = _load_raw("female")
    raw_male = _load_raw("male")
    arranged_female = _load_arranged("female")
    arranged_male = _load_arranged("male")

    assert arranged_female[col].equals(raw_female[col])
    assert arranged_male[col].equals(raw_male[col])
    assert not arranged_female[col].equals(arranged_male[col])


@pytest.fixture
def input_kwargs():
    return {
        "ztl_start": Index("I_test_ztl_start", 8 * 3600),
        "ztl_end": Index("I_test_ztl_end", 18 * 3600),
        "avail_postpone": Index("I_test_avail_postpone", 3.0),
        "avail_anticipate": Index("I_test_avail_anticipate", 3.0),
        "median_postpone": Index("I_test_median_postpone", 1.0),
        "median_anticipate": Index("I_test_median_anticipate", 1.0),
        "ticket_costs": {},
        "euro_class_split": {},
    }


@pytest.mark.parametrize("gender", GENDERS)
def test_minors_model_smoke(input_kwargs, gender):
    model = EthicalModel(
        inputs=EthicalModel.Inputs(**input_kwargs),
        config=EthicalModel.Config(gender=gender, case_id="minors", base_dir=REPO_ROOT),
    )
    assert model.I_baseline_mpi
    assert model.I_modified_mpi
    assert model.I_delta_mpi
    assert model.I_vulnerability_by_area


def test_minors_bundle_smoke(input_kwargs):
    bundle = EthicalModelBundle(
        genders=GENDERS,
        case_id="minors",
        base_dir=REPO_ROOT,
        **input_kwargs,
    )
    assert set(bundle.models.keys()) == set(GENDERS)
