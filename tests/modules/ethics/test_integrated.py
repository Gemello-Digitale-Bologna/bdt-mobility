from __future__ import annotations

import numpy as np
import pytest

from bdt_mobility.model import MobilityModel, evaluate_model
from bdt_mobility.model.config_vars import behavioral_params, policy_params, config_params

pytestmark = pytest.mark.integration

ENSEMBLE_SIZE = 10


@pytest.fixture(scope="module")
def model_and_subs():
    default_config = MobilityModel.default_config(
        config_params={**config_params, "enable_ethics": True}
    )
    default_inputs = MobilityModel.default_inputs(
        config=default_config, 
        policy_params=policy_params, 
        behavioral_params=behavioral_params
    )
    m = MobilityModel(
        config=default_config,
        inputs=default_inputs,
    )
    subs = evaluate_model(m, overrides=None, size=ENSEMBLE_SIZE)
    return m, subs


def test_all_cases_built(model_and_subs):
    m, _ = model_and_subs
    assert set(m.ethical) == {"elderly", "minors", "income"}


def test_all_cases_expose_canonical_output_maps(model_and_subs):
    m, subs = model_and_subs
    expected_maps = {"baseline_mpi", "modified_mpi", "delta_mpi"}

    for case_id, bundle in m.ethical.items():
        maps = bundle.extract_output_maps(subs)
        assert expected_maps.issubset(maps), f"[{case_id}] missing maps: {expected_maps - set(maps)}"
        assert all("None" in maps[name] for name in expected_maps), f"[{case_id}] missing 'None' gender slice"

        none_maps = {name: maps[name]["None"] for name in expected_maps}
        assert all(len(none_maps[name]) > 0 for name in expected_maps), f"[{case_id}] empty canonical map"
        assert set(none_maps["modified_mpi"]) == set(none_maps["baseline_mpi"]) == set(none_maps["delta_mpi"]), (
            f"[{case_id}] area sets differ across maps"
        )


def test_vulnerability_by_area_populated(model_and_subs):
    m, subs = model_and_subs

    for case_id, bundle in m.ethical.items():
        vuln = {
            area: float(np.asarray(subs.expected_value(idx)).reshape(-1)[0])
            for area, idx in bundle.I_vulnerability_by_area["None"].items()
        }
        if case_id != "income":
            assert len(vuln) > 0, f"[{case_id}] vulnerability_by_area is empty"
