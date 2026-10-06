from __future__ import annotations

import numpy as np
import pytest

_RANDOM_SEED = 42
_ENSEMBLE_SIZE = 20


@pytest.fixture(scope="session")
def _model_run():
    from bdt_mobility.model import MobilityModel, compute_kpis, evaluate_model
    from bdt_mobility.model.config_vars import (
        behavioral_params,
        config_params,
        policy_params,
    )

    np.random.seed(_RANDOM_SEED)
    config = MobilityModel.default_config(config_params={**config_params, "enable_ethics": True})
    inputs = MobilityModel.default_inputs(
        config=config,
        policy_params=policy_params,
        behavioral_params=behavioral_params,
    )
    m = MobilityModel(
        config=config,
        inputs=inputs,
    )
    evals = evaluate_model(m, overrides=None, size=_ENSEMBLE_SIZE, rng=_RANDOM_SEED)
    kpis = compute_kpis(m, evals)
    return m, evals, kpis


@pytest.fixture(scope="session")
def m(_model_run):
    return _model_run[0]


@pytest.fixture(scope="session")
def evals(_model_run):
    return _model_run[1]


@pytest.fixture(scope="session")
def kpis(_model_run):
    return _model_run[2]
