"""Regression tests for MobilityModel.

Expected values were captured with RANDOM_SEED=42, ENSEMBLE_SIZE=20, using the
default model configuration (policy='cordon-based', decision_strategy='parallel',
time_shift_strategy='flexible', modal_shift_option='lpt') with the ethics module enabled.

The expected values depend on both the model and the extracted data. Extraction is
deterministic, so re-running it on the same inputs reproduces them; when the model logic,
the extraction code or the input dataitems change intentionally, refresh them by running:
    uv run python tests/simulation/test_model.py --capture
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.simulation

# Representative zones with non-zero traffic data (some zones have none, e.g. 29, 30, 31).
SPOT_CHECK_ZONES = [0, 2, 3, 4]


def _zone_values(evals, index, zone):
    values = np.asarray(evals.expected_value(index))
    return values[zone, ...]


# ---------------------------------------------------------------------------
# KPI regression (snapshot)
# ---------------------------------------------------------------------------

EXPECTED_KPIS = {
    "Base inflow [veh/day]": 178696,
    "Mode-shifted inflow [veh/day]": 11682,
    "Inflow difference [veh/day]": -37033,
    "Lost inflow [veh/day]": 25341,
    "Modified inflow [veh/day]": 141663,
    "Time-shifted inflow [veh/day]": 5841,
    "Base traffic [max veh]": 15095,
    "Modified traffic [max veh]": 13805,
    "Traffic difference [max veh]": -1290,
    "Paying inflow [veh/day]": 72861,
    "Average cost [€/day]": 3.91,
    "Collected fees [€/day]": 285121,
    "Base emissions [NOx gr/day]": 1072598,
    "Emissions [NOx gr/day]": 997417,
    "Emissions difference [NOx gr/day]": -75181,
}


def test_kpi_values(_model_run):
    _, _, kpis = _model_run
    for name, expected in EXPECTED_KPIS.items():
        actual = kpis[name]
        if isinstance(expected, float):
            tol = abs(expected) * 0.01
            ok = abs(actual - expected) <= tol
        else:
            ok = abs(actual - expected) <= max(1, abs(expected) // 100)
        assert ok, f"KPI '{name}': expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Invariants: non-negativity
# ---------------------------------------------------------------------------


def test_global_timeseries_non_negative(_model_run):
    m, evals, _ = _model_run
    for name in [
        "I_traffic",
        "I_modified_traffic",
        "I_modified_inflow",
        "TS_inflow",
        "TS_starting",
    ]:
        arr = evals[getattr(m, name)]
        assert arr.min() >= -1e-6, f"{name} has negative values (min={arr.min():.4f})"


def test_zone_timeseries_non_negative(_model_run):
    m, evals, _ = _model_run
    for zone_position, zone in enumerate(m.in_zones):
        for idx, label in [
            (m.TS_inflow_zone, "TS_inflow_zone"),
            (m.I_modified_inflow_zone, "I_modified_inflow_zone"),
            (m.TS_traffic_zone, "TS_traffic_zone"),
            (m.I_modified_traffic_zone, "I_modified_traffic_zone"),
        ]:
            arr = _zone_values(evals, idx, zone_position)
            assert arr.min() >= -1e-6, (
                f"zone {zone} {label} has negative values (min={arr.min():.4f})"
            )


# ---------------------------------------------------------------------------
# Invariants: policy reduces inflow and traffic
# ---------------------------------------------------------------------------


def test_modified_inflow_le_base(_model_run):
    _, _, kpis = _model_run
    assert kpis["Modified inflow [veh/day]"] <= kpis["Base inflow [veh/day]"], (
        "Modified inflow exceeds base inflow"
    )


def test_modified_traffic_le_base(_model_run):
    _, _, kpis = _model_run
    assert kpis["Modified traffic [max veh]"] <= kpis["Base traffic [max veh]"], (
        "Modified peak traffic exceeds base peak traffic"
    )


def test_zone_modified_inflow_le_base(_model_run):
    m, evals, _ = _model_run
    for zone_position, zone in enumerate(m.in_zones):
        base = float(_zone_values(evals, m.TS_inflow_zone, zone_position).mean())
        mod = float(_zone_values(evals, m.I_modified_inflow_zone, zone_position).mean())
        assert mod <= base + 1e-6, (
            f"zone {zone}: modified mean inflow ({mod:.4f}) > base ({base:.4f})"
        )


# ---------------------------------------------------------------------------
# Invariants: timeseries shape
# ---------------------------------------------------------------------------


def test_global_timeseries_shape(_model_run):
    m, evals, _ = _model_run
    T = 288  # 5-min intervals over 24h
    for name in [
        "I_traffic",
        "I_modified_traffic",
        "I_modified_inflow",
        "TS_inflow",
        "TS_starting",
    ]:
        arr = evals[getattr(m, name)]
        assert arr.shape[-1] == T, f"{name}: expected last dim {T}, got {arr.shape}"


def test_zone_timeseries_shape(_model_run):
    m, evals, _ = _model_run
    T = 288
    for zone_position, zone in enumerate(m.in_zones):
        for idx, label in [
            (m.TS_inflow_zone, "TS_inflow_zone"),
            (m.TS_traffic_zone, "TS_traffic_zone"),
        ]:
            arr = _zone_values(evals, idx, zone_position)
            assert arr.shape[-1] == T, (
                f"zone {zone} {label}: expected last dim {T}, got {arr.shape}"
            )


# ---------------------------------------------------------------------------
# Invariants: zone inflow composition (inside + outside == total)
# ---------------------------------------------------------------------------


def test_zone_inflow_composition(_model_run):
    m, evals, _ = _model_run
    for zone_position, zone in enumerate(m.in_zones):
        inside = _zone_values(evals, m.TS_inflow_zone_from_inside, zone_position)
        outside = _zone_values(evals, m.TS_inflow_zone_from_outside, zone_position)
        total = _zone_values(evals, m.TS_inflow_zone, zone_position)
        diff = np.max(np.abs((inside + outside) - total))
        assert diff < 1e-6, f"zone {zone}: inflow composition mismatch (max_diff={diff:.2e})"


# ---------------------------------------------------------------------------
# Invariants: zone traffic peaks during daytime (6am–10pm)
# ---------------------------------------------------------------------------


def test_zone_traffic_peak_in_daytime(_model_run):
    m, evals, _ = _model_run
    step_6am = 72  # 6h * 12 steps/h
    step_10pm = 264  # 22h * 12 steps/h
    for zone_position, zone in enumerate(m.in_zones):
        ts = _zone_values(evals, m.TS_traffic_zone, zone_position)
        if ts.max() < 1e-6:
            continue  # zone has no traffic data
        peak = int(np.argmax(ts))
        assert step_6am <= peak <= step_10pm, (
            f"zone {zone}: base traffic peak at step {peak} "
            f"({peak / 12:.1f}h) is outside 6am–10pm window"
        )


# ---------------------------------------------------------------------------
# Zone-specific regression (spot checks)
# ---------------------------------------------------------------------------

# (base_inflow, mod_inflow, base_traffic, mod_traffic) — mean over ensemble
EXPECTED_ZONE = {
    0: (195.10, 188.14, 4.34, 4.09),
    2: (24.38, 23.12, 0.36, 0.34),
    3: (240.04, 216.59, 2.02, 1.91),
    4: (9.52, 8.72, 1.24, 1.15),
}


def test_zone_spot_checks(_model_run):
    m, evals, _ = _model_run
    for zone in SPOT_CHECK_ZONES:
        zone_position = list(m.in_zones).index(zone)
        expected = EXPECTED_ZONE[zone]
        actuals = (
            float(_zone_values(evals, m.TS_inflow_zone, zone_position).mean()),
            float(_zone_values(evals, m.I_modified_inflow_zone, zone_position).mean()),
            float(_zone_values(evals, m.TS_traffic_zone, zone_position).mean()),
            float(_zone_values(evals, m.I_modified_traffic_zone, zone_position).mean()),
        )
        labels = ("base_inflow", "mod_inflow", "base_traffic", "mod_traffic")
        for label, actual, exp in zip(labels, actuals, expected):
            tol = max(0.01, abs(exp) * 0.01)
            assert abs(actual - exp) <= tol, (
                f"zone {zone} {label}: expected {exp:.2f}, got {actual:.2f}"
            )


# ---------------------------------------------------------------------------
# Capture helper — run directly to refresh expected values:
#   uv run python tests/simulation/test_model.py --capture
# ---------------------------------------------------------------------------


def _capture_expected(m, evals, kpis):
    print("=== EXPECTED_KPIS ===")
    for k, v in kpis.items():
        print(f"    {k!r}: {v!r},")
    print()
    print("=== EXPECTED_ZONE ===")
    for zone in SPOT_CHECK_ZONES:
        zone_position = list(m.in_zones).index(zone)
        vals = (
            float(_zone_values(evals, m.TS_inflow_zone, zone_position).mean()),
            float(_zone_values(evals, m.I_modified_inflow_zone, zone_position).mean()),
            float(_zone_values(evals, m.TS_traffic_zone, zone_position).mean()),
            float(_zone_values(evals, m.I_modified_traffic_zone, zone_position).mean()),
        )
        print(f"    {zone}: ({vals[0]:.2f}, {vals[1]:.2f}, {vals[2]:.2f}, {vals[3]:.2f}),")


# ---------------------------------------------------------------------------
# Ethics module gating
# ---------------------------------------------------------------------------


def test_ethics_disabled_by_default():
    from bdt_mobility.model import MobilityModel
    from bdt_mobility.model.config_vars import behavioral_params, config_params, policy_params

    config = MobilityModel.default_config(config_params=config_params)
    inputs = MobilityModel.default_inputs(
        config=config,
        policy_params=policy_params,
        behavioral_params=behavioral_params,
    )
    m = MobilityModel(config=config, inputs=inputs)

    assert config.ethical_projection is None
    assert m.ethical == {}
    assert m.expose.ethical_elderly_none_delta_mpi == {}


def test_ethics_enabled(m):
    assert set(m.ethical) == {"elderly", "minors", "income"}
    assert len(m.expose.ethical_elderly_none_delta_mpi) > 0


if __name__ == "__main__":
    if "--capture" not in sys.argv:
        print("Usage: uv run python tests/simulation/test_model.py --capture", file=sys.stderr)
        sys.exit(1)

    from bdt_mobility.model import MobilityModel, compute_kpis, evaluate_model
    from bdt_mobility.model.config_vars import behavioral_params, policy_params, config_params

    _RANDOM_SEED = 42
    _ENSEMBLE_SIZE = 20

    np.random.seed(_RANDOM_SEED)
    _default_config = MobilityModel.default_config(
        config_params={**config_params, "enable_ethics": True}
    )
    _default_inputs = MobilityModel.default_inputs(
        config=_default_config,
        policy_params=policy_params,
        behavioral_params=behavioral_params
    )
    _m = MobilityModel(
        config=_default_config,
        inputs=_default_inputs,
    )
    _evals = evaluate_model(_m, overrides=None, size=_ENSEMBLE_SIZE, rng=_RANDOM_SEED)
    _kpis = compute_kpis(_m, _evals)
    _capture_expected(_m, _evals, _kpis)
