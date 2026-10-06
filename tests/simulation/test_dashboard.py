"""Smoke tests for the simulation dashboard.

Uses Streamlit's AppTest framework — the script only executes when at.run()
is called inside the test body, so there is no collection-time data loading.

Requires real data (same as the simulation model). Run with:
    uv run pytest -m simulation
"""
from __future__ import annotations

import importlib.util

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.simulation

_TIMEOUT = 120  # seconds — covers model init + 100-sample evaluate()


def _app() -> AppTest:
    spec = importlib.util.find_spec("bdt_mobility.simulation.dashboard")
    return AppTest.from_file(spec.origin, default_timeout=_TIMEOUT)


@pytest.fixture(scope="session")
def app():
    """Runs the dashboard once and returns the AppTest instance."""
    at = _app()
    at.run()
    return at


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_dashboard_loads(app):
    assert not app.exception


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_dashboard_title(app):
    # Title should be present (language-agnostic check)
    assert len(app.title) > 0


def test_dashboard_initial_scenario_header(app):
    # Scenario header should be present
    assert len(app.header) > 0


def test_dashboard_simulation_buttons_present(app):
    # Should have at least 3 buttons (save, load, compare)
    assert len(app.button) >= 3


def test_dashboard_parameter_sliders_rendered(app):
    assert len(app.slider) > 0


# ---------------------------------------------------------------------------
# Navigation: Simulazione → Comparazione
# ---------------------------------------------------------------------------

def test_dashboard_navigation_to_comparison():
    at = _app()
    at.run()
    assert not at.exception

    # Click the compare button (should be the third button)
    compare_button = at.button[2]
    compare_button.click()
    at.run()

    assert not at.exception
    assert at.session_state["__selected__"] == "Comparazione"
    # Should now have a back button
    assert len(at.button) >= 1
