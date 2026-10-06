"""Integration test for the full extraction pipeline (incl. OD routing).

Run with:
    uv run pytest -m integration
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).parents[2]
_DATA_TAG = os.environ.get("BDT_DATA_TAG", "san_felice")
_SOURCE_DATA = Path(os.environ.get("BDT_DATA_ROOT", str(_REPO_ROOT))) / "data" / _DATA_TAG

from bdt_mobility.extraction.pipeline import PIPELINE

_STEPS = [step.__name__ for step in PIPELINE]

_EXPECTED_OUTPUTS = [
    "total_flows_from_od.parquet",
    "total_starts_from_od.parquet",
    "inflow_estimated.parquet",
    "outflow_estimated.parquet",
    "pt_metrics_per_zone.parquet",
    "cars_joint_by_comune.parquet",
    "euro_class_split.parquet",
    "quality_of_life_survey.parquet",
]

_EXPECTED_SIMULATOR_OUTPUTS = [
    "nodes.csv",
    "edges.csv",
    "inflow_per_zone_from_simulator.parquet",
    "traffic_per_zone_from_simulator.parquet",
    "starting_per_zone_from_simulator.parquet",
    "metrics_per_road_from_simulator.parquet",
    "travel_distance_from_simulator.parquet",
    "travel_time_from_simulator.parquet",
    "non_linear_interpolation_params.parquet",
]


@pytest.fixture(scope="session")
def pipeline_data_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extraction")
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True)

    return tmp


def test_extraction_pipeline(pipeline_data_dir):
    env = {
        **os.environ,
        "BDT_DATA_ROOT": str(pipeline_data_dir),
        "BDT_DATA_TAG": _DATA_TAG,
        "BDT_LOG_LEVEL": "WARNING",
    }

    for step in _STEPS:
        result = subprocess.run(
            [sys.executable, "-m", step],
            env=env,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"Step {step} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    outputs = pipeline_data_dir / "data" / _DATA_TAG / "analysis_outputs"
    for name in _EXPECTED_OUTPUTS:
        assert (outputs / name).exists(), f"Expected output missing: {name}"
    simulator = outputs / "traffic_simulator"
    for name in _EXPECTED_SIMULATOR_OUTPUTS:
        assert (simulator / name).exists(), f"Expected simulator output missing: {name}"
