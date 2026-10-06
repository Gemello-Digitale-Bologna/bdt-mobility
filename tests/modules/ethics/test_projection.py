from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

from bdt_mobility.modules.ethics.model.projection import (
    _build_projection_map,
    build_projection_map_from_zone_ids,
    project_float_map,
    project_index_map,
)
from civic_digital_twins.dt_model import Index


def test_build_projection_map_sums_duplicate_weights():
    df = pd.DataFrame(
        {
            "codice_area_statistica": [101, 101, 102, 103],
            "zone_index": [7, 7, 7, 9],
            "weight": [0.25, 0.15, 0.60, 1.00],
        }
    )

    projection = _build_projection_map(
        df,
        area_col="codice_area_statistica",
        zone_col="zone_index",
        weight_col="weight",
    )

    assert projection == {
        7: ((101, 0.40), (102, 0.60)),
        9: ((103, 1.00),),
    }


def test_build_projection_map_from_zone_ids_translates_to_simulation_indices():
    crosswalk = pd.DataFrame(
        {
            "codice_area_statistica": [101, 102, 103],
            "id_zone": [14, 14, 15],
            "perc_area_intersect": [0.40, 0.60, 1.00],
        }
    )
    od_zones = pd.DataFrame(
        {
            "id_zone": [14, 15],
            "index": [7, 9],
        }
    )

    projection = build_projection_map_from_zone_ids(crosswalk, od_zones)

    assert projection == {
        7: ((101, 0.40), (102, 0.60)),
        9: ((103, 1.00),),
    }


def test_build_projection_map_from_zone_ids_rejects_unmapped_zone_ids():
    crosswalk = pd.DataFrame(
        {
            "codice_area_statistica": [101],
            "id_zone": [999],
            "perc_area_intersect": [1.00],
        }
    )
    od_zones = pd.DataFrame(
        {
            "id_zone": [14],
            "index": [7],
        }
    )

    try:
        build_projection_map_from_zone_ids(crosswalk, od_zones)
    except ValueError as exc:
        assert "not present in the simulation zones" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unmapped raw zone ids.")


def test_build_projection_map_from_zone_ids_keeps_last_duplicate_zone_mapping():
    crosswalk = pd.DataFrame(
        {
            "codice_area_statistica": [101],
            "id_zone": [14],
            "perc_area_intersect": [1.00],
        }
    )
    od_zones = pd.DataFrame(
        {
            "id_zone": [14, 14],
            "index": [7, 9],
        }
    )

    projection = build_projection_map_from_zone_ids(crosswalk, od_zones)

    assert projection == {
        9: ((101, 1.00),),
    }


def test_project_float_map_skips_missing_areas():
    projection = {
        7: ((101, 0.40), (102, 0.60)),
        9: ((103, 1.00),),
        11: ((999, 0.50),),
    }

    projected = project_float_map(
        {
            101: 10.0,
            102: 30.0,
        },
        projection,
    )

    assert projected == {
        7: 22.0,
    }


def test_project_index_map_returns_zone_keyed_indexes():
    projection = {
        7: ((101, 0.40), (102, 0.60)),
        11: ((999, 1.00),),
    }

    projected = project_index_map(
        {
            101: Index("I_area_value[101]", 10.0),
            102: Index("I_area_value[102]", 30.0),
        },
        projection,
        index_name="I_projected_value",
    )

    assert set(projected) == {7}
    assert isinstance(projected[7], Index)
    assert projected[7].name == "I_projected_value[7]"


def test_build_projection_map_rejects_negative_weights():
    df = pd.DataFrame(
        {
            "codice_area_statistica": [101],
            "zone_index": [7],
            "weight": [-0.10],
        }
    )

    try:
        _build_projection_map(
            df,
            area_col="codice_area_statistica",
            zone_col="zone_index",
            weight_col="weight",
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative projection weights.")
