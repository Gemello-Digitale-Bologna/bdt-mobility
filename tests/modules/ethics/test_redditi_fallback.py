import pandas as pd
import pytest

from bdt_mobility.modules.ethics.preparation.step_01_redditi_mediani import (
    MIN_SAMPLE,
    TOTAL_VALUE,
    process_incomes,
)

pytestmark = pytest.mark.unit

AREAS = pd.DataFrame({
    "codice_area_statistica": [1, 2, 3],
    "area_statistica": ["ALFA", "BETA", "GAMMA"],
    "cod_zona": ["A", "A", "B"],
    "zona": ["ZONA A", "ZONA A", "ZONA B"],
    "cod_quar": [11, 11, 11],
    "quartiere": ["Q11", "Q11", "Q11"],
})


def source(codes, code_column, median, mean, n, sesso=None, eta=None):
    return pd.DataFrame({
        code_column: codes,
        "anno": [2023] * len(codes),
        "cittad_f": [None] * len(codes),
        "sesso": [sesso] * len(codes),
        "eta_fgc": [eta] * len(codes),
        "reddito_irpv_Median": median,
        "reddito_irpv_Mean": mean,
        "reddito_irpv_N": n,
    })


def run(area, zona, quart):
    return process_incomes(area, zona, quart, AREAS)


def test_prefers_the_area_then_zona_then_quartiere():
    # Only area 1 has its own value; area 2 falls back to its zona, area 3 has
    # no zona row and falls back to the quartiere.
    area = source([1], "COD_AREAST", [1000.0], [1100.0], [200])
    zona = source(["A"], "CODZONA", [2000.0], [2100.0], [500])
    quart = source([11], "NUMQUART", [3000.0], [3100.0], [900])

    df = run(area, zona, quart)
    total = df[df["categoria_demografica"] == "Totale - Totale - Totale"]
    by_area = total.set_index("codice_area_statistica")

    assert by_area.loc[1, "fonte_reddito_mediano"] == "area_statistica"
    assert by_area.loc[2, "fonte_reddito_mediano"] == "zona"
    assert by_area.loc[3, "fonte_reddito_mediano"] == "quartiere"
    assert by_area.loc[1, "reddito_mediano"] == 1000.0
    assert by_area.loc[2, "reddito_mediano"] == 2000.0
    assert by_area.loc[3, "reddito_mediano"] == 3000.0


def test_mean_and_contributors_travel_with_the_median():
    area = source([1], "COD_AREAST", [1000.0], [1100.0], [200])
    empty = source([], "CODZONA", [], [], [])
    df = run(area, empty, source([], "NUMQUART", [], [], []))

    row = df[(df["codice_area_statistica"] == 1)
             & (df["categoria_demografica"] == "Totale - Totale - Totale")].iloc[0]
    assert row["reddito_medio"] == 1100.0
    assert row["contribuenti"] == 200


def test_samples_below_the_minimum_are_dropped():
    area = source([1], "COD_AREAST", [1000.0], [1100.0], [MIN_SAMPLE - 1])
    zona = source(["A"], "CODZONA", [2000.0], [2100.0], [MIN_SAMPLE])

    df = run(area, zona, source([], "NUMQUART", [], [], []))
    row = df[(df["codice_area_statistica"] == 1)
             & (df["categoria_demografica"] == "Totale - Totale - Totale")].iloc[0]
    assert row["fonte_reddito_mediano"] == "zona"
    assert row["reddito_mediano"] == 2000.0


def test_falls_back_to_a_more_general_category():
    # Only the sex total exists, so the male row borrows it.
    area = source([1], "COD_AREAST", [1000.0], [1100.0], [200])
    df = run(area, source([], "CODZONA", [], [], []), source([], "NUMQUART", [], [], []))

    row = df[(df["codice_area_statistica"] == 1)
             & (df["sesso"] == "Maschile")
             & (df["fascia_anni"] == TOTAL_VALUE)
             & (df["cittadinanza"] == TOTAL_VALUE)].iloc[0]
    assert row["reddito_mediano"] == 1000.0
    assert row["livello_fallback_categoria"] > 0


def test_missing_everywhere_is_marked_not_available():
    nothing = source([], "COD_AREAST", [], [], [])
    df = process_incomes(
        source([1], "COD_AREAST", [1000.0], [1100.0], [200]),
        source([], "CODZONA", [], [], []),
        source([], "NUMQUART", [], [], []),
        AREAS,
    )
    # Areas 2 and 3 share no territory with area 1's row, so they stay empty.
    missing = df[df["codice_area_statistica"] == 3]
    assert (missing["fonte_reddito_mediano"] == "non_disponibile").all()
    assert missing["reddito_mediano"].isna().all()
