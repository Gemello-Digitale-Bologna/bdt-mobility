from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from bdt_mobility.utils.paths import PathRegistry


# NOTE: this can be a pre-computed parquet in analysis_outputs/ (for consistency)
# right now: derived at runtime from the raw demographics file

def load_population_by_area() -> pd.DataFrame:

    path = PathRegistry.FRAGILITIES_DATA / "popolazione_residente.parquet"
    df = pd.read_parquet(path)
    df["anno"] = df["anno"].astype(str).str[:4]
    df["codice_area_statistica"] = df["codice_area_statistica"].astype(str)
    return (
        df.groupby(["codice_area_statistica", "anno"], as_index=False)["residenti"]
        .sum()
    )


def compute_reference_values(
    df: pd.DataFrame,
    df_population: pd.DataFrame,
    indicator_cols: list[str],
    *,
    ref_year: str,
    geo_col: str = "codice_area_statistica",
    year_col: str = "anno",
    people_col: str = "residenti",
) -> dict[str, float]:

    df_ref = df[df[year_col] == ref_year].copy()
    df_ref[geo_col] = df_ref[geo_col].astype(str)

    df_pop_ref = df_population[df_population[year_col] == ref_year].copy()
    df_pop_ref[geo_col] = df_pop_ref[geo_col].astype(str)

    df_merged = pd.merge(df_ref, df_pop_ref[[geo_col, people_col]], on=geo_col, how="left")
    df_merged[people_col] = df_merged[people_col].fillna(0)
    total_residents = df_merged[people_col].sum()

    ref_values: dict[str, float] = {}
    for col in indicator_cols:
        if col not in df_merged.columns:
            continue
        ref_values[col] = float(
            (df_merged[col] * df_merged[people_col]).sum() / total_residents
        )

    return ref_values


def normalize_indicators(
    df: pd.DataFrame,
    ref_values: dict[str, float],
    indicator_cols: list[str],
    *,
    minmax_years: tuple[str, str],
    year_col: str = "anno",
    inverse_cols: tuple[str, ...] = (),
) -> pd.DataFrame:

    df_out = df.copy()

    mask = (df[year_col] >= minmax_years[0]) & (df[year_col] <= minmax_years[1])
    df_range = df.loc[mask, indicator_cols]

    val_min = df_range.min()
    val_max = df_range.max()

    for col in indicator_cols:
        if col not in ref_values:
            continue

        ref = ref_values[col]
        denom = float(val_max[col] - val_min[col])

        if denom == 0.0:
            df_out[col] = 100.0
        elif col in inverse_cols:
            df_out[col] = 100.0 - 100.0 * (df[col] - ref) / denom
        else:
            df_out[col] = 100.0 + 100.0 * (df[col] - ref) / denom

    return df_out
