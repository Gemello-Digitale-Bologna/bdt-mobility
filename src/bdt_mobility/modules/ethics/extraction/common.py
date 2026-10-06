"""Helpers shared by the ethics extraction steps."""
from __future__ import annotations

import pandas as pd


def map_to_as(
    df: pd.DataFrame,
    df_areas: pd.DataFrame,
    target_columns: list[str],
    geo_column: str,
) -> pd.DataFrame:
    """Fan values given by zona or quartiere out to the statistical areas inside."""
    df_mapped = pd.merge(df, df_areas[["codice_area_statistica", geo_column]], on=geo_column)
    return df_mapped[["anno", "codice_area_statistica"] + target_columns]


def convert_percent_to_ratio(df: pd.DataFrame, exclude: tuple[str, ...] = ("rmpe_fam",)) -> pd.DataFrame:
    """0-100 percentages to 0-1 ratios. ``exclude`` names float columns to skip."""
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == "float64" and column not in exclude:
            df[column] = df[column] / 100
    return df
