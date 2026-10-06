import geopandas as gpd
import pandas as pd
import polars as pl


def OD_flows(
    df_flows: pd.DataFrame,
    df_shapes: pd.DataFrame
) -> pd.DataFrame:
    """
    Filters the OD flows by merging them with shape data and removing mismatched flows.

    Args:
        `df_flows (pd.DataFrame)`: DataFrame containing the OD flows.
        `df_shapes (pd.DataFrame)`: DataFrame containing shape information.

    Returns:
        `pd.DataFrame`: Processed DataFrame with updated OD flows.
    """
    df_flows = df_flows.merge(
            df_shapes.rename({'id': 'from'}, axis=1),
            on="from",
            how='inner', # if an area has no geometry, then its flow is removed
        ).merge(
            df_shapes.rename({'id': 'to'}, axis=1),
            on="to",
            how='inner', # if an area has no geometry, then its flow is removed
        )
    return df_flows


def AOI_OD_shapes(
    df_shapes: gpd.GeoDataFrame,
    df_around: gpd.GeoDataFrame,
):
    """
    When considering OD zones as AOI, it is important to remove the overlapping zones,
    that were added to better modeling the Visum simulation, but that have not a physical
    meaning. The function identifies the overlapping zones and removes them as not AOI.

    Moreover, the name of the zones is corrected: when missing name, using the code.

    Args:
        `df_shapes (pd.DataFrame)`: DataFrame containing shape information.
        `df_around (pd.DataFrame)`: DataFrame of the border dividing internal and external zones.

    Returns:
        `list`: Processed DataFrame with updated shape information, both inside and outside.
    """
    # Renaming
    df_shapes.loc[df_shapes['name'].isna(), 'name'] = df_shapes.loc[df_shapes['name'].isna(), 'code']
    df_shapes = df_shapes.rename({"id": "id_zone"}, axis=1)

    # RM overlapping zones
    aoi_inside = _remove_nonpartitioning_zones(df_shapes)
    aoi_inside = gpd.overlay(aoi_inside, df_around, how="intersection", keep_geom_type=True)

    aoi_outside = gpd.overlay(df_shapes, df_around, how="difference", keep_geom_type=True)
    aoi_outside = _remove_overlapping_zones(aoi_outside, overlap_threshold=0.99)
        
    return aoi_inside, aoi_outside


def _remove_nonpartitioning_zones(
        df_shapes: gpd.GeoDataFrame, 
        coverage_threshold: float = 0.5,
        boundary_tolerance: float = 250.0
    ) -> gpd.GeoDataFrame:
    """
    Keeps only the zones that are part of the true partition:
    - Zones whose borders are mostly covered by other zones' borders or the outer perimeter
    - Zones that significantly touch the total area boundary
    
    Removes spurious zones with exposed edges in empty space.
    """
    df = df_shapes.copy().reset_index(drop=True)
    
    # Ensure to take the exterior border
    total_union = df.geometry.unary_union
    if hasattr(total_union, 'exterior'):
        total_boundary = total_union.exterior
    elif hasattr(total_union, 'geoms'):
        from shapely.ops import unary_union as shapely_union
        exteriors = [geom.exterior for geom in total_union.geoms if hasattr(geom, 'exterior')]
        total_boundary = shapely_union(exteriors) if exteriors else total_union.boundary
    else:
        total_boundary = total_union.boundary
    
    # Find the zones to keep
    zones_to_keep = set()
    for i in range(len(df)):
        geom_i = df.geometry.iloc[i]
        boundary_i = geom_i.boundary
        total_boundary_length = boundary_i.length
        
        if total_boundary_length < boundary_tolerance:
            continue
        
        covered_length = 0.0
        # Coverage from the outer perimeter
        try:
            external_intersection = boundary_i.intersection(total_boundary)
            if hasattr(external_intersection, 'length'):
                covered_length += external_intersection.length
            elif hasattr(external_intersection, 'geoms'):
                covered_length += sum(
                    getattr(geom, 'length', 0) for geom in external_intersection.geoms
                    if hasattr(geom, 'length')
                )
        except:
            pass
        
        # Coverage from other zones' borders
        for j in range(len(df)):
            if i != j:
                boundary_j = df.geometry.iloc[j].boundary
                try:
                    intersection = boundary_i.intersection(boundary_j)
                    if hasattr(intersection, 'length'):
                        covered_length += intersection.length
                    elif hasattr(intersection, 'geoms'):
                        covered_length += sum(
                            getattr(geom, 'length', 0) for geom in intersection.geoms
                            if hasattr(geom, 'length')
                        )
                except:
                    continue
        
        coverage_ratio = covered_length / total_boundary_length
        
        if coverage_ratio >= coverage_threshold:
            zones_to_keep.add(i)
   
    return df.loc[list(zones_to_keep)].reset_index(drop=True)


def _remove_overlapping_zones(
        df: gpd.GeoDataFrame, 
        overlap_threshold: float = 0.05
    )-> gpd.GeoDataFrame:
    """
    Removes zones that have more than overlap_threshold of their area overlapping with other zones.
    overlap_threshold is the fraction of the polygon's area.
    """
    df = df.copy()
    df["area"] = df.geometry.area

    to_remove = set()

    for i, geom_i in df.geometry.items():
        if i in to_remove:
            continue

        inter_area_total = 0.0
        for j, geom_j in df.geometry.items():
            if i == j or j in to_remove:
                continue
            inter = geom_i.intersection(geom_j)
            if not inter.is_empty:
                inter_area_total += inter.area

        coverage_area_ratio = inter_area_total / df.at[i, "area"]

        if coverage_area_ratio > overlap_threshold:
            to_remove.add(i)

    return df.drop(index=to_remove).reset_index(drop=True)


def spira_flows(
    df_spiras: pd.DataFrame
)-> pd.DataFrame:
    """
    Removes all records of days where the count was always 0 for the entire day.

    Args:
        `df_spiras` (`pd.DataFrame`): The input DataFrame containing 'spira_unique_id', 'DateTime', and 'count' columns.

    Returns:
        `pd.DataFrame`: A new DataFrame with the records of such days removed.
    """
    df_spiras = df_spiras.assign(Date=lambda x: x['DateTime'].dt.date)
    grouped = df_spiras.groupby(['spira_unique_id', 'Date'])

    all_zero_days = grouped['count'].apply(lambda x: (x == 0).all())
    days_to_drop = all_zero_days[all_zero_days].index

    records_to_keep = ~df_spiras.set_index(['spira_unique_id', 'Date']).index.isin(days_to_drop)
    cleaned_dataframe = df_spiras[records_to_keep].reset_index(drop=True)

    return cleaned_dataframe


def str_norm_expr(col: str | pl.Expr):
    if isinstance(col, str):
        col = pl.col(col)
    return col.str.replace_all(" +|'", "_").str.normalize().str.to_lowercase()


def municipality_norm_expr(col):
    return str_norm_expr(col).str.replace("s.", "san_", literal=True)


def drop_null_cols(df):
    return df.select(
        df.select(pl.all().is_null().all())
        .unpivot()
        .filter(~pl.col("value"))
        .get_column("variable")
        .to_list()
    )

def get_filtered_years(df,year_column, min_year, max_year):
    """function that filters a df for the given time intervals
    df: dataframe
    year_column: string with year column name
    min_yera= min year value
    max_year= max year value"""
    df = df[df[year_column]>=min_year]
    df = df[df[year_column]<=max_year]
    return df

def clean_name_as(df, colum_name):
    """Function that aligns names of statistical area zones between different datasets
    S. Viola -> Santa Viola
    S. → San"""
    df[colum_name] = df[colum_name].str.replace(r"\bS\. Viola\b", "Santa Viola", regex=True)
    df[colum_name] = df[colum_name].str.replace(r"\bS\.\s*", "San ", regex=True)
    return df