import geopandas as gpd
import plotly.express as px

DEFAULT_CENTER = {"lat": 44.4949, "lon": 11.3426}
DEFAULT_MAP_STYLE = "carto-positron"
DEFAULT_ZOOM = 11.3


# last-point map plotting
def plot_pums_zones_choropleth(
    gdf: gpd.GeoDataFrame,
    *,
    value_col: str,
    hover_name_col: str = "name",
    hover_cols: list[str] | None = None,
    label: str | None = None,
    color_scale: str = "Reds",
    color_min: float | None = None,
    color_max: float | None = None,
):
    if value_col not in gdf.columns:
        raise KeyError(f"Expected value column '{value_col}' in map dataframe.")
    if hover_name_col not in gdf.columns:
        raise KeyError(f"Expected hover name column '{hover_name_col}' in map dataframe.")

    hover_cols = hover_cols or []
    hover_data = {col: True for col in hover_cols}
    hover_data[value_col] = True

    range_color = None
    if color_min is not None and color_max is not None:
        range_color = (color_min, color_max)

    fig = px.choropleth_map(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color=value_col,
        hover_name=hover_name_col,
        hover_data=hover_data,
        labels={value_col: label or value_col},
        center=DEFAULT_CENTER,
        map_style=DEFAULT_MAP_STYLE,
        zoom=DEFAULT_ZOOM,
        opacity=0.55,
        color_continuous_scale=color_scale,
        range_color=range_color,
    )
    fig.update_layout(
        autosize=False,
        width=850,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(bordercolor="gray", font=dict(color="black")),
        coloraxis_colorbar_title_text=None,
    )
    return fig

def plot_mpi_map(
    values_by_zone: dict[int, float],
    *,
    label: str,
    column: str,
    od_zones: gpd.GeoDataFrame,
    year: int = 2023,
    year_label: str = "anno",
    color_min: float | None = None,
    color_max: float | None = None,
):

    # use "index" to navigate gdf
    gdf_ready = od_zones.copy().set_index("index")

    # check for zones in incoming dict against gdf
    invalid_zone_ids = sorted({int(zone) for zone in values_by_zone if int(zone) not in gdf_ready.index})
    if invalid_zone_ids:
        raise ValueError(
            "plot_mpi_map expects simulation-zone keyed values. "
            f"Unexpected zone keys: {invalid_zone_ids[:10]}"
        )

    gdf_ready[column] = gdf_ready.index.map(values_by_zone)
    gdf_ready[year_label] = year

    return plot_pums_zones_choropleth(
        gdf_ready,
        value_col=column,
        hover_cols=["id_zone", year_label],
        label=label,
        color_min=color_min,
        color_max=color_max,
    )
