import itertools
import math
import numpy as np
import pandas as pd
from scipy import stats

import plotly.express as px
from plotly import graph_objects as go
import geopandas as gpd

from bdt_mobility.simulation.utils import to_number
from bdt_mobility.simulation.i18n import t
from bdt_mobility.model.axis import ZoneInIndex, ZoneInTimeseriesIndex

TS = np.array([(t - pd.Timestamp('00:00:00')).total_seconds()
               for t in pd.date_range(start='00:00:00', periods=12 * 24, freq='5min')])

AREAS = None

def set_map(areas):
    global AREAS
    AREAS = areas

def roundup(val):
    v = val * 1.4
    l = math.floor(math.log10(v)) if val > 0 else 0.0
    res = math.ceil(v / 10 ** l) * 10 ** l
    return res

def distribution(field, size=10000, num=100):
    field = np.asarray(field)
    if field.ndim == 1:
        field = field[np.newaxis, :]
    else:
        field = np.squeeze(field)
        if field.ndim == 1:
            field = field[np.newaxis, :]
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)

field_color = ['rgba(255,255,255,0.0)',
               'rgba(195,95,100,0.8)',
               'rgba(165,15,21,1.0)',
               'rgba(195,95,100,0.8)',
               'rgba(255,255,255,0.0)']

def plot_field_graph(field,
                     horizontal_label,
                     vertical_label,
                     *,
                     vertical_unit=None,
                     vertical_size=None,
                     vertical_transformer=None,
                     reference_line=None):
    field = np.asarray(field).squeeze()
    if field.ndim == 0:
        field = np.full(TS.shape, field, dtype=float)
    if field.ndim == 1:
        field = field[np.newaxis, :]
    if field.ndim != 2 or field.shape[-1] != TS.size:
        raise ValueError(f"Expected field shape (samples, {TS.size}), got {field.shape}")

    if vertical_size is None:
        vertical_size = roundup(np.max(field))
    if vertical_unit is None:
        vertical_unit = ""
    if vertical_transformer is None:
        vertical_transformer = lambda x: x

    dist = distribution(field, vertical_size, 100)

    fig = go.Figure()
    fig.add_heatmap(
        z=dist.T,
        x=pd.date_range(start='00:00:00', periods=12 * 24 +1, freq='5min'),
        y=np.linspace(0, vertical_transformer(vertical_size), 100 + 1),
        hoverinfo='skip',
        colorscale=field_color,
        autocolorscale=False,
        colorbar=dict(
            bordercolor='black',
            tickfont=dict(color='black'),
            tickmode='array',
            tickvals=[0.0,0.25,0.5,0.75,1.0],
            ticktext=['0%','25%','50%','75%','100%'],
        ),
    )
    if reference_line is not None:
        fig.add_trace(go.Scatter(x=pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                                 y=vertical_transformer(reference_line),
                                 mode='lines',
                                 line=dict(color='black', width=1, dash='dash'),
                                 name=t("plot_reference"),
                                 hovertemplate='<br>%{y:.0f} '+vertical_unit))
    fig.add_trace(go.Scatter(x=pd.date_range(start='00:00:00', periods=12 * 24, freq='5min'),
                             y=vertical_transformer(field.mean(axis=0)),
                             mode='lines',
                             line=dict(color='black', width=2),
                             name=t("plot_modified"),
                             hovertemplate='<br>%{y:.0f} '+vertical_unit))
    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        hoverlabel=dict(
            bordercolor='gray',
            font=dict(color='black'),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bordercolor='gray',
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(
                text=horizontal_label,
                font=dict(color='black'),
            ),
            tickformat='%H:%M:%S',
            ticks='outside',
            tickson='labels',
            tickcolor='black',
            tickangle=45,
            tickfont=dict(color='black'),
            linecolor='black',
            linewidth=1,
            mirror=True,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(
                text=vertical_label + " [" + vertical_unit + "]" if vertical_unit else vertical_label,
                font=dict(color='black'),
            ),
            ticks='outside',
            tickson='labels',
            tickcolor='black',
            tickfont=dict(color='black'),
            linecolor='black',
            linewidth=1,
            mirror=True,
            showgrid=True,
        ),
    )
    return fig

def plot_multifield_graph(evals, index, zones,
                          horizontal_label,
                          vertical_label,
                          *,
                          vertical_unit=None,
                          vertical_size=None,
                          vertical_transformer=None,
                          reference_index=None):
    if isinstance(index, ZoneInTimeseriesIndex):
        values = np.asarray(evals.expected_value(index))
        field = values[list(zones)].sum(axis=0)
        if reference_index is not None:
            reference_line = np.asarray(evals.expected_value(reference_index))[list(zones)].sum(axis=0)
        else:
            reference_line = None
    else:
        field = sum([np.asarray(evals.expected_value(index[z])) for z in zones])
        if reference_index is not None:
            reference_line = sum([np.asarray(evals.expected_value(reference_index[z])) for z in zones])
        else:
            reference_line = None
    return plot_field_graph(field, horizontal_label, vertical_label,
                            vertical_unit=vertical_unit,
                            vertical_size=vertical_size,
                            vertical_transformer=vertical_transformer,
                            reference_line=reference_line)

def plot_map_graph(evals, index, label, *, unit=None, time=None, range=None, function=sum):
    if isinstance(index, (ZoneInIndex, ZoneInTimeseriesIndex)):
        zone_ids = AREAS["index"].to_numpy()
        zonal_values = np.asarray(evals.expected_value(index))
        values = dict(zip(zone_ids, zonal_values))
    else:
        values = {
            key: np.asarray(evals.expected_value(value)).squeeze()
            for key, value in index.items()
        }
    if time is None:
        df_zones = pd.DataFrame.from_dict(
            {k: v if np.asarray(v).ndim == 0 else function(v) for k, v in values.items()},
            orient='index')
    else:
        if type(time) is tuple:
            (start,end) = time
        else:
            (start,end) = (time, time+pd.Timedelta(seconds=15*60))
        df_zones = pd.DataFrame.from_dict(
            {k: function(itertools.compress(v,
                                            (to_number(start) <= TS) &
                                            (TS < to_number(end)))) for k, v in values.items()},
            orient='index')

    df_zones.index.name = 'index'
    df_zones.rename(columns={0: 'value'}, inplace=True)
    df_zones = df_zones.merge(AREAS, on='index', how='left')
    df_zones.set_index("index", inplace=True)
    gdf_zones = gpd.GeoDataFrame(df_zones, geometry='geometry')
    gdf_zones.set_crs(epsg=4326, inplace=True, allow_override=True)
    # TODO: Manage range
    fig = px.choropleth_map(
        gdf_zones,
        geojson=gdf_zones.geometry,
        locations=gdf_zones.index,
        color="value",
        hover_name="name",
        hover_data={"value": ":d"},
        labels={'id_zone': t("plot_zone_code"), 'value': f'{label} ({unit})' if unit is not None else label},
        center={"lat": 44.492, "lon": 11.341},
        map_style="carto-positron",
        zoom=11.5,
        opacity=0.5,
        color_continuous_scale='Reds',
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bordercolor='gray',
            font=dict(color='black'),
        ),
        coloraxis_colorbar_title_text=None,
    )
    return fig

