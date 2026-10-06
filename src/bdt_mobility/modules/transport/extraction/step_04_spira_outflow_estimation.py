from datetime import datetime
import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
from scipy.signal import savgol_filter
import numpy as np
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
import geopandas as gpd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.extraction.platform.outputs import PlatformOutput

from bdt_mobility.utils import data_reader
from bdt_mobility.utils import spatial_utils

MAIN_OUTPUTS = [
    PlatformOutput("outflow_estimated", PathRegistry.OUTFLOW_ESTIMATED, "table"),
]
from bdt_mobility.modules.transport.extraction import spatial_assignment
from bdt_mobility.utils import temporal_utils

def get_outflow_from_spiras(start_date, end_date):

    logging.info("Loading the data")
    # inflow from gate estimation (vehicles/hour)
    inflow_df = pd.read_parquet(PathRegistry.INFLOW_ESTIMATED)

    # Load ZOI
    gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)

    # Load road data
    edges = gpd.read_parquet(PathRegistry.GEO_EDGES_ALL)

    # Load spira data
    spira = data_reader.spira_shapes(PathRegistry.SPIRA_FLOW_DATA, df_around=gdf_zoi)
    spira_codes = data_reader.spira_codes(PathRegistry.SPIRA_FLOW_DATA)

    logging.info("Transforming the data")
    logging.debug("...Roads to in/out")
    # Create a buffer around the border
    out_boundary = gdf_zoi.copy()
    out_boundary['geometry'] = spatial_utils.buffer_around(gdf=out_boundary, buffer_size=400)
    in_boundary = gdf_zoi.copy()
    in_boundary['geometry'] = spatial_utils.buffer_around(gdf=in_boundary, buffer_size=-100)
    buffered_boundary = out_boundary.difference(in_boundary)

    # Find the edges that touches a portion of the buffered border
    filtered_edges = []
    for _, edge in edges.iterrows():
        if buffered_boundary.intersects(edge.geometry).any():
            filtered_edges.append(edge['id'])
    edges = edges.assign(is_inside=edges['id'].isin(filtered_edges))

    logging.debug("...Assigning spira to roads")
    # Assign spiras to roads
    spira = spatial_assignment.spira_to_road(df_spira=spira, df_edge=edges)

    # Get the direction of the road with respect to the boundary
    edges['linestring_direction'] = edges.apply(
        lambda row: spatial_utils.direction_wrt_polygon(row['geometry'], polygon=gdf_zoi.geometry.iloc[0]), 
        axis=1
    )
    spira = (
        spira
        .merge(edges[['u', 'v', 'key', 'linestring_direction', 'is_inside', 'is_relevant']], on=['u', 'v', 'key'], how='left')
        .rename(columns={'linestring_direction': 'spira_direction'})
    )

    logging.debug("...Filtering spiras based on position")
    # Filter the edges that are on the border and that are important roads, then filter the spiras
    buffer_fr = 10 #buffer from roads
    buffer_fa = 150 #buffer from zoi
    spira = _analyse_spira_position(spira, buffered_boundary, buffer_fr, buffer_fa)

    logging.debug("...Filtering spiras based on accuracy")
    # Spira flows and accuracy
    spira_in_ok_codes, spire_out_ok_codes = _analyze_spira_accuracy(
        spira, spira_codes, start_date, end_date
    )

    logging.info("Computing outflows")
    # Load 5min spiras
    sel5m = [[('sensor_id', '==', id)] for id in spira_in_ok_codes + spire_out_ok_codes]
    spira5m = data_reader.spira_flows(PathRegistry.SPIRA_FLOW5M_DATA, filters=sel5m)

    spira5m = spira5m[spira5m['DateTime'].between(start_date, end_date, inclusive='left')]
    spira5m = pd.merge(spira5m, spira_codes, left_on='sensor_id', right_on='spira_code', how='left').drop(columns=['sensor_id'])
    spira5m = spira5m.sort_values('DateTime')


    outflow = {}
    for daytype in ["Weekday", "Saturday", "Holiday", "All"]:
        logging.debug(f"...Computations for daytype {daytype}")
        # Filter the data measured during the day of interest
        if daytype != "All":
            is_day = temporal_utils.add_daytype_to_series(spira5m['DateTime']) == daytype
            spira5m_day = spira5m[is_day]

        in_flow = np.array(inflow_df[daytype].to_list())

        # Compute the in-flow from spiras
        flow_in = spira5m_day[spira5m_day['spira_code'].isin(spira_in_ok_codes)].copy()
        flow_in['time'] = flow_in['DateTime'].dt.time
        flow_in = flow_in.drop(columns=['spira_unique_id','spira_code','DateTime'])
        flow_in = flow_in.groupby('time').sum()

        # Compute the out-flow from spiras
        flow_out = spira5m_day[spira5m_day['spira_code'].isin(spire_out_ok_codes)].copy()
        flow_out['time'] = flow_out['DateTime'].dt.time
        flow_out = flow_out.drop(columns=['spira_unique_id','spira_code','DateTime'])
        flow_out = flow_out.groupby('time').sum()

        flow = pd.merge(flow_in, flow_out, on='time', suffixes=['_in','_out'])
        flow = flow / flow.mean()
        flow['out-in-ratio'] = flow['count_out'] / flow['count_in']

        # Compute the outflow from gates given the inflow from gates and the out-in ratio
        raw_out_flow = in_flow * flow['out-in-ratio'].values

        # Smooth it
        smoothed_out_flow = savgol_filter(raw_out_flow, 12*2+1, 2)

        # Check if the TI=TO during the day & fix the difference
        logging.debug(f"Check if the TI-TO==0: TO-TI={sum(in_flow)-sum(smoothed_out_flow)}") ##Apparently there is 250 vehicles higher than the outflow 
        out_flow = smoothed_out_flow * sum(in_flow) / sum(smoothed_out_flow)

        outflow[daytype] = out_flow
    
    logging.info("Saving results")
    out_df = pd.DataFrame(data=outflow)
    PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(PathRegistry.OUTFLOW_ESTIMATED)

    return out_df


def _analyse_spira_position(spira, buffered_boundary, buffer_fr = 10, buffer_fa = 150):
    # Filter the edges that are on the border and that are important roads, then filter the spiras
    spira = spira[(spira['is_inside']) & (spira['is_relevant'])].reset_index(drop=True)

    # Check and filter the distance from the spiras to the corresponding road and to the border
    spira['distances_from_roads'] = spira.apply(lambda row: row['geometry_x'].distance(row['geometry_y']), axis=1)
    spira['distances_from_zoi'] = spira.apply(lambda row: row['geometry_x'].distance(buffered_boundary), axis=1)

    spira = spira[(spira['distances_from_roads'] < buffer_fr) & (spira['distances_from_zoi']  < buffer_fa)].drop(columns=['distances_from_roads', 'distances_from_zoi'])
    logging.debug(f"There is a total of {spira.shape[0]} spiras")

    return spira

def _analyze_spira_accuracy(spira, spira_codes, start_date, end_date):
    
    spira_code_directions = pd.merge(
        spira[['spira_unique_id', 'spira_direction']], 
        spira_codes
    )
    spire_in = spira_code_directions['spira_code'][
        spira_code_directions['spira_direction'] == 'incoming'
    ].to_list()
    spire_out = spira_code_directions['spira_code'][
        spira_code_directions['spira_direction'] == 'outgoing'
    ].to_list()
    
    # Load and filter spira accuracy
    sel5macc = [[('spira_code', '==', id)] for id in spire_in + spire_out]
    accuracy = data_reader.spira_accuracy(PathRegistry.SPIRA_ACCUR_DATA, filters=sel5macc)
    
    accuracy = accuracy[accuracy['DateTime'].between(start_date, end_date, inclusive='left')]
    accuracy = accuracy.sort_values('DateTime')
    
    # Compute statistics
    df = accuracy.groupby('spira_code').describe(
        percentiles=[.1,.25,.5,.75,.9], 
        include=[np.number]
    ).reset_index()
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                  for col in df.columns]
    df = df.rename(columns={'spira_code_': 'spira_code'})
    df = df.merge(spira_code_directions, on='spira_code')
    
    # Print statistics
    _print_accuracy_statistics(df)
    
    # Select reliable spiras (10% percentile >= 100)
    spira_in_ok_codes = df['spira_code'][
        (df['spira_direction']=='incoming') & (df['count_10%']>=100)
    ].to_list()
    spire_out_ok_codes = df['spira_code'][
        (df['spira_direction']=='outgoing') & (df['count_10%']>=100)
    ].to_list()
    
    logging.debug(f"Reliable incoming loops: {len(spira_in_ok_codes)}")
    logging.debug(f"Reliable outgoing loops: {len(spire_out_ok_codes)}")
    
    return spira_in_ok_codes, spire_out_ok_codes


def _print_accuracy_statistics(df):
    for direction in ['incoming', 'outgoing']:
        logging.debug(f"\n{direction.upper()} spiras: {df[df['spira_direction']==direction].shape[0]}")
        for pct in [10, 25, 50, 75, 90]:
            col = f'count_{pct}%'
            count = df[(df[col]<100)&(df['spira_direction']==direction)].shape[0]
            total = df[df['spira_direction']==direction].shape[0]
            percentage = round(count/total*100) if total > 0 else 0
            logging.debug(f"  {pct}% percentile < 100: {count} ({percentage}%)")
    
    logging.debug(f"\nTOTAL spiras: {df.shape[0]}")
    for pct in [10, 25, 50, 75, 90]:
        col = f'count_{pct}%'
        count = df[df[col]<100].shape[0]
        percentage = round(count/df.shape[0]*100) if df.shape[0] > 0 else 0
        logging.debug(f"  {pct}% percentile < 100: {count} ({percentage}%)")


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        # Selected period ## TODO: move this to the config file
        start_date = datetime(2024,6,1)
        end_date = datetime(2024,7,31)
        get_outflow_from_spiras(start_date, end_date)


if __name__ == "__main__":
    main()