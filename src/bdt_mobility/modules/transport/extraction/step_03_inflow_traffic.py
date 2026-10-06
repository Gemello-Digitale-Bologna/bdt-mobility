from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils import temporal_utils
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("inflow_estimated", PathRegistry.INFLOW_ESTIMATED, "table"),
]

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import euclidean


def get_inflows_from_gates(start_date, end_date, problematic_gates=None):
    logging.info("Reading the data")
    # Load data
    gates = pd.read_parquet(PathRegistry.GATES_PARQUET)
    data = pd.read_parquet(PathRegistry.GATE_DATA_PARQUET)

    logging.info("Preparing the data for inflow estimation")
    # Manipulate data
    gate_data = (
        data
        .merge(
            gates, 
            on="gate", 
            how="left", 
            validate="many_to_one"
        )
        .drop(
            columns=["ID", "Indirizzo", "Settore", "Link google maps"]
        )
    )
    gate_data["Data"] = pd.to_datetime(gate_data["Data"])

    hourly_total_vehicle = (
        gate_data[["Data", "gate", "count"]]
        .groupby(["Data", "gate"], as_index=False)
        .sum()
        .pivot_table(index="Data", columns="gate", values="count", fill_value=0)
        .astype(int)
    )
    average_hourly_vehicle_flow = (
        hourly_total_vehicle
        .groupby(hourly_total_vehicle.index.hour)
        .mean()
        .astype(int)
    )
        
    # Fill the missing values of problematic gates
    if problematic_gates is not None:
        logging.info("Computing gates similarities")
        similar_gates_dict, min_dist = _compute_gate_similarity(average_hourly_vehicle_flow, problematic_gates)
    else:
        similar_gates_dict = {}

    logging.info("Computing inflow for all days")
    average_hourly_vehicle_flow = (
        hourly_total_vehicle[start_date:end_date]
        .groupby(hourly_total_vehicle[start_date:end_date].index.hour)
        .mean()
        .astype(int)
    )
    predicted_average_hourly_vehicle_flow = _predict_missing_gate(average_hourly_vehicle_flow, similar_gates_dict)
    average_hourly_vehicle_flow_filtered = predicted_average_hourly_vehicle_flow.astype(int)
    average_hourly_vehicle_flow_filtered_sumed = average_hourly_vehicle_flow_filtered.sum(axis=1)
    total_flow = _extend_to_5min(average_hourly_vehicle_flow_filtered_sumed)

    logging.info("Computing inflow for each day type")
    hourly_vehicle = hourly_total_vehicle[start_date:end_date].reset_index()
    hourly_vehicle['Data'] = pd.to_datetime(hourly_vehicle['Data'])
    hourly_vehicle['Hour'] = hourly_vehicle['Data'].dt.hour
    hourly_vehicle['DayType'] = temporal_utils.add_daytype_to_series(hourly_vehicle['Data'])

    day_types = ['Weekday', 'Saturday', 'Holiday']
    flows_by_day = {}

    for day in day_types:
        day_data = hourly_vehicle[hourly_vehicle['DayType'] == day]
        numeric_columns = hourly_vehicle.columns.difference(['Data', 'Hour', 'DayType'])
        hourly_avg = day_data.groupby('Hour')[numeric_columns].mean().astype(int)

        predicted_hourly_avg = _predict_missing_gate(hourly_avg, similar_gates_dict)
        to_extract = pd.Series(
            _extend_to_5min(predicted_hourly_avg.sum(axis=1)),index=range(0, 12 * 24)
        )
        flows_by_day[day] = to_extract

    # Check the avg daily flow (weighted avg of flows per daytypes)
    combined_flow = (5/7) * flows_by_day['Weekday'] + (1/7) * flows_by_day['Saturday'] + (1/7) * flows_by_day['Holiday']

    # Save the data
    # PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    in_df = pd.DataFrame(flows_by_day)
    in_df["All"] = total_flow
    in_df.to_parquet(PathRegistry.INFLOW_ESTIMATED)
    logging.info("Process finished and data saved")


def _predict_missing_gate(df, similar_gates, smoothness=1.5):
    time = np.arange(0, 24, 1)
    for target_gate, source_gate in similar_gates.items():
        data = df[source_gate].values.flatten()
        spline = UnivariateSpline(time, data, s=smoothness)
        df[target_gate] = spline(time)
    return df

def _extend_to_5min(df):
    to_extract = df.copy()
    to_extract[24] = to_extract[0]
    to_extract.index = to_extract.index*12
    to_extract = to_extract.reindex(range(12*24+1)).interpolate(method='quadratic')
    if to_extract.min() < 0:
        to_extract = to_extract - to_extract.min()
    to_extract = to_extract /12
    to_extract = to_extract[:-1]
    
    return to_extract.values

def _compute_gate_similarity(average_hourly_vehicle_flow, problematic_gates):
    # Mapping of gates with missing values as a key to their similar gates as values.
    # This similarity is based on the pattern of vehicle flow throughout the day beyond the date range we considered above.

    # Problematic gates and reference gates
    problematic_hourly_vehicle_flow = average_hourly_vehicle_flow[problematic_gates]
    average_hourly_vehicle_flow_ref = average_hourly_vehicle_flow.drop(columns=problematic_gates)

    # Initialize data
    similar_gates_dict = {}
    df_data = []
    n_problematic = len(problematic_hourly_vehicle_flow.columns)
    n_ok = len(average_hourly_vehicle_flow_ref.columns)
    mat_distances = np.zeros(shape=(n_problematic, n_ok))
    i = 0
    # For each problematic gate
    for problematic_gate in problematic_hourly_vehicle_flow.columns:
        gate_distances = []
        # Find the most similar "ok" gate and their distance
        j = 0
        for gate in average_hourly_vehicle_flow_ref.columns:
            gate_vector = average_hourly_vehicle_flow_ref[gate].values
            problematic_gate_vector = problematic_hourly_vehicle_flow[problematic_gate].values
            dist = euclidean(gate_vector, problematic_gate_vector)
            gate_distances.append((gate, dist))

            mat_distances[i,j] = dist
            j += 1
        # Save
        min_distance_gate = min(gate_distances, key=lambda x: x[1])
        similar_gates_dict[problematic_gate] = min_distance_gate[0]
        df_data.append([problematic_gate, min_distance_gate[0], min_distance_gate[1]])
        
        i += 1

    min_distances_df = pd.DataFrame(df_data, columns=['Problematic Gate', 'Reference Gate', 'Euclidean Distance'])

    return similar_gates_dict, min_distances_df


def main():
    # Selected period ## TODO: move this to the config file
    start_date = datetime(2024,6,1)
    end_date = datetime(2024,7,31)
    # These are the gates that we do not have data for the selected period.
    problematic_gates = ["Colombo", "Della Pietra", "Di Vittorio","San Mamolo", "Terrapieno", "Togliatti", "Toscana"]
    logging.basicConfig(level=logging_level)
    with timed():
        get_inflows_from_gates(start_date, end_date, problematic_gates)


if __name__ == "__main__":
    main()
