
import numpy as np
import pandas as pd

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

# estimation_method is hardcoded to "probabilistic" in main(), which writes the _alternative files.
MAIN_OUTPUTS = [
    PlatformOutput("starting_estimated_alternative", PathRegistry.STARTING_ESTIMATED, "table"),
    PlatformOutput("traffic_estimated_alternative", PathRegistry.TRAFFIC_ESTIMATED, "table"),
    PlatformOutput("passengers_to_vehicles", PathRegistry.P2V, "table"),
]

def read_model_inputs():
    # weekday in-flow (5min)
    inflow_df = pd.read_parquet(PathRegistry.INFLOW_ESTIMATED)
    inflow = np.array(inflow_df["Weekday"].to_list())

    # outflow in weekdays from spira_outflow_estimation
    outflow_df = pd.read_parquet(PathRegistry.OUTFLOW_ESTIMATED)
    outflow = np.array(outflow_df["Weekday"].to_list())

    # Total inflow, from OD data (in passengers)
    totals_df = pd.read_parquet(PathRegistry.TOTAL_FLOWS_FROM_OD)
    TInside_tilda = totals_df.loc[(totals_df["from_zoi"]) & (totals_df["to_zoi"]), "flow"].values[0]
    TI_tilda = totals_df.loc[(~totals_df["from_zoi"]) & (totals_df["to_zoi"]), "flow"].values[0]
    TO_tilda = totals_df.loc[(totals_df["from_zoi"]) & (~totals_df["to_zoi"]), "flow"].values[0]

    # Passengers-to-vehicles conversion
    p2v = sum(inflow) / TI_tilda
    logging.debug(f"How many passengers to vehicles: {p2v}")
    tot_inflow = TI_tilda * p2v
    tot_outflow = TO_tilda * p2v
    tot_inside = TInside_tilda * p2v

    return p2v, inflow, outflow, tot_inflow, tot_outflow, tot_inside


def get_model_parameters(dt_mean, t_unit):
    dt_units = dt_mean / t_unit # mean dwell time in time-units
    dur = int(24*60/t_unit) # number of time units in a day

    return dt_units, dur


def get_starting_traffic_estimation_probabilistic(dt_mean, t_unit):
    logging.info("Reading data")
    # Estimated for inflow, outflow, totals
    p2v, inflow, outflow, TI, TO, TInside = read_model_inputs()

    logging.info("Starting the model computations")
    dt_units, dur = get_model_parameters(dt_mean, t_unit)

    alpha_hat = TI / sum(inflow)
    beta_hat = (TInside) / (TInside + TO)

    lambdda = 1/(dt_mean/t_unit)
    tt = np.arange(dur+1)
    p_t = (np.exp(-lambdda * tt) - np.exp(-lambdda * np.roll(tt, -1)))[:-1]

    t_limit = np.where(p_t < 0.001)[0][0] # 0.001 since a reduction of 0.1% is considered negligible
    p_t[t_limit:] = 0

    T_start = - (1-alpha_hat)*inflow
    for t in range(t_limit):
        T_start = T_start + np.roll(outflow, shift=-t) * p_t[t]
    T_start = T_start / (1 - beta_hat)

    c = alpha_hat*inflow + beta_hat*T_start
    Pmat = np.array([np.roll(p_t, shift=i) for i in range(dur)])
    T_stop = np.linalg.solve(Pmat, c)

    Traffics, _ = ts_solve(T_start+inflow, 500)

    return p2v, T_start, T_stop, Traffics


def get_starting_traffic_estimation_scaling(dt_mean, t_unit):

    logging.info("Reading data")
    # Estimated for inflow, outflow, totals
    p2v, inflow, outflow, tot_inflow, tot_outflow, tot_inside = read_model_inputs()

    # Spira traffic estimated
    spira_df = pd.read_parquet(PathRegistry.SPIRA_TRAFFIC_ESTIMATED)
    spira = np.array(spira_df["Weekday"].to_list()) 
    tot_spira = sum(spira) 


    logging.info("Starting the model computations")
    dt_units, _ = get_model_parameters(dt_mean, t_unit)

    c = dt_units * (tot_inflow + tot_outflow + tot_inside) / tot_spira
    logging.debug(f"c-value: {c}")

    traffic = c * spira
    tot_traffic = sum(traffic)
    logging.debug(f"total traffic value: {tot_traffic}")

    traffic_plus_one = np.roll(traffic, shift=-1)
    starting_minus_stopping = traffic_plus_one - traffic - inflow + outflow

    size = starting_minus_stopping.size
    tmp_starting = starting_minus_stopping - outflow + np.roll(inflow, dt_units)
    tmp_starting = tmp_starting.cumsum() / dt_units
    starting = tmp_starting + (tot_inside + tot_outflow - sum(tmp_starting)) / size

    stopping = np.roll(starting, dt_units)
    traffic2, _ = ts_solve(init=starting+inflow)

    return p2v, starting, stopping, traffic2


def ts_solve(init, steps=500, min_diff=1e-5):
    series = init
    sequence = [series]
    for i in range(steps):
        mu = 1 + 3
        alfa = (mu-1)/mu
        new_series = init + np.roll(series,1) * alfa
        sequence += [new_series]
        dist = np.max(np.abs(series - new_series))
        if dist < min_diff:
            break
        series = new_series
    logging.debug(f"N. iterations before convergence: {i}")
    logging.debug(f"Dist between series at the end: {dist}")
    return series, sequence


def main():
    dwell_time_mean = 20  # minutes #TODO: set in the config file
    time_unit = 5  # minutes #TODO: set in the config file
    estimation_method = "probabilistic"  # TODO: set in the config file
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info(f"Traffic model type: {estimation_method}")
        if estimation_method == "scaling":
            p2v, starting, _, traffic = get_starting_traffic_estimation_scaling(dt_mean=dwell_time_mean, t_unit=time_unit)
        elif estimation_method == "probabilistic":
            p2v, starting, _, traffic = get_starting_traffic_estimation_probabilistic(dt_mean=dwell_time_mean, t_unit=time_unit)

        logging.info("Saving final results")
        PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data={"Weekday": starting}).to_parquet(PathRegistry.STARTING_ESTIMATED)
        pd.DataFrame(data={"Weekday": traffic}).to_parquet(PathRegistry.TRAFFIC_ESTIMATED)
        pd.DataFrame(data={"Value": [p2v]}).to_parquet(PathRegistry.P2V)


if __name__ == "__main__":
    main()