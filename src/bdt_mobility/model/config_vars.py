from civic_digital_twins.dt_model import (
    Index,
    DistributionIndex
)
import pandas as pd
from scipy import stats
import numpy as np

from bdt_mobility.model.axis import ECL_AXIS

# ---------------------------------------------------------------
# Policy parameters  (I_P_*)
# ---------------------------------------------------------------
i_p_start_time = Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds())
i_p_end_time = Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds())
i_p_cost = Index(
    "cost per euro level", 
    np.array([5.00 - e * 0.25 for e in range(7)]),
    axes=(ECL_AXIS,)
)
i_p_fraction_exempted = Index("exempted vehicles %", 0.15)
i_p_pt_frequency_modification = Index("modification of the actual frequency of the PT", 0.0)
i_p_pt_capillarity_modification = Index("modification of the actual capillarity of the PT", 0.0)
i_p_pt_cost_modification = Index("modification of the actual cost of the PT", 0.0)
i_p_pt_time_modification = Index("modification of the actual time difference of the PT", 0.0)

policy_params = {
    "i_p_start_time": i_p_start_time,
    "i_p_end_time": i_p_end_time,
    "i_p_cost": i_p_cost,
    "i_p_fraction_exempted": i_p_fraction_exempted,
    "i_p_pt_frequency_modification": i_p_pt_frequency_modification,
    "i_p_pt_capillarity_modification": i_p_pt_capillarity_modification,
    "i_p_pt_cost_modification": i_p_pt_cost_modification,
    "i_p_pt_time_modification": i_p_pt_time_modification,
}
# ---------------------------------------------------------------
# Behavioural parameters  (I_B_*)
# ---------------------------------------------------------------
i_b_p50_cost = DistributionIndex("cost 50% threshold", stats.uniform, {"loc": 4.0, "scale": 7.0})
i_b_p50_anticipating = Index("anticipation 50% likelihood", 0.25)
i_b_p50_postponing = Index("postponement 50% likelihood", 0.5)
i_b_p50_anticipation = Index("anticipation distribution 50% threshold", 0.50)
i_b_p50_postponement = Index("postponement distribution 50% threshold", 0.50)
i_b_pt_capillarity = Index("importance level of capillarity of the pt", 4.5)
i_b_pt_frequency = Index("importance level of frequency per stop of the pt", -1.45)
i_b_pt_cost = Index("importance level of the cost of the pt", -0.30)
i_b_pt_time = Index("importance level of the time difference of the car and pt", -0.034)

behavioral_params = {
    "i_b_p50_cost": i_b_p50_cost,
    "i_b_p50_anticipating": i_b_p50_anticipating,
    "i_b_p50_postponing": i_b_p50_postponing,
    "i_b_p50_anticipation": i_b_p50_anticipation,
    "i_b_p50_postponement": i_b_p50_postponement,
    "i_b_pt_capillarity": i_b_pt_capillarity,
    "i_b_pt_frequency": i_b_pt_frequency,
    "i_b_pt_cost": i_b_pt_cost,
    "i_b_pt_time": i_b_pt_time,
}

# ---------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------
ENSEMBLE_SIZE = 100
ETHICAL_YEAR = 2023

# ---------------------------------------------------------------
# Config parameters
# ---------------------------------------------------------------
config_params = {
    "decision_strategy": "parallel", ## choices are taken sequentially or parallely? use values in ['sequential', 'parallel']
    "time_shift_strategy": "flexible", ## choice: flexibly shift their activity time (earlier or later), or fixed and predetermined time shift. use values in ['fixed', 'flexible']
    "modal_shift_option": "lpt", ## choice: if the modal shift is included, and if yes which type. use values in ['no', 'lpt', 'active', 'lpt-active']
    "policy": "cordon-based", ## choice: area-based (for impact inflow and starting) or cordon-based (for impact inflow)
    "enable_ethics": False, ## choice: build the ethics module in model/simulation (not extraction)
}