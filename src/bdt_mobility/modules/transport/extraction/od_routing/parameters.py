
context = {
    # Road network
    "road_network": "geo", # "geo" for relevand highways only, or "geo_all" for all roads, or "dsf" for the DSF road network
    "activate_ltz": False, # If True, the road network is modified by removing the edges that are within the LTZ area

    # Zone type
    "set_aoi_type": "od",

    # Simulation parameters
    "simulator": "fts", # "fts" or "uxsim" or "dsf"
    "total_seeds": 1,
    "first_hour": 3,

    # What-if scenario
    "is_new_scenario": False,
    "car_volume_variation": -0.1, # in [-1.0, +1.0]

    # Output level
    "aggregation_step": 60*60 # Output is aggregated over this time granularity. Remark: it is a divisor of (24*3600)=1day & a multiple of time_step
}

_params_simulator = {
    "fts": {
        "road_length_threshold": 0,
        "platoon_size": 1,
        "SP": 300,
        "random_seed": 42,
        "refresh_workers": 8,
    },
    "uxsim": {
        "platoon_size": 50,
        "total_hours": 24,
        "duo_time": 900,
        "jam": 0.2,
        "reaction_time": 1,  # reaction_time*platoon_size will give the time step of simulation
        "road_length_threshold": 30,
    },
    "dsf": {
        "platoon_size": 1,
        "scale": 1,
        "dt_agent": 1, #time-step interval for launching new platoons
        "save_data_interval": 300,
        "update_paths" : {
            "interval": 300,
            "throw_on_empty": False
        },
        "set_edge_weight" : {
            "weight" : "traveltime",
            "threshold" : 0.05
        },
        "error_probability": 0.05,
        "kill_stagnant_agents": 40.0,
        "agent_insertion_method": "RANDOM_ODS",
        "road_length_threshold": 0,
        "logging_level": None ## or "trace", or "debug"
    }
}

if context["simulator"] not in _params_simulator:
    raise NotImplementedError(f"Simulator '{context['simulator']}' not supported")

context.update(_params_simulator[context["simulator"]])

_time_step_calculators = {
    "fts": lambda ctx: 1,
    "uxsim": lambda ctx: ctx["platoon_size"] * ctx["reaction_time"],
    "dsf": lambda ctx: 1,
}

context["time_step"] = _time_step_calculators[context["simulator"]](context)