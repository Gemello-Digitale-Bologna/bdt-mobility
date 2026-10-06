import polars as pl
import numpy as np
import math

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.modules.transport.extraction.od_routing.edge_matrix import (
    preprocess_trips as _step05_preprocess_trips,
    traffic_per_edge_min_granularity,
)
from bdt_mobility.modules.transport.extraction.od_routing.utils import melt_matrix

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import DynamicPlatformOutputs

MAIN_OUTPUTS = DynamicPlatformOutputs(PathRegistry.TRAFFIC_SIMULATOR, "table")


def _round_sig(x, sig=9):
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def preprocess_trips(
	trips: pl.LazyFrame, 
	edges: pl.LazyFrame,
	time_step: int
) -> pl.LazyFrame:
	return (
        trips
		.drop("x","s", "v", "dn", "orig", "dest", strict=False)
		.rename({"link": "link_id", "name": "platoon_id"})
        .join(
			edges.select("link_id", "length", "maxspeed", "n_lanes"), 
			on="link_id"
		)
		.sort(["platoon_id", "t", "link_id"])
        .with_columns(
			speed = pl.col("length") / (pl.col("t").shift(-1).over("platoon_id") - pl.col("t")),
			time_step_id = pl.col("t") // time_step,
			free_travel_time = pl.col("length") / pl.col("maxspeed")
			)
		.drop_nulls("speed")
		.filter(pl.col("speed")>0)
        .select("platoon_id", "t", "time_step_id", "link_id", "length", "speed", "maxspeed", "n_lanes", "free_travel_time")
    )


def vehicles_from_new_scenario(
    vehicles_df: pl.LazyFrame,
    car_volume_variation: float,
    seed: int,
) -> pl.LazyFrame:
    """_summary_
    Create a new vehicles dataframe for a what-if scenario where we have a different volumne of cars
    In the beginning we add or remove cars homogeneously across different fuel, euro-level and travelling time
    Args:
        vehicles_df (pl.LazyFrame): Dataset of cars with engine and euro level informations. 
        car_volume_variation (float): fraction of cars differences between simulation and what-if scenario
        seed (int): simulation seed

    Returns:
        pl.LazyFrame: _description_
    """
    rng = np.random.default_rng(seed)
    nr_vehicles = vehicles_df.select(pl.len()).collect().item()
    vehicles_df = (
        vehicles_df
        .with_columns(rnd = rng.random(nr_vehicles))
    )
    
    if car_volume_variation <= 0:
        new_vehicles = (
            vehicles_df
            .filter(pl.col('rnd') > np.abs(car_volume_variation))
            .drop('rnd')
        )
        
    else:
        new_vehicles = (
            vehicles_df
            .filter(pl.col('rnd') <= np.abs(car_volume_variation))
            .drop('rnd')
            .with_row_index(
                name = 'vehicle_id',
                offset=nr_vehicles + 1
            )
        )
        
        new_vehicles = pl.concat([vehicles_df.drop('rnd'), new_vehicles])
    
    return new_vehicles.sort(["time_step_id","link_id", "pollutant"])


def _power_law_expr(x):
	return (
		pl.when(x <= pl.col("x_break"))
		.then(pl.col("alpha1") * x.log10() + pl.col("intercept1"))
		.otherwise(pl.col("alpha2") * x.log10() + pl.col("intercept2"))
	)

def _estimate_y1_expr():
    x0 = pl.col("traffic_density")
    x1 = pl.col("traffic_density_changed")

    f_x0 = _power_law_expr(x0)
    f_x1 = _power_law_expr(x1)

    return (
        pl.when((x0 <= 0) | (x1 <= 0))
        .then(None) 
        .otherwise(
            pl.col("ratio_speed") + (f_x1 - f_x0)
        )
    )

def add_fuel_usage(ef):
    return ef.with_columns(
        pl.when(pl.col("fuel_type") == pl.col("fuel_actual"))
        .then(1.0)
        .when(pl.col("fuel_actual").is_in(["lpg", "cng", "electricity"]))
        .then(0.8)
        .otherwise(0.2)
        .alias("usage")
    )

def estimate_segment_emissions(trips, vehicles, ef, alias="emission"):
    # expression in excel file gives result in g/km, to convert to g/m divide by 1k here:
    speed = pl.max_horizontal(pl.col('speed'), pl.col('min speed [km/h]') / 3.6)
    ef_expr = (
        pl.col("usage")
        * (
            pl.col("alpha") * speed ** 2
            + pl.col("beta") * speed
            + pl.col("gamma")
            + pl.col("delta") / speed
        )
        / (
            pl.col("epsilon") * speed ** 2
            + pl.col("zita") * speed
            + pl.col("hta")
        )
        * (1 - pl.col("reduction factor [%]"))
        / 1000
    )
    return ( 
		trips.join(
			vehicles.join(
				ef.filter((pl.col("mode") == "")),
				on=["fuel", "segment", "euro_level"],
			),
			on="platoon_id",
		)
		.sort(["platoon_id", "time_step_id", "link_id", "t"])
		.with_columns((ef_expr * pl.col("length")).alias(alias))
	)


def simulate_new_scenario(
	trips: pl.DataFrame,
	traffic: pl.LazyFrame,
	car_volume_variation: float,
	road_length_threshold: int,
	fit_params_df: pl.DataFrame
):
	# Join between trips and traffic to add columns of traffic densisity and other traffic metrics
	trips = (
		trips
		.join(
			traffic.select('t', 'link_id', 'traffic_volume',),
			on=['link_id', 't'],
			how='left'
		)
		.with_columns(
			pl.col("traffic_volume").fill_null(0)
		)
	)

	trips = (
		trips
		.rename({'maxspeed': 'free_speed'})
		.with_columns([
			(pl.col('traffic_volume') / (pl.col('length') * pl.col('n_lanes'))).alias("traffic_density"), # initial traffic density
		    (pl.col('speed') / pl.col('free_speed')).alias("ratio_speed") # Ratio: speed vs free-speed
		])
		.with_columns([
			(pl.col('traffic_density') * (1 + car_volume_variation)).alias("traffic_density_changed"), # New traffic density
			(pl.when(pl.col('length') > road_length_threshold)
				.then(pl.lit("long"))
				.otherwise(pl.lit("short"))
				.alias('edge_type'))
		])
	)

	trips = trips.join(
		fit_params_df,
		on='edge_type',
		how='left'
	)

	trips = (
		trips
		.with_columns(speed_ratio_changed = _estimate_y1_expr())
		.with_columns(new_speed = pl.col('speed_ratio_changed') * pl.col('free_speed'))
		.cast({'speed_ratio_changed' : pl.Float64, 'new_speed' : pl.Float64})
	)

	# Keep only columns that I need for a the new emission estiamtion
	trips = (
		trips.select('platoon_id', 't', 'link_id', 'length', 'new_speed', "time_step_id")
			.rename({'new_speed': 'speed'})
	)

	return trips


def compute_emissions_min_granularity(
	i_seed: int,
	edges: pl.DataFrame,
	fit_params_df: pl.DataFrame,
	poll_ef: pl.DataFrame,
	co2_ef: pl.DataFrame,
	vehicles: pl.DataFrame,
	trips: pl.DataFrame,
	time_step: int,
	road_length_threshold: float,
	is_new_scenario: bool = False,
	car_volume_variation: float = 0.0,
	traffic: pl.LazyFrame | None = None
) -> pl.LazyFrame:

	trips = preprocess_trips(trips=trips, edges=edges, time_step=time_step)

	### Estimate variations of cars for a new scenario
	if is_new_scenario:
		# At the moment, we change car volumes homogeneously by a car_variation variable
		# We are not considering any diffences by fuel engine or euro level
		vehicles = vehicles_from_new_scenario(vehicles, car_volume_variation, i_seed)
		trips = simulate_new_scenario(trips, traffic, car_volume_variation, road_length_threshold, fit_params_df)
	trips = trips.select('platoon_id', "time_step_id", 't', 'link_id', 'length', 'speed')

	# Get emissions: POLL
	poll_emissions = (
        estimate_segment_emissions(trips, vehicles, ef=poll_ef, alias="emission")
		.sort(["time_step_id", "link_id", "pollutant"])
        .group_by("time_step_id", "link_id", "pollutant", maintain_order=True)
        .agg(pl.col("emission").sum())
		.with_columns(pl.col("emission").cast(pl.Float64).alias("emission"))
    )

	# Get emissions: CO2
    # calorific is in MJ/kg, so multiply by 1k here to get grams of CO2
	co2_expr = (
        1000 * pl.col("ec") / pl.col("fuel_calorific") * pl.col("fuel_co2_combustion")
    )
	co2_emissions = (
        estimate_segment_emissions(trips, vehicles, co2_ef, alias="ec")
        .with_columns(emission=co2_expr)
        .with_columns(pollutant=pl.lit("co2"))
		.sort(["time_step_id", "link_id", "pollutant"])
        .group_by("time_step_id", "link_id", "pollutant", maintain_order=True)
        .agg(pl.col("emission").sum())
		.with_columns((pl.col("emission")/1000).cast(pl.Float64).alias("emission"))
    )

	return (
        pl.concat([poll_emissions, co2_emissions])
		.sort(["time_step_id", "link_id", "pollutant"])
        .sort(pl.all().exclude("emission"))
        .fill_nan(0)
        .with_columns(
            start_t = pl.col("time_step_id") * time_step,
            end_t = (pl.col("time_step_id") + 1) * time_step,
        )
		.select("link_id", "start_t", "end_t", "pollutant", "emission")
		.sort(["link_id", "start_t", "end_t", "pollutant"])
    )


def aggregate_emissions_per_zone(
	edges: pl.LazyFrame,
	emissions: pl.LazyFrame,
	aggregation_step: int,
):
	return (
		emissions
		.with_columns(
			start_time_step_id = pl.col("start_t") // aggregation_step
		)
		.join(edges, on="link_id", how="left")
		.drop("link_id")
		.select(["id_zone", "start_time_step_id", "pollutant", "emission"])
		.group_by(["id_zone", "start_time_step_id", "pollutant"], maintain_order=True)
		.agg(pl.col("emission").sum())
		.with_columns(
			start_t = pl.col("start_time_step_id") * aggregation_step,
			end_t = (pl.col("start_time_step_id") + 1) * aggregation_step,
			)
		.select(["id_zone", "start_t", "end_t", "pollutant", "emission"])
		.sort(["pollutant", "id_zone", "start_t", "end_t"])
	)


def aggregate_emissions_per_road(
	emissions: pl.DataFrame,
	aggregation_step: int
):
	return (
		emissions
		.with_columns(
			start_time_step_id = pl.col("start_t") // aggregation_step,
			)
		.select(["link_id", "start_time_step_id", "pollutant", "emission"])
		.group_by(["link_id", "start_time_step_id", "pollutant"], maintain_order=True)
		.agg(pl.col("emission").sum())
		.with_columns(
			start_t = pl.col("start_time_step_id") * aggregation_step,
			end_t = (pl.col("start_time_step_id") + 1) * aggregation_step,
			)
		.select(["link_id", "start_t", "end_t", "pollutant", "emission"])
		.sort(["pollutant", "link_id", "start_t", "end_t"])
	)


def _traffic_df_min_granularity(i_seed: int, time_step: int, first_hour: int, edges: pl.DataFrame):
	trips = _step05_preprocess_trips(
			trips=pl.read_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed)),
			platoons=pl.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=i_seed)),
			edges=edges
		)
	mat_t, link_t_mapping = traffic_per_edge_min_granularity(
			trips=trips, edges=edges, time_step=time_step, t_min=first_hour*3600
		)
	result = (
		melt_matrix(mat_t, link_t_mapping, time_step=time_step, t_min=first_hour*3600)
		.rename({"counts": "traffic_volume", "spatial_unit": "link_id"})
		.sort(["link_id", "t"])
		.lazy()
	)
	return result


def main():
	logging.basicConfig(level=logging_level)
	with timed():
		### Read the input data
		logging.info("Read the input data")
		poll_ef = pl.scan_parquet(PathRegistry.POLL_EF_PARQUET)
		poll_ef = add_fuel_usage(poll_ef) ## TODO: put this step in 08_emissions_preprocess
		co2_ef = pl.scan_parquet(PathRegistry.CO2_EF_PARQUET)
		co2_ef = add_fuel_usage(co2_ef) ## TODO: put this step in 08_emissions_preprocess

		edges = (
			pl.read_csv(PathRegistry.SIM_EDGES)
			[['id', 'length', 'maxspeed_mps', 'lanes', 'id_zone']]
			.rename({'id': 'link_id', 'maxspeed_mps':'maxspeed', 'lanes':'n_lanes'})
			.cast({"link_id": pl.Utf8, 'length': pl.Float64, 'id_zone': pl.Int64})
			.lazy()
		)

		fit_params_df = pl.scan_parquet(
				PathRegistry.NON_LINEAR_FIT_PARAMS
			).rename({'edge': "edge_type"})

		dataframes = {
			"edges": edges,
			"poll_ef": poll_ef, 
			"co2_ef": co2_ef, 
			"fit_params_df": fit_params_df
		}

		### Cycle for each seed
		keys = ["time_step", "road_length_threshold", "car_volume_variation", "is_new_scenario"]
		params = {k: context[k] for k in keys if k in context}
		names = []
		list_seeds = [i for i in range(context['total_seeds'])]
		for i_seed in list_seeds:
			logging.info(f"Considering simulation seed {i_seed}")
			# Read vehicles input data
			vehicles = pl.scan_parquet(PathRegistry.VEHICLES_SEED.substitute(seed=i_seed))
			trips = pl.scan_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=i_seed))
			if context["is_new_scenario"]:
				logging.debug("Computing traffic per edge, min granularity")
				traffic = _traffic_df_min_granularity(i_seed=i_seed, edges=edges.collect(), time_step=context["time_step"], first_hour=context["first_hour"])
			else:
				traffic = None

			# Compute emissions per road
			logging.info("Computing emissions per minimum time granularity")
			e_df = compute_emissions_min_granularity(
				i_seed=i_seed, 
				**dataframes,
				vehicles=vehicles,
				trips=trips,
				traffic=traffic,
				**params
			)

			# Average the emissions per road hourly
			logging.info("Computing and saving metrics per road per aggregated times")
			e_edges_df = aggregate_emissions_per_road(
				emissions=e_df,
				aggregation_step=context['aggregation_step']
			).with_columns(
				pl.col("emission")
				.map_elements(_round_sig, return_dtype=pl.Float64)
				.alias("emission")
			)
			# Save the file
			if context["is_new_scenario"]:
				filepath = PathRegistry.EMISSIONS_PER_ROAD_VARIATION.substitute(seed=i_seed, variation=context["car_volume_variation"])			
			else:
				filepath = PathRegistry.EMISSIONS_PER_ROAD.substitute(seed=i_seed)
			e_edges_df.sink_parquet(filepath)
			names.append((filepath.stem, filepath))

			# Average the emissions per zone hourly
			logging.info("Computing and saving metrics per zone per aggregated times")
			e_zones_df = aggregate_emissions_per_zone(
				edges = edges.select(["link_id", "id_zone"]).lazy(),
				emissions=e_df,
				aggregation_step=context['aggregation_step']
			).with_columns(
				pl.col("emission")
				.map_elements(_round_sig, return_dtype=pl.Float64)
				.alias("emission")
			)
			# Save the file
			if context["is_new_scenario"]:
				filepath = PathRegistry.EMISSIONS_PER_ZONE_VARIATION.substitute(seed=i_seed, variation=context["car_volume_variation"])
			else:
				filepath = PathRegistry.EMISSIONS_PER_ZONE.substitute(seed=i_seed)
			e_zones_df.sink_parquet(filepath)
			names.append((filepath.stem, filepath))
		# TODO: add the computation of the average emissions and std per zone

	return names


if __name__ == "__main__":
	main()