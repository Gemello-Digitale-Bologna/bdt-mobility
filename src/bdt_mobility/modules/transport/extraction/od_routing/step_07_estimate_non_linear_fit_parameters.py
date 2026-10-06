import pandas as pd
import gc
import polars as pl
import geopandas as gpd
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
import numpy as np


from bdt_mobility.modules.transport.extraction.od_routing.edge_matrix import (
    inflow_per_edge_min_granularity,
    travel_time_per_edge_min_granularity,
)
from bdt_mobility.modules.transport.extraction.od_routing.utils import melt_matrix
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("no_linear_interpolation_params", PathRegistry.NON_LINEAR_FIT_PARAMS, "table"),
]

from bdt_mobility.modules.transport.extraction.od_routing.parameters import context
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed


def preprocess_trips(
    trips: pl.DataFrame,
    platoons: pl.DataFrame,
    edges: pl.DataFrame,
) -> pl.DataFrame:
    return (
        trips
        .drop("x", "s", "v", "attribute", "dn", strict=False)
        .rename({"name": "platoon_id", "link": "link_id"})
        .cast({"orig": int, "dest": int, "platoon_id": int})
        .join(
            platoons.select(['platoon_id', 'nr_vehicles']),
            on='platoon_id'
        )
        .join(
            edges.select("link_id"),
            on="link_id"
        )
    )


def fit_piecewise_powerlaw(
	traffic: pl.DataFrame, 
	edge_type: str, 
	min_points: int = 50
) -> pd.DataFrame:
	"""
	Fit a double power-law (piecewise power-law) model with an unknown breakpoint
	to traffic data using a log–log linear regression approach.

	The function models the relationship between traffic lane density and
	speed ratio as two distinct power-law regimes separated by an optimally
	estimated breakpoint. The breakpoint is identified by minimizing the
	residual sum of squares in log–log space, subject to a minimum number of
	data points in each regime.

	The fitting procedure is performed in logarithmic space, where power-law
	relationships become linear, and the optimal breakpoint is estimated via
	scalar optimization.

	Args:
			traffic (pd.DataFrame): dataframe computed from simulation that contains number of vehicles in each edge for each time interval
			edge_type (str): contain information about length of edges. Must be 'short' or 'long'
			min_points (int, optional): Minum number of point to starting fit interpolation. Defaults to 50.

	Returns:
			_type_: _description_
	"""
	traffic_lanes_density = traffic["traffic_lanes_density"].to_numpy()  # x-axis
	ratio_speed = traffic['ratio_speed'].to_numpy()     # y-axis

	mask = (traffic_lanes_density > 0) & (ratio_speed > 0)
	x = traffic_lanes_density[mask]
	y = ratio_speed[mask]
	# log-transform
	xlog = np.log10(x)
	ylog = np.log10(y)

	# sort by x
	order = np.argsort(xlog)
	xlog = xlog[order]
	ylog = ylog[order]

	def neg_log_likelihood(xc):
			# split
			left = xlog < xc
			right = ~left

			# enforce minimum points
			if left.sum() < min_points or right.sum() < min_points:
					return np.inf

			# linear regressions
			s1, i1, _, _, _ = linregress(xlog[left], ylog[left])
			s2, i2, _, _, _ = linregress(xlog[right], ylog[right])

			# predictions
			yhat1 = s1 * xlog[left] + i1
			yhat2 = s2 * xlog[right] + i2

			# RSS
			rss = np.sum((ylog[left] - yhat1)**2) + \
						np.sum((ylog[right] - yhat2)**2)

			return rss

	# optimize breakpoint
	res = minimize_scalar(
			neg_log_likelihood,
			bounds=(xlog.min(), xlog.max()),
			method="bounded"
	)

	xc_opt = res.x

	# final fit with optimal breakpoint
	left = xlog < xc_opt
	right = ~left

	s1, i1, _, _, _ = linregress(xlog[left], ylog[left])
	s2, i2, _, _, _ = linregress(xlog[right], ylog[right])

	fit_params = {
			"alpha1": s1,
			"intercept1": i1,
			"alpha2": s2,
			"intercept2": i2,
			"x_break": 10**xc_opt,
			"x_break_log": xc_opt,
			"edge" : edge_type
	}

	return pd.DataFrame.from_dict([fit_params])


def compute_params(
	edges: gpd.GeoDataFrame,
	traffic_mean: pl.DataFrame, 
	road_length_threshold: int
) -> pd.DataFrame: 
	mask_short = (edges['length'] <= road_length_threshold)
	mask_long = (edges['length'] > road_length_threshold)
 
	edges_short = list(edges[mask_short]['link_id'])
	edges_long = list(edges[mask_long]['link_id'])
 
	traffic_short_roads = traffic_mean.filter(pl.col("link_id").is_in(edges_short))
	traffic_long_roads = traffic_mean.filter(pl.col("link_id").is_in(edges_long))

	fits_param_df = pd.DataFrame()

	if len(edges_short) > 0:
		fits_param_df = fit_piecewise_powerlaw(traffic_short_roads, "short")
	
	if len(edges_long) > 0:
		fits_param_df = (
			pd.concat([
				fits_param_df,
				fit_piecewise_powerlaw(traffic_long_roads, "long")
			], ignore_index=True)
			if len(fits_param_df) > 0
			else fit_piecewise_powerlaw(traffic_long_roads, "long")
		)
	
	return fits_param_df


def main():
	logging.basicConfig(level=logging_level)
	with timed():
		edges = (
			pl.read_csv(PathRegistry.SIM_EDGES)
			[['id', 'u', 'v', 'length', 'maxspeed_mps', 'lanes', 'id_zone']]
			.rename({'id': 'link_id', 'u':'start_node', 'v':'end_node', 'maxspeed_mps':'maxspeed', 'lanes':'n_lanes', 'id_zone':'zone_id'})
			.cast({"link_id": pl.Utf8})
		)

		list_seeds = [i for i in range(context["total_seeds"])]

		logging.info("Computing traffic per edge with the min granularity")
		mat_t_tot = None
		mat_tt_tot = None
		for seed in list_seeds:
			trips = preprocess_trips(
				trips=pl.read_parquet(PathRegistry.SIMULATION_VEHICLES_SEED.substitute(seed=seed)),
				platoons=pl.read_parquet(PathRegistry.PLATOONS_SEED.substitute(seed=seed)),
				edges=edges
			)

			logging.debug(f"computing traffic matrix for seed {seed}")
			mat_t_temp, link_t_mapping = inflow_per_edge_min_granularity(
				trips=trips, edges=edges, time_step=context["time_step"], t_min=context['first_hour']*3600
			)
			if mat_t_tot is None:
				mat_t_tot = mat_t_temp
			else:
				mat_t_tot += mat_t_temp

			logging.debug(f"computing travel-time matrix for seed {seed}")
			mat_tt_temp, link_tt_mapping = travel_time_per_edge_min_granularity(
				trips=trips, edges=edges, time_step=context["time_step"], t_min=context['first_hour']*3600
			)
			if mat_tt_tot is None:
				mat_tt_tot = mat_tt_temp
			else:
				mat_tt_tot += mat_tt_temp
		del mat_t_temp, mat_tt_temp
		gc.collect()

		logging.debug("averaging")
		if context["total_seeds"] > 1:
			mat_t_tot = mat_t_tot // context["total_seeds"]
			mat_tt_tot = mat_tt_tot // context["total_seeds"]
			# TODO: save the results per each seed?

		logging.debug("melting")
		traffic_mean = (
			melt_matrix(mat_t_tot, link_t_mapping, time_step=context["time_step"], t_min=context['first_hour']*3600)
			.rename({"counts": "traffic_volume_mean", "spatial_unit": "link_id"})
			.sort(["link_id", "t"])
		)
		del mat_t_tot
		gc.collect()

		tt_mean = (
			melt_matrix(mat_tt_tot, link_tt_mapping, time_step=context["time_step"], t_min=context['first_hour']*3600)
			.rename({"counts": "average_travel_time_mean", "spatial_unit": "link_id"})
			.sort(["link_id", "t"])
		)
		del mat_tt_tot
		gc.collect()

		logging.debug("merging")
		traffic_mean = traffic_mean.select("link_id", "t","traffic_volume_mean").join(
			tt_mean.select("link_id", "t","average_travel_time_mean"),
			on=["link_id", "t"],
			how="inner"
		)
		del tt_mean
		# TODO (araiari): check the outer join
		gc.collect()

		traffic_mean = traffic_mean.join(
			edges.select([
				"link_id", 
				pl.col("maxspeed").alias("free_speed"), 
				"length", 
				"n_lanes"
			]),
			on="link_id",
			how="left"
		)

		logging.debug("creating columns")
		traffic_mean = traffic_mean.with_columns([
			# Density
			(pl.col("traffic_volume_mean") / pl.col("length")).alias("traffic_density_mean"),
			# Speed
			pl.when(pl.col("average_travel_time_mean") != 0)
			.then(pl.col("length") / pl.col("average_travel_time_mean"))
			.otherwise(0.0)
			.alias("average_speed")
		]).with_columns([
			# Ratio speed and traffic density
			(pl.col("average_speed") / pl.col("free_speed")).alias("ratio_speed"),
			(pl.col("traffic_density_mean") / pl.col("n_lanes")).alias("traffic_lanes_density")
		])

		logging.info("Loading the road network")
		if context["road_network"] == "geo":
			edges = gpd.read_parquet(PathRegistry.GEO_EDGES)
		elif context["road_network"] == "geo_all":
			edges = gpd.read_parquet(PathRegistry.GEO_EDGES_ALL)
		elif context["road_network"] == "dsf":
			edges = gpd.read_parquet(PathRegistry.DSF_EDGES)
		edges = edges.rename(columns={"id": "link_id"}).astype({"link_id": str})

		logging.info("Fitting parameters")
		ft_params = compute_params(
			edges=edges, 
			traffic_mean=traffic_mean, 
			road_length_threshold=context['road_length_threshold']
		)
		ft_params.to_parquet(PathRegistry.NON_LINEAR_FIT_PARAMS)
 

if __name__ == '__main__':
	main()