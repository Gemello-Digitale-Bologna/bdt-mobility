import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, snap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bdt_mobility.utils import data_reader
from bdt_mobility.modules.transport.extraction import road_network
from bdt_mobility.utils.constants import CRS_LATLONG, CRS_PROJECTED

from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import TMP_PATH, logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("pt_metrics_per_zone", PathRegistry.PT_METRICS_PER_ZONE, "table"),
]


## Plot functions
def plot_frequency(points, od_shape, points_zone, n_stop_zone, counts_per_stop, n_trips_hour):
    output_image_dir = TMP_PATH / "output_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    points.plot(ax=ax, markersize=2)
    od_shape.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
    plt.title("Spatial distribution of stops")
    plt.savefig(output_image_dir / "tpm_stop_spatial_location_distribution.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 10))
    od_shape.plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5)
    points_zone['zone_id'] = points_zone['zone_id'].astype(str)
    points_zone.plot(ax=ax, markersize=2, column='zone_id', cmap='Set1')
    points_zone['zone_id'] = points_zone['zone_id'].astype(int)
    plt.title("Spatial distribution of stops colored per assigned zone")
    plt.savefig(output_image_dir / "tpm_stop_spatial_assignment_to_zones.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8,8))
    od_shape.plot(ax=ax, color='grey', edgecolor='black', linewidth=0.5)
    od_shape_plot = od_shape.merge(n_stop_zone, on=['zone_id'], how='outer')
    od_shape_plot['zero_stops'] = od_shape_plot['n_stops'] == 0
    od_shape_plot[od_shape_plot['zero_stops']].plot(ax=ax, color='yellow', edgecolor='black', linewidth=0.5, alpha=0.5)
    plt.title("Zones with zero stops in yellow")
    plt.savefig(output_image_dir / "tpm_stop_spatial_0number_per_zone.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=counts_per_stop, x='counts')
    plt.xlabel("")
    plt.grid()
    plt.title("Histogram of daily counts per stop")
    plt.savefig(output_image_dir / "tpm_stop_histogram_counts_per_stop.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    logging.debug(f"How many stops with no trips: {counts_per_stop[counts_per_stop['counts'] == 0].shape[0]}") # 795 stops with no trips!
    id_zero_trips = counts_per_stop.loc[counts_per_stop['counts'] == 0, 'stop_id'].to_list()
    id_nonzero_trips = counts_per_stop.loc[counts_per_stop['counts'] > 0, 'stop_id'].to_list()
    points_zone[~points_zone['stop_id'].isin(id_zero_trips)].plot(ax=ax, color='black', markersize=1)
    points_zone[points_zone['stop_id'].isin(id_nonzero_trips)].plot(ax=ax, color='blue', markersize=1)
    points_zone[points_zone['stop_id'].isin(id_zero_trips)].plot(ax=ax, color='red', markersize=3)
    od_shape.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
    plt.title("Stops with zero trips")
    plt.savefig(output_image_dir / "tpm_stop_spatial_0frequency_distribution.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=n_trips_hour, x='counts')
    plt.xlabel("")
    plt.grid()
    plt.title("Histogram of n. trips per hour per stop")
    plt.savefig(output_image_dir / "tpm_stop_histogram_hourly_counts_per_stop.png", dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=n_trips_hour, x='frequency')
    plt.xlabel("")
    plt.title("Histogram of frequency of trips per hour per stop")
    plt.grid()
    plt.savefig(output_image_dir / "tpm_stop_histogram_hourly_frequency_per_stop.png", dpi=300, bbox_inches='tight')

def plot_capillarity(od_shape, unique_all_roads, unique_pt_roads, total_length_all_roads, total_length_pt_roads):
    output_image_dir = TMP_PATH / "output_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    od_shape.plot(ax=ax, alpha=0.3, color='grey')
    od_shape['zone_id'] = od_shape['zone_id'].astype(str)
    od_shape.plot(ax=ax, alpha=0.3, column='zone_id', edgecolor='black',cmap='Set1')
    od_shape['zone_id'] = od_shape['zone_id'].astype(int)
    unique_all_roads.plot(ax=ax, color='gray', linewidth=0.5, alpha=1, label='All Roads')
    unique_pt_roads.plot(ax=ax, color='blue', linewidth=1, label='Roads with TPM')
    plt.legend()
    plt.title("Road network and TPM network in the area")
    plt.savefig(output_image_dir / "tpm_capillarity_spatial_distribution_networks.png", dpi=300, bbox_inches='tight')

    logging.debug(f"Length of the road network: {total_length_all_roads}")
    logging.debug(f"Length of the TPM network: {total_length_pt_roads}")
    avg_capillarity = round(float(total_length_pt_roads / total_length_all_roads), 5)
    logging.debug(f"Average capillarity: {avg_capillarity}")

def plot_cost(od_shape, cost_df):
    output_image_dir = TMP_PATH / "output_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    od_shape.plot(ax=ax, alpha=0.3, color='grey')
    gpd.GeoDataFrame(
        od_shape.merge(cost_df, right_on="zone_id", left_on="id"), geometry='geometry'
    ).plot(ax=ax, alpha=0.5, column='cost', edgecolor='black', cmap='viridis', legend=True)
    plt.title("Average cost per area")
    plt.savefig(output_image_dir / "tpm_cost_spatial_distribution.png", dpi=300, bbox_inches='tight')

def plot_time(times, times_ok, method):
    output_image_dir = TMP_PATH / "output_images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    status_matrix = times.drop_duplicates(subset=['from', 'to']).pivot(index='from', columns='to', values='status_nopolicy') != 'success'
    status_not_ok = status_matrix.apply(sum, axis=1).reset_index(name='n_not_ok').sort_values(by='n_not_ok', ascending=False)
    logging.debug(f"There are {len(status_not_ok.loc[status_not_ok['n_not_ok']==1, 'from'])} missing origs:")
    logging.debug(status_not_ok.loc[status_not_ok['n_not_ok']==1, 'from'].to_list())

    plt.figure(figsize=(10,10))
    sns.heatmap(status_matrix, cmap='Set1_r', cbar=True)
    plt.title('Nopolicy Status Matrix (success=False)')
    plt.xlabel('Destination Zone ID')
    plt.ylabel('Origin Zone ID')
    plt.savefig(output_image_dir / f"tpm_times_nopolicy_status_matrix_{method}.png", dpi=300, bbox_inches='tight')

    status_matrix = times.pivot(index=['from', 'origin_lat', 'origin_lon'], columns=['to', 'dest_lat', 'dest_lon'], values='status_policy') != 'success'
    status_not_ok = status_matrix.apply(sum, axis=1).reset_index(name='n_not_ok').sort_values(by='n_not_ok', ascending=False)
    # logging.debug(f"There are {len(status_not_ok.loc[status_not_ok['n_not_ok']==1, 'from'])} missing origins")
    not_ok_start_id = status_not_ok.loc[status_not_ok['n_not_ok']==1, 'from'].to_list()
    # logging.debug(not_ok_start_id)

    plt.figure(figsize=(10,10))
    sns.heatmap(status_matrix, cmap='Set1_r', cbar=True)
    plt.title('Policy Status Matrix (success=False)')
    plt.xlabel('Destination Zone ID')
    plt.ylabel('Origin Zone ID')
    plt.savefig(output_image_dir / f"tpm_times_policy_status_matrix_{method}.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10,10))
    sns.histplot(data=times_ok, x='duration_seconds_diff')
    plt.savefig(output_image_dir / f"tpm_times_difference_distribution_{method}.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10,10))
    sns.kdeplot(data=times_ok, x='duration_seconds_diff', hue="avg_physical_distance_m", legend=False, warn_singular=False)
    plt.savefig(output_image_dir / f"tpm_times_difference_distribution_per_zone_{method}.png", dpi=300, bbox_inches='tight')

    total_flow = times_ok['flow'].sum()
    avg_diff = float((times_ok['duration_seconds_diff'] * times_ok['flow'] / total_flow).sum())
    logging.debug(f"Mean average difference in seconds: {avg_diff}")

## Compute-metrics functions
def compute_frequency(DO_PLOT):
    logging.info("Reading and processing input data")
    # Read the OD shapes
    shapes_in, shapes_out = data_reader.AOI_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od",
            df_around = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER),
            return_type="both"
        )
    shapes_out["type_zoi"] = "outside"
    shapes_in["type_zoi"] = "inside"
    od_shape = gpd.GeoDataFrame(pd.concat([shapes_in, shapes_out]), crs=shapes_in.crs, geometry="geometry").rename(columns={"id_zone":"zone_id"})
    od_shape['zone_id'] = od_shape['zone_id'].astype(int)

    # Read the bus stops from the dataset
    points = (
        gpd.read_parquet(PathRegistry.PT_BUS_STOPS)
        .set_geometry("geopoint")
        [['codice', 'denominazione', 'geopoint']]
        .drop_duplicates()
        .to_crs(CRS_PROJECTED)
        .rename(columns={'codice': 'stop_id', 'denominazione': 'stop_name'})
    )

    # Assign each bus stop to a zone; remove if no association
    points_zone = gpd.sjoin(points, od_shape[['zone_id',"type_zoi", 'geometry']], how="left", predicate="intersects")[['stop_id', 'stop_name', 'geopoint', 'zone_id', "type_zoi"]]
    points_zone = points_zone[~points_zone['zone_id'].isna()]
    points_zone['zone_id'] = points_zone['zone_id'].astype(int)

    # Compute the n. stops per zone
    n_stop_zone = points_zone.groupby(['zone_id',"type_zoi"])['stop_id'].count().reset_index().sort_values(by='stop_id', ascending=False)
    n_stop_zone = n_stop_zone.merge(od_shape[['zone_id',"type_zoi"]], on=['zone_id',"type_zoi"], how='outer').sort_values(by='stop_id', ascending=False)
    n_stop_zone['stop_id'] = n_stop_zone['stop_id'].fillna(0).astype(int)
    n_stop_zone = n_stop_zone.rename(columns={'stop_id': 'n_stops'})

    # Load the scheduling data
    file_scheduling = PathRegistry.PT_SCHEDULING
    data = gpd.read_parquet(file_scheduling)
    data['stop_id'] = data['stop_id'].astype(int)

    logging.info("Merging data sources to compute frequency")
    # Merge the scheduling data with the bus stops and zones
    data = data.merge(points_zone, on=['stop_id', 'stop_name'], how='right')

    # Count the number of trips per stop (and zone)
    counts_per_stop = (
        data
        .groupby(['stop_id', 'zone_id', "type_zoi"])
        .count()
        .reset_index()
        [['stop_id', 'zone_id', 'trip_id',"type_zoi"]]
        .rename(columns={'trip_id': 'counts'})
        .sort_values(by='counts')
    )

    # n. passaggi medi alla fermata all'ora per zona
    n_trips_hour = (counts_per_stop.groupby(['zone_id', "type_zoi"])['counts'].mean()/24).reset_index()
    #logging.debug(n_trips_hour['counts'].describe())

    min_count = n_trips_hour[n_trips_hour['counts'] > 0]['counts'].median()
    n_trips_hour.loc[n_trips_hour['counts'] <= 1/6, 'counts'] = min_count # 1/6 = 4 per day

    n_trips_hour['frequency'] = n_trips_hour['counts'].apply(lambda x: -1 if x==0 else 1/x)
    #logging.debug(n_trips_hour['frequency'].describe())

    # frequenza media oraria in generale
    avg_frequency = n_trips_hour['frequency'].median()
    float(avg_frequency)

    # Create the output: frequency_df
    frequency_df = n_trips_hour[['zone_id', "type_zoi", 'frequency', 'counts']].rename(columns={'counts': 'n_trips'})

    if DO_PLOT:
        logging.info("Plotting")
        plot_frequency(points, od_shape, points_zone, n_stop_zone, counts_per_stop, n_trips_hour)

    return frequency_df

def compute_capillarity(DO_PLOT):
    '''
    '''
    logging.info("Processing the pt data")
    # Read the shp of the trips
    file_scheduling = PathRegistry.PT_SCHEDULING
    shp = gpd.read_parquet(file_scheduling)

    # Pre-process the linestrings
    shp = shp.rename(columns={"shape_linestring": "geometry"}).set_geometry("geometry")
    shp = shp.to_crs(CRS_PROJECTED)
    gdf = shp[~shp['geometry'].isnull()][['geometry']]

    # Calculate the length of the PT road newtork
    unique_pt_roads, total_length_pt_roads = _process_lines(gdf=gdf)

    shapes_in, shapes_out = data_reader.AOI_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od",
            df_around = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER),
            return_type="both"
        )
    shapes_out["type_zoi"] = "outside"
    shapes_in["type_zoi"] = "inside"
    od_shape = gpd.GeoDataFrame(pd.concat([shapes_in, shapes_out]), crs=shapes_in.crs, geometry="geometry")
    od_shape = od_shape.rename(columns={'id_zone': 'zone_id', 'code': 'zone_name'})
    od_shape['zone_id'] = od_shape['zone_id'].astype(int)

    length_pt_roads_zones = unique_pt_roads.sjoin(od_shape, how='left', predicate='intersects')[['zone_id',"type_zoi", 'geometry']]
    length_pt_roads_zones['length'] = length_pt_roads_zones.geometry.length
    length_pt_roads_zones = length_pt_roads_zones.groupby(['zone_id',"type_zoi"])['length'].sum().reset_index()
    length_pt_roads_zones['zone_id'] = length_pt_roads_zones['zone_id'].astype(int)

    length_pt_roads_zones.head()

    logging.info("Processing the road network data")
    # Create a convex hull around the roads (to extract roads from OSM)
    unique_pt_roads = unique_pt_roads[['geometry']]
    unique_pt_roads = unique_pt_roads.to_crs(CRS_LATLONG)

    merged = unary_union(unique_pt_roads.geometry)
    convex_hull = merged.convex_hull

    unique_pt_roads = unique_pt_roads.to_crs(CRS_PROJECTED)

    # Download the road newtork from OSM
    # TODO: use the version of the road network compatible with the version of the data (=same year, same period)
    convex_hull_gdf = gpd.GeoDataFrame({"geometry": [convex_hull] }, crs=CRS_LATLONG)
    rn2, _ = road_network.create_road_data(convex_hull_gdf, relevant_highway=False, connected_network=False)
    rn2 = rn2.reset_index()[['u','v','key','length','geometry']]

    # Calculate the length of the road newtork
    gdf = rn2[['geometry']]
    gdf = gdf.to_crs(CRS_PROJECTED)

    unique_all_roads, total_length_all_roads = _process_lines(gdf=gdf)

    logging.info("Merging the two sources of information")
    length_all_roads_zones = unique_all_roads.sjoin(od_shape, predicate='intersects')[['zone_id', "type_zoi",'geometry']]
    length_all_roads_zones['length'] = length_all_roads_zones.geometry.length
    length_all_roads_zones = length_all_roads_zones.groupby(['zone_id',"type_zoi"])['length'].sum().reset_index()

    capillarity_df = pd.merge(
        length_pt_roads_zones.rename(columns={'length':'length_pt'}),
        length_all_roads_zones.rename(columns={'length': 'length_all'})
    )
    capillarity_df['capillarity'] = capillarity_df['length_pt'] / capillarity_df['length_all']
    capillarity_df = capillarity_df.drop(columns=['length_pt','length_all'])

    if DO_PLOT:
        logging.info("Plotting")
        plot_capillarity(od_shape, unique_all_roads, unique_pt_roads, total_length_all_roads, total_length_pt_roads)

    return capillarity_df

def compute_cost(DO_PLOT):

    logging.info("Reading and preparing the input data")
    # Read the zoning file
    shapes_in, shapes_out = data_reader.AOI_shapes(
            PathRegistry.OD_SHAPE_ZONE,
            aoi_type="od",
            df_around = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER),
            return_type="both"
        )
    shapes_out["type_zoi"] = "outside"
    shapes_in["type_zoi"] = "inside"
    location_od_areas = gpd.GeoDataFrame(pd.concat([shapes_in, shapes_out]), crs=shapes_in.crs, geometry="geometry").reset_index(drop=True)
    location_od_areas = location_od_areas.rename(columns={'id_zone': 'id', 'code': 'zone_name'})
    location_od_areas['id'] = location_od_areas['id'].astype(int)
    location_od_areas["temp_id"] = location_od_areas.index.astype(int)
    index_od_in = location_od_areas.loc[
        location_od_areas['type_zoi']=="inside", 'temp_id'
    ].to_list()

    # Read the cost-locality file
    file2 = PathRegistry.PT_COST_ZONES
    location_tarifs = pd.read_parquet(file2).reset_index(drop=True)
    location_tarifs["temp_id"] = location_tarifs.index.astype(int)

    # Build a geoDataFrame
    geometry = [Point(xy) for xy in zip(location_tarifs['lon'], location_tarifs['lat'])]
    geo_location_tarifs = gpd.GeoDataFrame(location_tarifs[["temp_id"]], geometry=geometry, crs=CRS_LATLONG).to_crs(CRS_PROJECTED)
    geo_location_tarifs = geo_location_tarifs[~location_tarifs['lat'].isnull()]

    # Build a dictionary with key: temp_id, value: cost
    location_tarifs_mapping = dict(zip(location_tarifs['temp_id'], location_tarifs['cost']))

    logging.info("Processing data: getting the zonal cost")
    # Now, for each zone in location_od_areas, find the list of the elements in geo_location_tarifs that are within 5km, and assign to the zone the average cost of those elements. If there are no elements within 5km, assign the closest in general.
    zone_cost_mapping = {}
    for _, area_row in location_od_areas.iterrows():
        area = area_row.geometry
        area_id = area_row['temp_id']

        distances = geo_location_tarifs.geometry.distance(area)
        nearby_points = geo_location_tarifs[distances < 5000]

        if not nearby_points.empty:
            nearby_costs = nearby_points['temp_id'].map(location_tarifs_mapping)
            zone_cost_mapping[area_id] = round(nearby_costs.mean(), 2)
        else:
            closest_idx = distances.idxmin()
            closest_cost = location_tarifs_mapping[geo_location_tarifs.loc[closest_idx, 'temp_id']]
            zone_cost_mapping[area_id] = round(closest_cost, 2)

    ## Correct: all the areas inside ZOI have the same cost, equal to 2.3
    for key in index_od_in:
        zone_cost_mapping[key] = location_tarifs["cost"].min() # 2.3

    logging.info("Preparing and returning the cost df")
    cost_df = (
        pd.DataFrame
        .from_dict(zone_cost_mapping, orient='index', columns=['cost'])
        .reset_index()
        .rename(columns={'index': 'temp_id'})
        .merge(location_od_areas[["temp_id", "id", "type_zoi"]], on="temp_id", how="right")
        .rename(columns={"id": "zone_id"})
        .drop(columns="temp_id")
    )
    logging.debug(f"How many missing matches: ", cost_df["zone_id"].isna().sum())

    if DO_PLOT:
        logging.info("Plotting")
        plot_cost(location_od_areas, cost_df)

    return cost_df

def compute_time_difference_inside_and_outside(DO_PLOT):
    # Case 1: outside to inside data
    od_points = pd.read_parquet(PathRegistry.OTP_OD_COORDS_EXTENDED)
    nopolicy_times = pd.read_parquet(PathRegistry.OTP_RESULTS_EXTENDED_ALL_BOLOGNA).drop(columns="row_index")
    policy_times = pd.read_parquet(PathRegistry.OTP_RESULTS_EXTENDED_RESTRICTED_ZOI).drop(columns="row_index")

    times_outside = _compute_time_difference(nopolicy_times=nopolicy_times, policy_times=policy_times, od_points=od_points)
    times_outside["type_zoi"] = "outside"
    times_outside = times_outside[["from", "to", "type_zoi", 'duration_seconds_diff', 'duration_perc_diff']]
    times_outside_mean = _compute_time_average(times=times_outside, method="outside", DO_PLOT=DO_PLOT)

    # Case 2: inside to inside data
    od_points = pd.read_parquet(PathRegistry.OTP_OD_COORDS_ZOI)
    nopolicy_times = pd.read_parquet(PathRegistry.OTP_RESULTS_ZOI_ALL_BOLOGNA_CPT).drop(columns="row_index")
    policy_times = pd.read_parquet(PathRegistry.OTP_RESULTS_ZOI_ALL_BOLOGNA_TRANSIT).drop(columns="row_index")

    times_inside = _compute_time_difference(nopolicy_times=nopolicy_times, policy_times=policy_times, od_points=od_points)
    times_inside["type_zoi"] = "inside"
    times_inside = times_inside[["from", "to", "type_zoi", 'duration_seconds_diff', 'duration_perc_diff']]
    times_inside_mean = _compute_time_average(times=times_inside, method="inside", DO_PLOT=DO_PLOT)

    # Case 3: inside to outside data -> assumption of equality in AR times
    times_in_out = pd.DataFrame(data={
        "from": times_outside["to"],
        "to": times_outside["from"],
        "duration_seconds_diff": times_outside["duration_seconds_diff"],
        "duration_perc_diff": times_outside["duration_perc_diff"]})
    times_in_out["type_zoi"] = "inside"
    times_in_out_mean = _compute_time_average(times=times_in_out, method="inside", DO_PLOT=DO_PLOT)

    times_inside_mean = _compute_time_average_inside(times1=times_inside_mean, times2=times_in_out_mean)

    return pd.concat([times_outside_mean, times_inside_mean])


def _compute_time_difference(nopolicy_times, policy_times, od_points):

    times = pd.merge(
        nopolicy_times.drop(columns=['origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']),
        policy_times.drop(columns=['origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']),
        on=['from', 'to'],
        suffixes=['_nopolicy', '_policy'],
        how='inner'
    )
    times = times.merge(
        od_points,
        on=['from', 'to'],
        how='left'
    )

    times['duration_seconds_diff'] = (times['duration_seconds_nopolicy'] - times['duration_seconds_policy'])
    times['duration_perc_diff'] = (times['duration_seconds_nopolicy'] - times['duration_seconds_policy']) / (5*60*60)

    valid = times[['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']].notnull().all(axis=1)
    times.loc[valid, 'physical_distance_m'] = _haversine_m(
        times.loc[valid, 'origin_lat'].to_numpy(),
        times.loc[valid, 'origin_lon'].to_numpy(),
        times.loc[valid, 'dest_lat'].to_numpy(),
        times.loc[valid, 'dest_lon'].to_numpy(),
    )
    times["avg_physical_distance_m"] = times.groupby("from")["physical_distance_m"].transform("mean")

    return times

def _compute_time_average(times, method, DO_PLOT):

    times_ok = times.copy()
    times_ok['duration_seconds_diff'] = times_ok['duration_seconds_diff'].fillna(-5*60*60)
    times_ok['duration_perc_diff'] = times_ok['duration_perc_diff'].fillna(-1)

    #Mean value per zone
    times_avg = (
        times_ok
        [['from', "type_zoi", 'duration_seconds_diff', 'duration_perc_diff']]
        .groupby(["from", "type_zoi"])
        .mean()
        .reset_index()
    ).merge(
            (times_ok
        [['from',"type_zoi", 'duration_seconds_diff', 'duration_perc_diff']]
        .groupby(["from", "type_zoi"])
        .std()
        .reset_index()),
        on=["from", "type_zoi"],
        suffixes=["_mean", "_std"]
    )

    if DO_PLOT:
        plot_time(times, times_ok, method)

    return times_avg

def _compute_time_average_inside(times1, times2):
    io_totals = pd.read_parquet(PathRegistry.TOTAL_FLOWS_FROM_OD)
    TI = float(io_totals.loc[(io_totals["from_zoi"]==True)&(io_totals["to_zoi"]==True), "flow"].values[0]) # Total inside
    TO = float(io_totals.loc[(io_totals["from_zoi"]==True)&(io_totals["to_zoi"]==False), "flow"].values[0]) # Total outflow

    times1["weight"] = TI/(TI+TO)
    times2["weight"] = TO/(TI+TO)
    times = pd.concat([times1, times2])
    times["duration_seconds_diff_mean"] = times["duration_seconds_diff_mean"] * times["weight"]
    times["duration_seconds_diff_std"] = times["duration_seconds_diff_std"]**2 * times["weight"]
    times["duration_perc_diff_mean"] = times["duration_perc_diff_mean"] * times["weight"]
    times["duration_perc_diff_std"] = times["duration_perc_diff_std"]**2 * times["weight"]
    times = times.groupby(["from", "type_zoi"])[
        ["duration_seconds_diff_mean", "duration_perc_diff_mean",
        "duration_seconds_diff_std", "duration_perc_diff_std"]
    ].sum().reset_index()
    times["duration_perc_diff_std"] = times["duration_perc_diff_std"]**(1/2)
    times["duration_seconds_diff_std"] = times["duration_seconds_diff_std"]**(1/2)
    return times

## Helper functions
def _process_lines(gdf, snap_tolerance=1e-8):
    # 1. Keep only geometry and reproject
    gdf = gdf[['geometry']]

    # 1.5: Remove directionality
    gdf['geometry'] = gdf['geometry'].apply(_normalize_linestring_direction)

    # 2. Remove exact duplicates
    gdf = gdf.drop_duplicates(subset='geometry')

    # 3. Snap geometries to themselves (cleans tiny gaps)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: snap(geom, geom, tolerance=snap_tolerance))

    # 4. Merge into a MultiLineString
    merged = unary_union(gdf.geometry)
    if isinstance(merged, LineString):
        merged = MultiLineString([merged])
    elif isinstance(merged, (list, tuple)):
        merged = MultiLineString(merged)

    # 5. Explode and drop overlapping segments
    exploded = gpd.GeoSeries(merged).explode(index_parts=False)
    unique = exploded.drop_duplicates()

    # 6. Calculate total length
    total_length = unique.length.sum()

    # Return as GeoDataFrame
    unique_gdf = gpd.GeoDataFrame(geometry=unique, crs=gdf.crs)
    return unique_gdf, total_length

def _normalize_linestring_direction(geom):
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return LineString(coords if coords < coords[::-1] else coords[::-1])
    return geom

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def main(DO_PLOT=False):
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Computing all metrics: frequency, capillarity, travel time and cost")

        logging.info("Compute Frequency")
        frequency_df = compute_frequency(DO_PLOT)

        logging.info("Compute Capillarity")
        capillarity_df = compute_capillarity(DO_PLOT)

        logging.info("Compute Cost")
        cost_df = compute_cost(DO_PLOT)

        logging.info("Compute Times")
        times_df = compute_time_difference_inside_and_outside(DO_PLOT)

        logging.info("Merge all the statistics")
        shapes_in, shapes_out = data_reader.AOI_shapes(
                PathRegistry.OD_SHAPE_ZONE,
                aoi_type="od",
                df_around = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER),
                return_type="both"
            )
        shapes_out["type_zoi"] = "outside"
        shapes_in["type_zoi"] = "inside"
        od_shape = pd.concat([shapes_in, shapes_out])[["id_zone", "type_zoi"]].rename(columns={"id_zone": "zone_id"})
        od_shape["zone_id"] = od_shape["zone_id"].astype(int)

        df = pd.merge(
            od_shape,
            times_df.rename(columns={'from': 'zone_id', 'duration_seconds_diff': 'diff_time_seconds', 'duration_perc_diff': 'diff_time_perc'}),
            on=['zone_id', 'type_zoi'],
            how='left'
        ).merge(
            frequency_df,
            on=['zone_id', "type_zoi"],
            how='left'
        ).merge(
            capillarity_df,
            on=['zone_id', "type_zoi"],
            how='left'
        ).merge(
            cost_df.rename(columns={'cost': 'cost_euro'}),
            on=['zone_id', "type_zoi"],
            how='left'
        )

        logging.info("Fix/filter statistics")
        avg_cost = cost_df["cost"].mean()
        df['cost_euro'] = df['cost_euro'].fillna(round(avg_cost, 2)) # if not present, it means that no association between two data sources was possible -> avg cost
        df['n_trips'] = df['n_trips'].fillna(round(0, 2)) # if not present, it means that there is no bus lines there -> zero n_trips
        max_frequency = frequency_df["frequency"].max()
        df['frequency'] = df['frequency'].fillna(round(max_frequency, 6)) # if not present, it means that there is no bus lines there -> infinite frequency
        df['capillarity'] = df['capillarity'].fillna(0.0) # if not present, it means that there is no bus lines there -> zero capillarity

        logging.info("Saving the files")
        file_output = PathRegistry.PT_METRICS_PER_ZONE
        df.to_parquet(file_output)
        logging.info("All metrics computed and saved")


if __name__ == "__main__":
    main()