from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("spira_traffic_estimated", PathRegistry.SPIRA_TRAFFIC_ESTIMATED, "table"),
]

from datetime import datetime
import math
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from bdt_mobility.utils import data_reader
from bdt_mobility.modules.transport.extraction import spatial_assignment
from bdt_mobility.utils import data_cleaner
from bdt_mobility.utils import temporal_utils


def get_traffic_from_spiras(use_daytype, aoi_type, start_date, end_date, time_unit):

    # Define dependent parameters
    if aoi_type in ["zoi", "road"]:
        path_aoi = PathRegistry.ZOI_BORDER
        use_zonegroup = False
    elif aoi_type == "od":
        path_aoi = PathRegistry.OD_SHAPE_ZONE
        use_zonegroup = True

    logging.info("Reading the data")
    # Load ZOI
    gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)

    # Load the area(s) of interest, aoi
    areas, _ = data_reader.AOI_shapes(
        path_aoi, aoi_type=aoi_type,
        df_around=gdf_zoi, return_type="both")

    # Load the spira data
    spira_loc = data_reader.spira_shapes(PathRegistry.SPIRA_FLOW_DATA, df_around=gdf_zoi)
    spira_ids = data_reader.spira_codes(PathRegistry.SPIRA_FLOW_DATA)

    # Load the road data
    edges = gpd.read_parquet(PathRegistry.GEO_EDGES_ALL)
    nodes = gpd.read_parquet(PathRegistry.GEO_NODES_ALL)

    logging.info("Transforming the data")
    logging.debug("...Spatial assignment")
    # Assign edges to aoi, then spira to edges
    edges_aoi = spatial_assignment.roads_to_AOI(df_edges=edges, df_aoi=areas)
    spira_close_roads = spatial_assignment.spira_to_road(df_spira=spira_loc, df_edge=edges_aoi)
    spira_close_roads = spira_close_roads.merge(
        edges_aoi[['u', 'v', 'key', 'is_relevant']], on=['u', 'v', 'key'], how='left')

    logging.debug("...Filtering")
    # Filter: elements on relevant highways only
    spira_close_roads = spira_close_roads[spira_close_roads['is_relevant'].astype('boolean').fillna(False)]
    spira_close_roads = spira_close_roads.reset_index(drop=True).drop(['is_relevant'], axis=1)

    edges_aoi_ok = edges_aoi[edges_aoi['is_relevant']]
    nodes = nodes[(nodes['id'].isin(edges_aoi_ok['u'].values)) | (nodes['id'].isin(edges_aoi_ok['v'].values))]

    logging.info("Catchment areas")
    logging.debug("...Computing")
    # Catchment time threshold, needed before computing the catchment areas so that
    # the shortest-path search can stop expanding at the threshold instead of
    # covering the whole network.
    tt = time_unit * 60 # time unit in seconds
    traffic_effect = 0.25 ## To be changed!!
    tt_mod = tt * traffic_effect

    # Compute the catchment areas of the spiras
    PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    spira_catchment_area = find_catchment_area(
        df_spiras=spira_close_roads, df_edges=edges_aoi_ok, df_nodes=nodes,
        time_threshold=tt_mod)

    # Filter the catchment areas of the spiras
    logging.debug("...Filtering")
    spira_catchment_area = filter_catchment_area(
        df_catch=spira_catchment_area,
        time_threshold=tt_mod
    )

    logging.info("Compute istantaneous traffic from spira to aoi")
    logging.debug("...Reading spira data")
    # Prepare filters to read spira data
    spira_code_ok = pd.merge(spira_close_roads['spira_unique_id'], spira_ids, how='left')['spira_code'].to_list()

    sel = [('start', '>=', start_date.strftime("%Y-%m-%d %H:%M")),
            ('start', '<=', end_date.strftime("%Y-%m-%d %H:%M")),
            ('sensor_id', 'in', spira_code_ok)]

    # Read and clean spira data
    spira5m = data_reader.spira_flows(PathRegistry.SPIRA_FLOW5M_DATA, filters=sel)
    spira5m = pd.merge(
        spira_ids.rename({'spira_code': 'sensor_id'}, axis=1),
        spira5m,
        on='sensor_id', how='inner')
    spira5m = data_cleaner.spira_flows(df_spiras=spira5m)

    logging.debug("...Compute 5m traffic per aoi")
    if aoi_type in ["od", "zoi"]:
        # Compute the traffic given the catchment areas
        traffic = compute_traffic_per_area(
            spira_data=spira5m,
            df_catch=spira_catchment_area,
            first_datetime=start_date,
            last_datetime=end_date,
            deltaTime=time_unit,
            verbose=True
        )
        traffic.rename(columns={'count_distributed': 'traffic_index'}, inplace=True)
        if aoi_type == "od":
            traffic = traffic[traffic["id_zone"]!="0"]

        # logging.debug("...Saving intermediate results") #Debug only
        # PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
        # traffic.to_parquet(PathRegistry.ANALYSIS_OUTPUTS / f"traffic_5m_estimation__aoi_{aoi_type}.parquet")

    elif aoi_type == "road":
        traffic = compute_traffic_per_road(
            spira_data=spira5m,
            df_catch=spira_catchment_area,
            first_datetime=start_date,
            last_datetime=end_date,
            deltaTime=time_unit,
            verbose=True
        )
        traffic.rename(columns={'count_distributed': 'traffic_index'}, inplace=True)

        # logging.debug("...Saving intermediate results") #Debug only
        # PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True) #Debug only
        # traffic.to_parquet(PathRegistry.ANALYSIS_OUTPUTS / f"traffic_5m_estimation_roads__aoi_{aoi_type}.parquet") #Debug only

    logging.info("Computing mean traffic per aoi")
    # Average per daytype and/or zone
    traffic_mean = average_traffic(
        dataset=traffic,
        use_daytype=use_daytype,
        use_zonegroup=use_zonegroup
    )
    traffic_mean_smooth = smooth_traffic(
        dataset=traffic_mean,
        use_daytype=use_daytype,
        use_zonegroup=use_zonegroup,
        method='savgol'
    )

    logging.info("Saving final results")
    traffic_df = pd.DataFrame(
        {"Weekday": 12 * np.array(traffic_mean_smooth
            [traffic_mean_smooth['DayType'] == "Weekday"]
            ['traffic_index']
            .to_list()),
         "Saturday": 12 * np.array( traffic_mean_smooth
            [traffic_mean_smooth['DayType'] == "Saturday"]
            ['traffic_index']
            .to_list()),
         "Holiday": 12 * np.array( traffic_mean_smooth
            [traffic_mean_smooth['DayType'] == "Holiday"]
            ['traffic_index']
            .to_list())}
    )
    PathRegistry.ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    traffic_df.to_parquet(PathRegistry.SPIRA_TRAFFIC_ESTIMATED)

    return traffic_df


def find_catchment_area(
    df_nodes,
    df_edges,
    df_spiras,
    time_threshold: float,
)-> pd.DataFrame:
    """
    Finds the catchment area for each spira by calculating time distances on the road
    network at free flow speed.

    Only (spira, edge) pairs strictly closer than `time_threshold` seconds are returned:
    passing the threshold here (instead of filtering afterwards) lets Dijkstra stop
    expanding at the threshold, and the searches run only from the spira edges'
    endpoints — the only sources whose distances are ever needed — rather than from
    every node of the network.
    """
    node_index = pd.Index(df_nodes['id'].values)
    edge_u = node_index.get_indexer(df_edges['u'].values)
    edge_v = node_index.get_indexer(df_edges['v'].values)
    assert (edge_u >= 0).all() and (edge_v >= 0).all(), "edge endpoints missing from df_nodes"

    # Sparse adjacency matrix of the road network, weighted by free-flow time.
    # Parallel edges (same u and v, different key) must first collapse to their
    # minimum weight: shortest paths only ever traverse the fastest parallel edge,
    # and csr_matrix would otherwise sum the duplicated entries.
    edge_weights = (
        pd.DataFrame({'u': edge_u, 'v': edge_v, 'w': df_edges['free_flow_time'].values})
        .groupby(['u', 'v'], as_index=False)['w'].min()
    )
    n_nodes = len(node_index)
    adjacency = csr_matrix(
        (edge_weights['w'], (edge_weights['u'], edge_weights['v'])),
        shape=(n_nodes, n_nodes))

    spira_nodes = pd.Index(pd.unique(np.concatenate([df_spiras['u'].values, df_spiras['v'].values])))
    sources = node_index.get_indexer(spira_nodes.values)
    # Distances from the spira endpoints to every node (dist_from), and from every
    # node to the spira endpoints (dist_to, via the transposed graph). Distances
    # beyond `limit` are returned as inf and get dropped by the threshold below.
    dist_from = dijkstra(adjacency, directed=True, indices=sources, limit=time_threshold)
    dist_to = dijkstra(adjacency.T, directed=True, indices=sources, limit=time_threshold)

    spira_u = spira_nodes.get_indexer(df_spiras['u'].values)
    spira_v = spira_nodes.get_indexer(df_spiras['v'].values)

    # For each (spira, edge) pair, the catchment distance is the shortest path between
    # any endpoint of the spira's edge and any endpoint of the candidate edge, in
    # either direction (the graph is directed): the minimum over the 8 combinations.
    from_u, from_v = dist_from[spira_u], dist_from[spira_v]
    to_u, to_v = dist_to[spira_u], dist_to[spira_v]
    duration = np.minimum.reduce([
        from_u[:, edge_u], from_u[:, edge_v], from_v[:, edge_u], from_v[:, edge_v],
        to_u[:, edge_u], to_u[:, edge_v], to_v[:, edge_u], to_v[:, edge_v],
    ])

    spira_rows, edge_cols = np.nonzero(duration < time_threshold)
    spira_catchment_area = pd.DataFrame({
        'u': df_edges['u'].values[edge_cols],
        'v': df_edges['v'].values[edge_cols],
        'key': df_edges['key'].values[edge_cols],
        'spira_unique_id': df_spiras['spira_unique_id'].values[spira_rows],
        'duration': duration[spira_rows, edge_cols],
        'free_flow_time': df_edges['free_flow_time'].values[edge_cols],
        'id_zone': df_edges['id_zone'].values[edge_cols],
    })

    return spira_catchment_area


def filter_catchment_area(
    df_catch: pd.DataFrame,
    time_threshold: float,
) -> pd.DataFrame:
    """
    Filters the spira catchment area based on a time threshold and calculates time-proportional weights.
    """
    # Filter close roads only
    df_catch = df_catch[df_catch['duration'] < time_threshold]
    # Compute time-proportional weight. Its factors free_flow_time and tot_t_imp are
    # kept in the output: the traffic computation needs them separately (see
    # _spira_set_medians).
    df_catch = (
        df_catch
        .assign(
            tot_t_imp = lambda df: df.groupby('spira_unique_id')['free_flow_time'].transform('sum'),
            prop_t_imp = lambda df: df['free_flow_time'] / df['tot_t_imp']
        )
    )
    return df_catch


# Caps the number of float64 elements materialized per np.nanmedian call in
# _spira_set_medians (2**26 elements = 512MB), bounding peak memory however many
# edges share contributing-spira sets of the same size.
_MEDIAN_BATCH_ELEMENTS = 2**26


def _spira_set_medians(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame,
    first_datetime: dt.datetime,
    last_datetime: dt.datetime,
    deltaTime: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """
    Shared core of compute_traffic_per_road / compute_traffic_per_area.

    The quantity to compute is, for every catchment edge e and deltaTime-minute time
    bucket b, the median over the edge's contributing spiras s of
        count_distributed(s, e, b) = count(s, b) * prop_t_imp(s, e).
    Since prop_t_imp(s, e) = free_flow_time(e) / tot_t_imp(s), and free_flow_time(e)
    is a positive constant within the median, it factors out exactly:
        median_s[count(s, b) * prop_t_imp(s, e)]
            = free_flow_time(e) * median_s[count(s, b) / tot_t_imp(s)].
    The median itself thus depends only on the *set* of spiras contributing to the
    edge, so it is computed once per distinct spira set — typically far fewer sets
    than edges, and far cheaper than the (catchment x time buckets) merge a direct
    implementation needs.

    A spira with no observation in a bucket is excluded from that bucket's median
    (NaN in the count matrix + nanmedian); buckets where none of a set's spiras have
    data yield NaN, and callers emit no row for them. Both match the inner-merge
    semantics of the original per-interval implementation.

    Returns:
        edges_sets: one row per catchment edge, with columns `u`, `v`, `key`,
            `id_zone`, `free_flow_time` and `set_id` (row index into `medians`).
        medians: (n_sets x n_buckets) array of median normalized counts.
        bucket_times: DatetimeIndex mapping bucket number to interval start.
    """
    delta = pd.Timedelta(minutes=deltaTime)
    window_end = last_datetime + delta
    n_buckets = math.ceil((window_end - first_datetime) / delta)
    bucket_times = pd.date_range(start=first_datetime, periods=n_buckets, freq=delta)

    spira_index = pd.Index(df_catch['spira_unique_id'].drop_duplicates().values)

    # Summed counts per (spira, bucket); NaN marks buckets where a spira has no
    # observation at all, so that it can be excluded from the medians below.
    spira_data = spira_data[spira_data['DateTime'].between(first_datetime, window_end, inclusive='left')]
    spira_pos = spira_index.get_indexer(spira_data['spira_unique_id'].values)
    bucket = ((spira_data['DateTime'] - first_datetime) // delta).values
    counts = np.full((len(spira_index), n_buckets), np.nan)
    summed = (
        pd.DataFrame({'s': spira_pos, 'b': bucket, 'count': spira_data['count'].values})
        [spira_pos >= 0]  # spiras without a catchment area cannot contribute
        .groupby(['s', 'b'])['count'].sum()
    )
    counts[summed.index.get_level_values('s'), summed.index.get_level_values('b')] = summed.values

    tot_t_imp = (
        df_catch.drop_duplicates('spira_unique_id')
        .set_index('spira_unique_id')['tot_t_imp']
        .reindex(spira_index)
        .values
    )
    ratios = counts / tot_t_imp[:, None]

    # One row per catchment edge, with the (canonically ordered) set of spiras
    # contributing to it.
    edges_sets = (
        df_catch
        .assign(_s=spira_index.get_indexer(df_catch['spira_unique_id'].values))
        .sort_values('_s')
        .groupby(['u', 'v', 'key'])
        .agg(id_zone=('id_zone', 'first'),
             free_flow_time=('free_flow_time', 'first'),
             spira_set=('_s', tuple))
        .reset_index()
    )
    set_ids, unique_sets = pd.factorize(edges_sets['spira_set'])
    edges_sets['set_id'] = set_ids
    edges_sets = edges_sets.drop(columns='spira_set')

    # Median per distinct spira set: sets of equal size are stacked into one
    # (n_sets, set_size, n_buckets) block so nanmedian runs vectorized, in batches
    # capping the block's size.
    medians = np.empty((len(unique_sets), n_buckets))
    set_sizes = np.fromiter((len(s) for s in unique_sets), dtype=int, count=len(unique_sets))
    with warnings.catch_warnings():
        # All-NaN buckets are expected and must stay NaN
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for size in np.unique(set_sizes):
            set_pos = np.nonzero(set_sizes == size)[0]
            spira_rows = np.array([unique_sets[i] for i in set_pos])
            batch = max(1, _MEDIAN_BATCH_ELEMENTS // (size * n_buckets))
            for start in range(0, len(set_pos), batch):
                sel = set_pos[start:start + batch]
                medians[sel] = np.nanmedian(ratios[spira_rows[start:start + batch]], axis=1)

    return edges_sets, medians, bucket_times


def compute_traffic_per_road(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame,
    first_datetime: dt.datetime,
    last_datetime: dt.datetime,
    deltaTime: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes distributed traffic counts per road for specified time intervals.

    Note the output holds one row per (catchment edge, time bucket) with data, so it
    inherently scales with the product of the two.
    """
    edges_sets, medians, bucket_times = _spira_set_medians(
        spira_data, df_catch, first_datetime, last_datetime, deltaTime)

    # Scale each set's median back to its edges (the free_flow_time factored out of
    # the median, see _spira_set_medians).
    values = edges_sets['free_flow_time'].values[:, None] * medians[edges_sets['set_id'].values]

    # Emit no row for buckets without data; nonzero on the transpose orders the rows
    # by DateTime first, as the per-interval implementation did.
    bucket_rows, edge_cols = np.nonzero(~np.isnan(values.T))
    counts_roads = pd.DataFrame({
        'u': edges_sets['u'].values[edge_cols],
        'v': edges_sets['v'].values[edge_cols],
        'key': edges_sets['key'].values[edge_cols],
        'id_zone': edges_sets['id_zone'].values[edge_cols],
        'count_distributed': values[edge_cols, bucket_rows],
        'DateTime': bucket_times[bucket_rows],
    })
    return counts_roads


def compute_traffic_per_area(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame,
    first_datetime: dt.datetime,
    last_datetime: dt.datetime,
    deltaTime: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes aggregated traffic counts per area for specified time intervals.
    """
    edges_sets, medians, bucket_times = _spira_set_medians(
        spira_data, df_catch, first_datetime, last_datetime, deltaTime)

    # Summing free_flow_time(e) * median(set(e), b) over a zone's edges only needs
    # each spira set's total free-flow time within the zone.
    zone_weights = edges_sets.groupby(['id_zone', 'set_id'], as_index=False)['free_flow_time'].sum()
    has_data = ~np.isnan(medians)
    medians_filled = np.where(has_data, medians, 0.0)

    frames = []
    for id_zone, group in zone_weights.groupby('id_zone'):
        totals = group['free_flow_time'].values @ medians_filled[group['set_id'].values]
        # A bucket where none of the zone's spiras have data produces no row
        # (matching the original per-interval implementation), rather than a zero.
        present = has_data[group['set_id'].values].any(axis=0)
        frames.append(pd.DataFrame({
            'DateTime': bucket_times[present],
            'id_zone': id_zone,
            'count_distributed': totals[present],
        }))
    counts_areas = (
        pd.concat(frames, ignore_index=True)
        .sort_values(['DateTime', 'id_zone'], ignore_index=True)
    )
    return counts_areas


def average_traffic(
    dataset: pd.DataFrame,
    use_daytype: bool = False,
    use_zonegroup: bool = False
) -> pd.DataFrame:
    """
    Calculates the average traffic index in 5-minute intervals, optionally grouping by day type and zone group.
    """
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
    dataset['Time'] = dataset['DateTime'].dt.time

    if use_daytype:
        dataset['DayType'] = temporal_utils.add_daytype_to_series(dataset['DateTime'])
    else:
        dataset['DayType'] = 'All'
    if not use_zonegroup:
        dataset['id_zone'] = 'All'
    traffic = dataset.groupby(['Time', 'DayType', 'id_zone'])['traffic_index'].mean().reset_index()

    return traffic


def smooth_traffic (
    dataset: pd.DataFrame,
    use_daytype: bool = False,
    use_zonegroup: bool = False,
    method: str = "savgol",
) -> pd.DataFrame:
    """
    Smoothens the traffic data using specified methods, optionally grouping by day type and zone group.
    """
    def smoothing_function(noisy, method=['splines', 'savgol']):
        if method=='splines':
            xx = np.array(range(len(noisy)))
            spl = make_smoothing_spline(
                x=xx,
                y=np.array(noisy),
                lam=50)
            denoised = spl(xx)
        else:
            denoised = savgol_filter(noisy, 12 * 2 + 1, 2)  #window of 65min
        return denoised

    dataset = dataset.sort_values(['id_zone', 'DayType', 'Time'])
    group_vars = ['DayType'] if use_daytype else []
    group_vars += ['id_zone'] if use_zonegroup else []

    if group_vars:
        dataset['smooth_traffic'] = dataset.groupby(group_vars)['traffic_index'].transform(lambda x: smoothing_function(x, method))
    else:
        dataset['smooth_traffic'] = smoothing_function(dataset['traffic_index'], method)
    return dataset



def main():
    # Decide whether to use Daytype classification or not
    use_daytype = True
    # Decide whether to use AOI zoning or not
    aoi_type = "zoi"  # TODO: set into the config file
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 1)
    time_unit = 5  # minutes
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("---Starting the spira traffic estimation---")
        get_traffic_from_spiras(use_daytype, aoi_type, start_date, end_date, time_unit)


if __name__ == '__main__':
    main()
    logging.info("---Spira traffic estimation completed---")
