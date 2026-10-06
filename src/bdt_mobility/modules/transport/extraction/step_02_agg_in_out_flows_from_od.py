import pandas as pd
from bdt_mobility.utils.paths import PathRegistry
from bdt_mobility.utils import data_reader
from bdt_mobility.modules.transport.extraction import spatial_assignment
from bdt_mobility.utils.config import logging, logging_level
from bdt_mobility.utils.timing import timed
from bdt_mobility.extraction.platform.outputs import PlatformOutput

MAIN_OUTPUTS = [
    PlatformOutput("total_flows_from_od", PathRegistry.TOTAL_FLOWS_FROM_OD, "table"),
    PlatformOutput("total_starts_from_od", PathRegistry.TOTAL_STARTS_FROM_OD, "table"),
]


def _prepare_gdf_areas(
    gdf_zoi,
):
    # Read the areas with a polygonal shape
    aree_gdf_inside, aree_gdf_outside = data_reader.AOI_shapes(
        PathRegistry.OD_SHAPE_ZONE,
        aoi_type="od", 
        df_around=gdf_zoi, 
        return_type="both"
    )
    aree_gdf_inside['id_zone'] = aree_gdf_inside['id_zone'].astype(int)
    aree_gdf_inside["type_zoi"] = "inside"
    aree_gdf_outside['id_zone'] = aree_gdf_outside['id_zone'].astype(int)
    aree_gdf_outside["type_zoi"] = "outside"
    aree_gdf = pd.concat([aree_gdf_inside, aree_gdf_outside], axis=0)
    aree_gdf = aree_gdf.sort_values(by=['id_zone', 'type_zoi'])

    # Group by id and compute the percentage of area with respect to the area of the total zone
    aree_gdf['area'] = aree_gdf.geometry.area
    aree_gdf2 = (
        aree_gdf
        .groupby('id_zone')
        .agg({'area': 'sum'})
        .rename(columns={'area': 'total_area'})
        .reset_index()
    )
    aree_gdf = aree_gdf.merge(aree_gdf2, on='id_zone', how='left')
    aree_gdf["perc_area"] = round(aree_gdf["area"] / aree_gdf["total_area"], 3)
    aree_gdf = aree_gdf.drop(columns=['area', 'total_area', 'geometry'])

    # Add the few external zones with no polygonal shape
    centers_gdf = data_reader.OD_centers(PathRegistry.OD_SHAPE_CENTROID)
    centers_gdf['id_zone'] = centers_gdf['id'].astype(int)
    centers_gdf = centers_gdf[~centers_gdf['id_zone'].isin(aree_gdf['id_zone'])]
    centers_gdf['type_zoi'] = "outside"
    centers_gdf['perc_area'] = 1.0
    aree_gdf = pd.concat([aree_gdf[['id_zone', 'type_zoi', 'perc_area']],
                          centers_gdf[['id_zone', 'type_zoi', 'perc_area']]], axis=0)

    return aree_gdf    


def get_flows_from_od(
    od,
    aree_gdf,
):
    # Standardize the type_zoi column to boolean values  
    aree_gdf   = aree_gdf.copy()
    aree_gdf["type_zoi"] = aree_gdf["type_zoi"] == "inside"

    # Aggregate flows based on ZOI intersections
    od_merge = od.merge(
            aree_gdf.rename(columns={'id_zone': 'from', 'type_zoi': 'from_zoi', 'perc_area': 'from_perc_area'}),
            on='from', 
            how='left'
        ).merge(
            aree_gdf.rename(columns={'id_zone': 'to', 'type_zoi': 'to_zoi', 'perc_area': 'to_perc_area'}), 
            on='to', 
            how='left'
        )
    od_merge['flow'] = od_merge['flow'] * od_merge['from_perc_area'] * od_merge['to_perc_area']

    df_tot_flows = od_merge.groupby(['from_zoi', 'to_zoi'])['flow'].sum().reset_index()
    logging.debug(f"- intra and inter flows in ZOI:\n{df_tot_flows}")

    return df_tot_flows


def get_start_flows_from_od(
    od, 
    aree_gdf
):
    # Aggregate flows based on ZOI intersections
    od_merge = od.merge(
            aree_gdf[['id_zone', 'type_zoi']].rename(columns={'id_zone': 'from', 'type_zoi': 'type_zoi_from'}),
            on='from', 
            how='left')
    
    df_start_flows = od_merge.groupby(['from','type_zoi_from'])['flow'].sum().reset_index()
    logging.debug(f"- intra and inter flows in ZOI:\n{df_start_flows}")

    return df_start_flows


def main():
    logging.basicConfig(level=logging_level)
    with timed():
        logging.info("Reading the data")
        # Load ZOI shape
        gdf_zoi = data_reader.ZOI_shape(PathRegistry.ZOI_BORDER)

        # Load OD shapes and clean overlapping zones
        aree_gdf = _prepare_gdf_areas(gdf_zoi)

        # Load OD flows -> only those with a shape, either polygonal or point
        # (unit: people)
        od = data_reader.OD_flows(PathRegistry.OD_FLOWS_EXCEL)
        tot_flows = od['flow'].sum()
        logging.debug(f"- total people: {tot_flows}")
        
        logging.info("Computing the total flows from OD data")
        df_tot_flows = get_flows_from_od(od=od, aree_gdf=aree_gdf)
        df_tot_flows.to_parquet(PathRegistry.TOTAL_FLOWS_FROM_OD)
        tot_flows = df_tot_flows['flow'].sum()
        logging.debug(f"- total people flows_from_od: {tot_flows}")

        logging.info("Computing the total starting flows from OD data")
        df_start_flows = get_start_flows_from_od(od=od, aree_gdf=aree_gdf)
        df_start_flows.to_parquet(PathRegistry.TOTAL_STARTS_FROM_OD)
        tot_flows = df_start_flows['flow'].sum()
        logging.debug(f"- total people start_flows_from_od: {tot_flows}")


if __name__ == "__main__":
    main()