import geopandas as gpd
import numpy as np
import os
import pandas as pd
from shapely import wkt

def compute_combined_consumption(secc_pc_df):
    '''Combine consumption between urban and rural based on the census population counts.
    
    Takes an average of the per-capita population consumption for urban and rural from SECC, weighted by 
    urban and rural population counts in the census data.
    
    Inputs: dataframe with SECC and census pc data already merged by shrid
    
    Returns: pd.series of combined per capita consumption 
    '''
    tot_cons_rural = secc_pc_df['secc_cons_pc_rural'].fillna(0) * secc_pc_df['shrid_pc11_pca_tot_p_r'].fillna(0)
    tot_cons_urban = secc_pc_df['secc_cons_pc_urban'].fillna(0) * secc_pc_df['shrid_pc11_pca_tot_p_u'].fillna(0)
    
    # only count population in the denominator if consumption for that type is not nan
    denom_rural = secc_pc_df['shrid_pc11_pca_tot_p_r'].fillna(0)*secc_pc_df['secc_cons_pc_rural'].notna().astype(int)
    denom_urban = secc_pc_df['shrid_pc11_pca_tot_p_u'].fillna(0)*secc_pc_df['secc_cons_pc_urban'].notna().astype(int)

    return (tot_cons_rural + tot_cons_urban) / (denom_rural + denom_urban)

def merge_raw_shrug_files(data_fp, 
                  shapefile_fp,
                  data_dir = '/data/mosaiks/shrug/',
                  out_fn = 'shrug.csv',
                  verbose=True):
    
    # gather label data
    data_full = pd.read_stata(data_fp)
    all_df = data_full.copy(deep=True)
    
    # read in shapefiles
    shapefile_fp = os.path.join(data_dir,'shrid2.gpkg')
    shapefiles_full = gpd.read_file(shapefile_fp)
    
    # remove any with thiessen shapefiles
    shps = shapefiles_full[shapefiles_full.polytype != 'thiessen'].copy()
    data_with_shps = data_full.merge(shps[['shrid','pc11_id','geometry']],on='shrid', how='inner')
    if verbose: len(data_with_shps)
    
    # 1. Drop any entries without consumption values
    all_df = data_with_shps.dropna(subset=['secc_cons_pc_rural','secc_cons_pc_urban'],how='all').copy()
    if verbose: print(f'{len(data_with_shps) - len(all_df)} entries dropped for not having consumption value')
    if verbose: print(f'{len(all_df)} entries remaining')
    
    # 2. compute per capita consumption at shrid unit as weighted average of rural and urban consumption
    
    # a few will have both urban and rural
    all_df.loc[:,'rural'] = all_df['secc_cons_pc_rural'].notna()
    all_df.loc[:,'urban'] = all_df['secc_cons_pc_urban'].notna()

    # combine consumption between urban and rural based on the census population counts
    all_df['secc_cons_pc_combined'] = compute_combined_consumption(all_df)
    
    # add urban and rural -- it's only going to be different from the existing data for like two rows, and barely
    all_df['pc11_pca_tot_p_combined'] = all_df['shrid_pc11_pca_tot_p_u'].fillna(0) + all_df['shrid_pc11_pca_tot_p_r'].fillna(0)
    all_df['frac_rural'] =  all_df['shrid_pc11_pca_tot_p_r'].fillna(0) / all_df['pc11_pca_tot_p_combined']
    all_df['frac_urban'] =  all_df['shrid_pc11_pca_tot_p_u'].fillna(0) / all_df['pc11_pca_tot_p_combined']
    
    # make sure consumption only different for rows that had consumption values for both urban and rural
    num_rows_with_urban_and_rural_secc = ((all_df['secc_cons_pc_rural'].notna().astype(int) + all_df['secc_cons_pc_urban'].notna().astype(int)) > 1).sum()
    num_rows_altered_by_combo = len(all_df) - np.isclose(all_df['secc_cons_pc_combined'],
                                                         all_df['secc_cons_pc_rural'].fillna(0) + all_df['secc_cons_pc_urban'].fillna(0)).sum()

    assert num_rows_with_urban_and_rural_secc == num_rows_altered_by_combo
    if verbose: print(f'for the {num_rows_with_urban_and_rural_secc} shrids with urban and rural, consumption ',
          'is a weighted average of urban and rural consumption, weighted by urban and rural population counts.')
    
    # 3. Save the processed data
    
    # prepate some extra columns
    all_df['state'] = all_df.apply(lambda x: x.shrid.split('-')[1], axis=1)
    all_df['dummy_ones'] = np.ones(len(all_df))
    
    out_fp = os.path.join(data_dir, out_fn)
    if verbose: print(f'saving compiled data in {out_fp}')
    all_df.to_csv(out_fp,index=False)
    

    out_fp_geojson = os.path.join(data_dir, out_fn.replace('csv','geojson'))
    all_gdf = gpd.GeoDataFrame(all_df)
    if verbose: print(f'saving geo compiled data in {out_fp_geojson}')
    all_gdf.to_file(out_fp_geojson,driver='GeoJSON')

    
    all_df.to_csv(out_fp,index=False)
    
    # these will get saved in a future step, so no need to also save them now
#     all_df[['shrid','dmsp_mean_light_2012']].to_csv(os.path.join(data_dir,'shrug_nl_dmsp_2012.csv'))
#     all_df[['shrid','viirs_mean_light_2020']].to_csv(os.path.join(data_dir,'shrug_nl_viirs_2020.csv'))
                      
    return 

def prepare_shrug_data_for_geom(shrug_csv_fp):
    
    df_by_shrid = pd.read_csv(shrug_csv_fp)
    
    # this line is needed to set crs
    df_by_shrid.loc[:,'geometry'] = df_by_shrid.loc[:,'geometry'].apply(wkt.loads)
    gdf_by_shrid = gpd.GeoDataFrame(df_by_shrid)
    # put in latlon
    gdf_by_shrid = gdf_by_shrid.set_crs('EPSG:4326')
    
    # add in projected area
    gdf_by_shrid.loc[:,'state'] = gdf_by_shrid.loc[:,'shrid'].apply(lambda x: x.split('-')[1])
    gdf_by_shrid_proj = gdf_by_shrid.to_crs('EPSG:3857') 
    gdf_by_shrid.loc[:,'proj_area'] = gdf_by_shrid_proj.geometry.area
    
    return gdf_by_shrid

def split_urban_rural(gdf_by_shrid_in):
    keys_to_keep = ['shrid', 
                     'viirs_mean_light_2020', 
                     'dmsp_mean_light_2012',
                     'geometry', 
                     'rural', 
                     'urban', 
                     'secc_cons_pc_combined',
                     'pc11_pca_tot_p_combined', 
                     'frac_rural', 
                     'frac_urban', 
                     'state',
                     'dummy_ones', 
                     'proj_area'
                   ]

    gdf_by_shrid = gdf_by_shrid_in.copy()[keys_to_keep]

    # urban is any shrid with urban (possible also with rural)
    gdf_urban = gdf_by_shrid[gdf_by_shrid.urban]
    # rural is any shrid with only rural
    gdf_rural = gdf_by_shrid[~gdf_by_shrid.urban]
    
    return {'gdf_urban': gdf_urban, 'gdf_rural': gdf_rural}
