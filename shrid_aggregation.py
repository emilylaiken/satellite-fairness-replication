import geopandas as gpd
import numpy as np
import os
import pandas as pd
import time

from multiprocessing import Pool
from spopt.region import MaxPHeuristic as MaxP
import libpysal
from spopt.region.maxp import infeasible_components

def loop_over_states(gdf_rural,
                     sq_km_thresh,
                     chunked_outfile_dir,
                     combine_strategy='ours',
                     num_threads=20,
                     verbose=False):

    """Run the aggregation process within district.
    
    Iterates over states and parallizing across districts in a state.
    """
    states = gdf_rural.state.unique()
    
    for state in states:
        print(state)
        t1 = time.time()
        
        outfile = os.path.join(chunked_outfile_dir, f'shrug_rural_state_{state}.geojson')
        
        # only consider this state
        df_this_state = gdf_rural[gdf_rural.state == state].reset_index(drop=True).copy()
        
        if len(df_this_state) <=1: 
            df_this_state['shrids_merged_str'] = df_this_state['shrid']
        
            print(f'df for state {state} of length {len(df_this_state)}; writing to {outfile}')
            df_this_state.to_file(outfile, driver='GeoJSON')
            continue
            
        # convert to projected CRS
        df_this_state = df_this_state.to_crs('EPSG:3857') 

        # add colum to track merged shrids
        df_this_state['shrids_merged'] = [[x] for x in df_this_state.shrid.values]
        df_this_state['checked'] = False
        df_this_state['region'] = [x.split('-')[2] for x in df_this_state.shrid.values]
        
        small_dfs_per_region_combined = combine_small_regions_within_districts(df_this_state,
                                                                               sq_km_thresh,
                                                                               combine_strategy,
                                                                               num_threads=num_threads)

        
        print(f'writing to {outfile}')
        small_dfs_per_region_combined.to_file(outfile, driver='GeoJSON')
        
        t2 = time.time()
        print(f'state {state} took {(t2-t1)/60:.2f} minutes')
        
    return

def combine_small_regions_within_districts(gdf_this,
                                           sq_km_thresh,
                                           combine_strategy='ours',
                                           num_threads=20):

    """Merges small regions within districts."""
    pool = Pool(num_threads)
    
    df_this_state = gdf_this.copy()
    
    dfs_by_region = []
    for region in np.unique(df_this_state.region):
        dfs_by_region.append(df_this_state.loc[df_this_state.region == region])

    if combine_strategy == 'ours':    
        # set max_merges_per_region as sq_km_thresh
        max_merges_per_region = sq_km_thresh
        todo_list = [(x, sq_km_thresh, max_merges_per_region) for x in dfs_by_region]
        results = pool.map(combine_small_regions, todo_list)
    else: print(f'combine strategy {combine_strategy} unknown')

    small_dfs_per_region_combined = pd.concat(results)
    
    # replace list with long string
    small_dfs_per_region_combined['shrids_merged_str'] = small_dfs_per_region_combined['shrids_merged'].apply(lambda x: ','.join(x))
    small_dfs_per_region_combined['shrids_merged_str'].tail()
    small_dfs_per_region_combined.drop(['shrids_merged'],axis=1,inplace=True)
    
    return small_dfs_per_region_combined


def combine_small_regions(args,
                          verbose=False
                          ):
    """ Combines small regions according to args. 
    Iteratively finds smallest region, and if less than args[0] in sq km, merges with region of heighest boundary
    overlap.
    
    args contains:
        df_this_state: geopandas.GeoDataFrame() containing the extent in which merging is OK (e.g. state/region)
        sq_km_thresh: the threshold at which a 
        max_merges_per_region: option, max number of merges for any region, set to ensure we don't get one giant region
            defautls to none, so no cap on number of merges
        
    kwargs:
        verbose: whether to print along the way
    
    """
    if len(args) == 2: 
        df_this_state, sq_km_thresh = args
        max_merges_per_region = None
    else:
        df_this_state, sq_km_thresh, max_merges_per_region = args

    # input named as if you're doing it by state but you can really do it by any region
    df_this_state_small = df_this_state.copy()
    candidates = df_this_state_small[df_this_state_small.proj_area < 1e6 * sq_km_thresh].copy()

    i = 0
    while(len(candidates) > 0):
        if verbose: print(f'{i}: num_candidates: {len(candidates)}')
        
        # get row with smallest projected area from among the canidates
        row_this = candidates.iloc[candidates['proj_area'].argmin():candidates['proj_area'].argmin()+1]

        # find all regions that are mergeable, i.e. have not been merged more than max_merges_per_region times.
        if max_merges_per_region is not None:
          #  merges_this_row_already = len(row_this.shrids_merged)
           # max_merges_for_neighbor = max_merges_per_region - merges_this_row_already
            less_than_k_merged = [len(x)  < max_merges_per_region for x in df_this_state_small.shrids_merged]
            mergeable = df_this_state_small[less_than_k_merged]
        else:
            mergeable = df_this_state_small
            
        # find all neighbors of this row
        neighbors = find_all_neighbors_in_df(row_this, mergeable)

        if verbose: print('num_neighbors: ', len(neighbors))

        if len(neighbors) > 0:
            # choose neighbor with heighest boundary overlap
            chosen_neighbor = select_neighbor_by_highest_boundary_overlap(row_this.geometry, neighbors)
            rows_to_merge = pd.concat([row_this,chosen_neighbor])

            # ammend dataframe to reflect the merge
            df_this_state_small = ammend_dataframe_merge_shrids(df_this_state_small, rows_to_merge)
            df_this_state_small.reset_index(drop=True)

        # mark as checked if no neighbors
        else:
            # find the index in the df
            idx = df_this_state_small.index[df_this_state_small.shrid.values == row_this.shrid.values][0]
            # mark it as checked
            df_this_state_small.loc[idx,'checked'] = True

        if verbose: print(f'df length: {len(df_this_state_small)}')

        # update candidates
        small_enough_areas = df_this_state_small[df_this_state_small.proj_area < 1e6 * sq_km_thresh]
        if len(candidates) > 0:
            # candidates are areas that are small enough and not already checked off
            candidates = small_enough_areas[~small_enough_areas.checked] 
            
        i += 1

    
    return df_this_state_small


def merge_rows(rows):
    """Determines how to merge each column when combining rows of the ddfs."""
    
    areas = rows.proj_area.values
    def mean_by_area(x_in):
        # weighted average of values by area (projected)
        x = x_in.copy().fillna(0)
        return np.sum(np.array(areas) * np.array(x)) / np.sum(areas)
    
    total_pop = rows.pc11_pca_tot_p_combined.values
    def mean_by_total_pop(x_in):
        # weighted average of values by popualtion (total pop = urban + rural pop)
        x = x_in.copy().fillna(0)
        return np.sum(np.array(total_pop) * np.array(x)) / np.sum(total_pop)
    
    # how to dissolce each column
    dissolve_dict = {'shrid': min, # shrid will still be unique
                     'viirs_mean_light_2020': mean_by_area, 
                     'dmsp_mean_light_2012': mean_by_area, 
                     'rural': lambda x: x.any(), 
                     'urban': lambda x: x.any(),  
                     'secc_cons_pc_combined': mean_by_total_pop,
                     'pc11_pca_tot_p_combined':sum, 
                     'frac_rural':mean_by_total_pop, 
                     'frac_urban': mean_by_total_pop,
                     'state': np.min,
                     'dummy_ones': np.min,
                     'proj_area': sum,
                     'shrids_merged': np.hstack,
                     'checked': lambda x: False # revert to unchecked for a merge
                     }
    new_row = rows.dissolve(aggfunc = dissolve_dict)
    return new_row

def ammend_dataframe_merge_shrids(df_start, rows_to_merge):
    """Merges rows_to_merge and combines with any rows of df_start not in rows_to_merge."""
    df_out = df_start.copy()
    shrids_to_merge = rows_to_merge.shrid.values
    
    # remove the shrids that will be merged
    for shrid in shrids_to_merge:
        df_out = df_out[df_out.shrid != shrid]
        
    # merge the shrids
    new_row = merge_rows(rows_to_merge)
    
    # now append:
    return pd.concat([df_out,new_row], ignore_index=True)


def find_all_neighbors_in_df(row_this, df_compare):
    """ Finds all geometries not disjoint from row_this in df_compare. A neighbor is anything not 
    disjoint from row_this.geometry
    """
    if len(df_compare) == 0: return []
    geom_this = row_this.geometry
    # get everything not disjoint from this geometry
    is_disjoint = np.hstack([geom_this.values.disjoint(x) for x in df_compare.geometry.values])
    neighbors = df_compare[~is_disjoint]
    # remove this row from its neighbors list
    neighbors = neighbors[neighbors.shrid != row_this.shrid.values[0]]
    return neighbors

def select_neighbor_by_highest_boundary_overlap(geom_this, neighbors):
    neighbor_geoms = neighbors.geometry.values
    bd_overlaps = np.hstack([geom_this.values.intersection(x).length for x in neighbor_geoms])
    
    # return as a df
    return neighbors.iloc[bd_overlaps.argmax():bd_overlaps.argmax()+1]
