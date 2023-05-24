import numpy as np
import pandas as pd
import scipy.stats
#import sklearn
import sklearn.metrics

import config


def compile_simulation_results(countries_and_wealth_measures,
                               spatial = False,
                               n=100,
                               clip_preds=False,
                               flip_yhat_rural=True,
                               add_binarized_yhat_rural = True,
                               verbose=True):
    """Reads in simulation results and processes in pd.DataFrames per country
    Inputs:
        countries_and_wealth_measures: list of tuples of (country_name, wealth_measure)
        spatial: bool, weather a spatial holdour was used
        clip_preds: bool, whether to clip preidctions to within the bounds of the train valuse
        flip_yhat_rural: bool, if yhat_rural is closer to P(region=urban), then take 1 - yhat_rural
        add_binarized_yhat: add, whether to binarize predicted rural (where threshold for binarization is chosen
            so that fraction of true and rural in train set are equal)
        n: int, number of trials per country
        verbose: bool, whether to print statements describing progress
    """
    
    if spatial:
        fn_end = '_spatial.csv'
    else:
        fn_end = '_no_spatial.csv'
    
    test_dfs_by_country = []
    train_dfs_by_country = []

    for country, poverty in countries_and_wealth_measures:
        if verbose: print(country)
        # get config -- slightly different processs for dhs and other countries
        if country.startswith('dhs/'):
            cfg = config.dataset_keys['dhs']
            fpath = cfg['OUTFILE_NAME'](country.split('/')[1])
        else:
            cfg = config.dataset_keys[country]
            fpath = cfg['OUTFILE_NAME']

        # read results of poverty predictions
        df_pov = pd.read_csv(fpath + poverty + fn_end)
        df_pov.rename(columns = {'y': 'y_pov', 'yhat': 'yhat_pov', 'y_noised': 'y_noised_pov'}, inplace=True)
        pov_cols = ['y_pov', 'yhat_pov', 'y_noised_pov', 'sim', 'split'] 
        df_pov = df_pov[pov_cols]
        
        # read results of rural predictions
        df_rur = pd.read_csv(fpath + 'rural' + fn_end)[['y','yhat','y_noised']]
        df_rur.rename(columns = {'y': 'rural', 'yhat': 'yhat_rural', 'y_noised': 'rural_noised'}, inplace=True)
        
        if flip_yhat_rural:
            # because we read the 0th entry of predict_proba()
            df_rur.loc[:,'yhat_rural'] = 1 - df_rur.loc[:,'yhat_rural'] 
            
        # merge predictions 
        df_all_sims = df_rur.merge(df_pov,left_index=True,right_index=True)
        
        
        # split train and test and keep track of each trial by n
        test_dfs, train_dfs = [], []
        for i in range(n):
            # get sim results for trial i
            df_sim_this_n = df_all_sims[df_all_sims.sim == i]
            train_df_this_n = df_sim_this_n[df_sim_this_n.split == 'train'].copy()
            test_df_this_n = df_sim_this_n[df_sim_this_n.split == 'test'].copy()

            # if flag set, clip predictions - train and test set to the min and max of train values
            if clip_preds: 
                train_ys = train_df_this_n.loc[:,'y_pov']
                clip_to_train_ys = lambda x: np.clip(x, tr.caluesain_ys.min(),train_ys.max())
                # apply to train and test
                train_df_this_n.loc[:,'yhat_pov'] = train_df_this_n.loc[:,'yhat_pov'].apply(clip_to_train_ys)
                test_df_this_n.loc[:,'yhat_pov'] = test_df_this_n.loc[:,'yhat_pov'].apply(clip_to_train_ys)
                   
        
            if add_binarized_yhat_rural:
                n_train_rural = np.array(train_df_this_n.loc[:,'rural'].values).sum() # .copy().values.sum()
                train_yhat_rurals = np.array(train_df_this_n.loc[:,'yhat_rural'].values)
                test_yhat_rurals = np.array(test_df_this_n.loc[:,'yhat_rural'].values)
                # assign thresh to match distribution in train set values
                pred_rural_thresh = np.sort(train_yhat_rurals)[len(train_yhat_rurals)-n_train_rural]
                train_df_this_n.loc[:,'yhat_rural_bin'] = np.array(train_yhat_rurals >= pred_rural_thresh).astype(int)
                test_df_this_n.loc[:,'yhat_rural_bin'] = np.array(test_yhat_rurals >= pred_rural_thresh).astype(int)
             
            train_dfs.append(train_df_this_n)
            test_dfs.append(test_df_this_n)

        train_dfs_by_country.append(train_dfs)
        test_dfs_by_country.append(test_dfs)
        
    return test_dfs_by_country, train_dfs_by_country
    
def compute_prediction_metrics(y, yhat, rural):
    rural = rural.astype(bool)
    metrics = {}
    metrics['bias_all'] = yhat.mean() - y.mean()
    metrics['bias_rural'] = y[rural].mean() - yhat[rural].mean()
    metrics['bias_urban'] = y[~rural].mean() - yhat[~rural].mean()

    metrics['r2_all'] = sklearn.metrics.r2_score(y, yhat)
    metrics['r2_rural'] = sklearn.metrics.r2_score(y[rural], yhat[rural])
    metrics['r2_urban'] = sklearn.metrics.r2_score(y[~rural], yhat[~rural])

    metrics['spearman_all'] = scipy.stats.spearmanr(y, yhat)[0]
    metrics['spearman_rural'] = scipy.stats.spearmanr(y[rural], yhat[rural])[0]
    metrics['spearman_urban'] = scipy.stats.spearmanr(y[~rural], yhat[~rural])[0]
    
    return metrics

def divide_nan_if_zero(x,y):
        if y == 0: return np.nan
        else: return x /y 
    
def compute_allocation_metrics(selected_by_y, selected_by_yhat, rural):
    metrics = {}
    # make sure everything is boolean
    rural = rural.astype(bool)
    selected_by_y = selected_by_y.astype(bool)
    selected_by_yhat = selected_by_yhat.astype(bool)

    # calcluate percent selected that is rural  
    rural_selected_by_y = np.logical_and(rural, selected_by_y)
    rural_selected_by_yhat = np.logical_and(rural, selected_by_yhat)
    urban_selected_by_y = np.logical_and(~rural, selected_by_y)
    urban_selected_by_yhat = np.logical_and(~rural, selected_by_yhat)
    
    rural_alloc_y = np.sum(rural_selected_by_y) / np.sum(selected_by_y)
    rural_alloc_yhat = np.sum(rural_selected_by_yhat) / np.sum(selected_by_yhat)
    metrics['rural_alloc_true'] = 100*rural_alloc_y
    metrics['rural_alloc_proxy'] = 100*rural_alloc_yhat 
    metrics['rural_alloc_diff'] = 100*(rural_alloc_yhat - rural_alloc_y)
    
    # calcluate percent of rural and urban under threshold that get targeted with yhat
    selected_by_both = np.logical_and(selected_by_y,selected_by_yhat)
    rural_selected_by_both = np.logical_and(selected_by_both, rural)
    urban_selected_by_both = np.logical_and(selected_by_both, ~rural)

    # recall metrics
  #  print(np.shape(selected_by_both), np.shape(selected_by_y))
  #  print(np.sum(selected_by_both), np.sum(selected_by_y))
    metrics['recall_all'] = 100 * np.sum(selected_by_both) / np.sum(selected_by_y)
    metrics['recall_rural'] = 100 * divide_nan_if_zero(np.sum(rural_selected_by_both),
                                                       np.sum(rural_selected_by_y))
    metrics['recall_urban'] = 100 * divide_nan_if_zero(np.sum(urban_selected_by_both),
                                                       np.sum(urban_selected_by_y))
    # precision metrics 
    metrics['precision_all'] = 100 * np.sum(selected_by_both) / np.sum(selected_by_yhat)
    metrics['precision_rural'] = 100 * divide_nan_if_zero(np.sum(rural_selected_by_both),
                                                          np.sum(rural_selected_by_yhat))
    metrics['precision_urban'] = 100 * divide_nan_if_zero(np.sum(urban_selected_by_both),
                                                          np.sum(urban_selected_by_yhat))
     
    return metrics
  