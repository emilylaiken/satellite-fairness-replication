import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from read_experiment_results import (
    compile_simulation_results, 
    compute_allocation_metrics,
    compute_prediction_metrics
)

def transform_and_compute_metrics(test_dfs, 
                                  train_dfs,
                                  prediction_transform_conditions,
                                  threshold_transform_conditions,
                                  percentile_targeting=20,
                                  n_trials=100):
    """Takes simulated about, transforms acoording to calibration conditions, and computes metrics.
    
    Predictions are transformed according to the settings in prediction_cal_conditions.
    Targeting/allocation thresholds may be set differently per group according to threshold_cal_conditions.
    """

    metrics_each_n = []
    output_each_n = []
    
    for i in range(n_trials):
        metrics_this_n = {}
        
        test_res = test_dfs[i]
        train_res = train_dfs[i]
        
        # read y and true urban/rural from test results
        y = test_res.y_pov
        rural = test_res.rural.astype(bool)
        
        # perform any transforms to the test set preidctions
        yhat = transform_predictions(test_res, 
                                     train_res, 
                                     **prediction_transform_conditions)
       
        test_res.loc[:,'yhat_transformed'] = yhat
        
        
        # compute prediction metrics
        pred_metrics = compute_prediction_metrics(y,yhat,rural)
        metrics_this_n.update(pred_metrics)
        
        # baseline to compare to picks lowest percentile_targeting y candidates
        selected_by_y = compute_allocations(test_res,
                                            train_res,
                                            'y_pov',
                                            percentile_targeting,
                                            thresh_strategy = 'single_thresh') 
        
        selected_by_yhat = compute_allocations(test_res,
                                               train_res,
                                               'yhat_transformed',
                                               percentile_targeting,
                                               **threshold_transform_conditions) 

        alloc_metrics = compute_allocation_metrics(selected_by_y, selected_by_yhat, rural)
        metrics_this_n.update(alloc_metrics)
        
        metrics_each_n.append(metrics_this_n)
        
        output_each_n.append({'y':y,
                              'rural':rural,
                              'yhat_pov': test_res.yhat_pov,
                              'yhat_rural': test_res.yhat_rural,
                              'yhat_rural_bin': test_res.yhat_rural_bin,
                              'yhat_pov_trasformed':yhat})
        
    # reorder prediction and allocation metrics sorted by n
    metrics_all_trials = {}
    for key in metrics_each_n[0].keys():
        metrics_all_trials[key] = np.array([metrics_this[key] for metrics_this in metrics_each_n])
        
    output_all_trials = {}
    for key in output_each_n[0].keys():
        output_all_trials[key] = [output_this[key] for output_this in output_each_n]
        
    return metrics_all_trials, output_all_trials


def transform_predictions(test_res, 
                          train_res, 
                          transform_strategy, 
                          g_column='rural',
                          g_column_compute_transform=None,
                          transform_with='train', 
                          verbose=False):
    
    if transform_strategy == 'none':
        # don't transform
        return test_res.yhat_pov
    elif transform_strategy == 'additive_constant':
        transform_fxn = transform_predictions_constant
    elif transform_strategy == 'affine':
        transform_fxn = transform_predictions_affine
    else: 
        print(f'{transform_strategy} unknown', end='')
        print(f'currently only "none", "additive_constant", and "affine" transforms supported')
        
        
    # fit transform via train or test transform 
    if transform_with == 'train':
        res_to_cal_with = train_res.copy()
    elif transform_with == 'test':
        res_to_cal_with = test_res.copy()
        
    if verbose: 
        print(f'transforming predictions via column {g_column} ',end='')
        print(f'in {transform_with}, according to {transform_strategy}')
        
 
    yhat_trans =  transform_fxn(test_res.copy(),
                                res_to_cal_with.copy(),
                                g_column=g_column,
                                g_column_compute_transform=g_column_compute_transform)

    return yhat_trans.ravel()


def transform_predictions_constant(res_to_transform, 
                                   res_to_compute_transform_params,
                                   g_column,
                                   g_column_compute_transform = None):
    
    if g_column_compute_transform is None:
        g_column_compute_transform = g_column
    # yhat_trans =  yhat + a * g + b
    yhat_train = res_to_compute_transform_params.yhat_pov.values.reshape(-1,1)
    y_train = res_to_compute_transform_params.y_pov.values.reshape(-1,1)
    g_train = res_to_compute_transform_params[g_column_compute_transform].values.reshape(-1,1)
 
    # m = Pipeline(steps=[('reg', LinearRegression(fit_intercept=True))])
    m = LinearRegression(fit_intercept=True)
    
    # train set: fit delta = yhat_trans - yhat = a * g + b
    m.fit(g_train, y_train-yhat_train)
    
    # apply trasnform
    g_to_transform = res_to_transform[g_column].values.reshape(-1,1)
    yhat_to_transform = res_to_transform.yhat_pov.values.reshape(-1,1)
    
    # test set: predict delta as a * g + b
    delta_transform = m.predict(g_to_transform)
    # test set: add yhat_trans = yhat + delta
    
    return yhat_to_transform + delta_transform

def transform_predictions_affine(res_to_transform, 
                                 res_to_compute_transform_params,
                                 g_column,
                                 g_column_compute_transform = None):
    
    if g_column_compute_transform is None:
        g_column_compute_transform = g_column
        
    # yhat_transform = a * yhat * g + b * yhat + c * g + d
    yhat_train = res_to_compute_transform_params.yhat_pov.values.reshape(-1,1)
    y_train = res_to_compute_transform_params.y_pov.values.reshape(-1,1)
    g_train = res_to_compute_transform_params[g_column_compute_transform].values.reshape(-1,1)
    X_train = np.hstack([yhat_train, g_train, yhat_train*g_train])
    
    # m = Pipeline(steps=[('reg', LinearRegression(fit_intercept=True))])
    m = LinearRegression(fit_intercept=True)
    m.fit(X_train, y_train)
    
    yhat_to_transform = res_to_transform.yhat_pov.values.copy()
    g_to_transform = res_to_transform[g_column].values 
    X_to_transform = np.vstack([yhat_to_transform, g_to_transform, yhat_to_transform*g_to_transform]).T
    
    yhat_transformed = m.predict(X_to_transform).ravel()
    
    return yhat_transformed


def compute_allocations(test_res,
                        train_res, 
                        y_key,
                        pop_targeting_percentile,
                        thresh_strategy, 
                        verbose=False,
                        rural_measure_train=None,
                        rural_measure='rural'):
    
    yhat_test = test_res[y_key]
    
    if thresh_strategy == 'single_thresh':
        
        targeting_thresh = np.percentile(yhat_test,pop_targeting_percentile)
        targeting_thresh_per_instance = np.ones(len(yhat_test)) * targeting_thresh
        
    elif thresh_strategy == 'match_rural_rates_in_train':
        t = group_thresh_to_match_train_distributions(test_res,
                                                      train_res, 
                                                      y_key,
                                                      pop_targeting_percentile,
                                                      rural_measure_train=rural_measure_train,
                                                      rural_measure=rural_measure,
                                                      verbose=verbose)
        targeting_thresh_per_instance = t
    else:
        print(f'thresh strategy {thresh_strategy} not understood', end='')
        print(f'options are "single_thresh", "match_rural_rates_in"')
    
    return yhat_test < targeting_thresh_per_instance



def group_thresh_to_match_train_distributions(test_res,
                                              train_res, 
                                              y_key,
                                              pop_targeting_percentile,
                                              rural_measure,
                                              rural_measure_train=None,
                                              verbose=True): 
    if rural_measure_train is None:
        rural_measure_train = rural_measure
        
    # find the relative frac of urban/rural below targeting percentile in train ys
    y_train = train_res.y_pov
    rural_train = train_res[rural_measure_train].astype(bool)
    
    # note: if no transform is used, test_res['yhat_transformed'] will just be test_res['y']
    yhat_test = test_res[y_key]
    rural_test = test_res[rural_measure].astype(bool)
        
    # compute rates of selection according to training set distribution
    y_train_thresh = np.percentile(y_train, pop_targeting_percentile)
    frac_train_selected_rural = np.mean(rural_train[y_train < y_train_thresh])
    frac_train_selected_urban = 1 - frac_train_selected_rural
    
    if verbose:
        print('frac_train_selected_rural ', frac_train_selected_rural)
        print('frac_train_selected_urban ', frac_train_selected_urban)
    
    # now select test predictions according to this -- keep the ratio between selected
    # from urban and rural the same
    # number of urban and rural to seelct from the test set 
    num_test_to_select = int(len(yhat_test)*pop_targeting_percentile/100)
    num_rural_to_select  = int(np.round(frac_train_selected_rural * num_test_to_select, 0))
    num_urban_to_select = int(np.round(frac_train_selected_urban * num_test_to_select, 0))
    if verbose:
        print('num_rural_to_select ', num_rural_to_select)
        print('num_urban_to_select ', num_urban_to_select)
    # calculate separate thresholds for urban and rural according to test set distributions
    yhat_test_rural = np.array(yhat_test[rural_test].values)
    yhat_test_urban = np.array(yhat_test[~rural_test].values)
    
    # select from the lowest predictions in each
    thresh_rural = np.sort(yhat_test_rural)[num_rural_to_select]
    thresh_urban = np.sort(yhat_test_urban)[num_urban_to_select]
    if verbose:
        print(thresh_rural, thresh_urban)
    
    # apply thresholds to set set
    threshes_per_unit_test = thresh_rural * rural_test + thresh_urban * (1-rural_test)
    return threshes_per_unit_test

