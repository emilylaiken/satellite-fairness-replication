
sim_dir = '/data/mosaiks/simulations_normalized/'
FEATURES_FOLDER = '/data/mosaiks/replication/features/'
FEATURES_FOLDER_1_TILE = '/data/mosaiks/replication/features_1_tile_per_region/'

dataset_keys = {
    'us' : {'FEATURES_FNAME' : FEATURES_FOLDER+'/mosaiks_features_by_puma_us.csv',
            'LABELS_FNAME' : '/data/mosaiks/surveys/us/groundtruth_by_puma_2019.csv',
            'INDIVIDUAL_FNAME': '/data/mosaiks/surveys/us/individual_level_data.csv',
            'MERGE_KEYS' : ['PUMA', 'State'],
            'SPLIT_KEYS' : {'spatial':'State', 'no_spatial':'PUMA'},
            'POVERTY' : 'FINCP', # Use 'FINCP' for income experiment or 'pop_density' for pop density experiment
            'WEIGHT' : 'PWGTP',
            'OUTFILE_NAME' : sim_dir+'us/',
            'FAIRNESS_VARS': ['rural'],
    },

    
    'mexico': {
        'FEATURES_FNAME' : FEATURES_FOLDER+'/mosaiks_features_by_municipality_mexico.csv',
        'LABELS_FNAME' : '/data/mosaiks/surveys/mexico/grouped.csv',
        'INDIVIDUAL_FNAME' : '/data/mosaiks/surveys/mexico/',
        'MERGE_KEYS' : ['municipality'],
        'SPLIT_KEYS' : {'spatial':'state', 'no_spatial':'municipality'},
        'POVERTY' : 'asset_index', # Use 'asset_index' for wealth, no population density data yet
        'WEIGHT' : 'weight',
        'OUTFILE_NAME' : sim_dir+'mexico/',
        'FAIRNESS_VARS': ['rural'],
    },
    
    'dhs': {
        'FEATURES_FNAME' : lambda country: FEATURES_FOLDER + 'dhs/mosaiks_features_by_cluster_' + country + '.csv',
        'LABELS_FNAME' : lambda country: '/data/mosaiks/surveys/dhs/' + country + '_grouped.csv',
        'INDIVIDUAL_FNAME' : lambda country: '/data/mosaiks/surveys/dhs/' + country + '_hh.csv',
        'MERGE_KEYS' : ['cluster'],
        'SPLIT_KEYS' : {'spatial':'region', 'no_spatial':'cluster'},
        'POVERTY' : 'wealth', # Use 'wealth' for wealth, no population density data yet
        'WEIGHT' : 'weight',
        'OUTFILE_NAME' : lambda country: sim_dir+'dhs/' + country + '/',
        'FAIRNESS_VARS': ['rural'],
    },
    
    'india': {
        'FEATURES_FNAME' : FEATURES_FOLDER +'mosaiks_features_by_shrug_condensed_regions_25_max_tiles_100_india.csv',
        'LABELS_FNAME' : '/data/mosaiks/shrug/shrug_condensed_regions_25.csv',
        'META_FNAME': '/data/mosaiks/sampled_tiles/india/shrug_condensed_regions_25_max_tiles_100_meta.csv',
        'INDIVIDUAL_FNAME' : None,
        'MERGE_KEYS' : ['condensed_shrug_id'],
        'SPLIT_KEYS' : {'no_spatial':'condensed_shrug_id'},
        'POVERTY' : 'secc_cons_pc_combined',
        'WEIGHT' : 'dummy_ones',
        'OUTFILE_NAME' : sim_dir+'india_condensed_regions_25_max_tiles_100/',
        'FAIRNESS_VARS': ['rural'],
    },
    
    'india-14': {
        'FEATURES_FNAME' : FEATURES_FOLDER + 'mosaiks_features_by_shrug_condensed_regions_14_india.csv',
        'LABELS_FNAME' : '/data/mosaiks/shrug/shrug_condensed_regions_14.csv',
        'INDIVIDUAL_FNAME' : None,
        'MERGE_KEYS' : ['condensed_shrug_id'],
        'SPLIT_KEYS' : {'no_spatial':'condensed_shrug_id'},
        'POVERTY' : 'secc_cons_pc_combined',
        'WEIGHT' : 'dummy_ones',
        'OUTFILE_NAME' : sim_dir+'india_condensed_regions_14/',
        'FAIRNESS_VARS': ['rural'],
    },

}