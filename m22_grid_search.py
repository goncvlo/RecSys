import m11_load_data, m21_model_selection, m23_evaluation
from itertools import product
import pandas as pd
from surprise.model_selection import KFold, RepeatedKFold, GridSearchCV
from surprise import BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, NMF, CoClustering

# read config file
config = m11_load_data.read_yaml()
# define algorithm objects
algorithms = { 'BaselineOnly' : BaselineOnly
              , 'KNNBasic' : KNNBasic, 'KNNWithMeans' : KNNWithMeans, 'KNNWithZScore' : KNNWithZScore, 'KNNBaseline' : KNNBaseline
              , 'MF_SVD' : SVD, 'NMF' : NMF
              , 'CoClustering' : CoClustering }

def grid_search( df_ratings : pd.DataFrame, algorithm_name : str, cv_iterator : str
                , measures : list = ['mae', 'rmse', 'mse', 'fcp'], return_train_measures : bool = True, refit : bool = False):

    data = m21_model_selection.cross_validation(df_ratings = df_ratings, cv_iterator = cv_iterator)
    
    if cv_iterator not in ['hold_out', 'ts_split']:
        run_summary = grid_search_surprise( data = data, algorithm_name = algorithm_name, cv_iterator = cv_iterator
                                           , measures = measures, return_train_measures = return_train_measures, refit = refit)
    else:
        run_summary = grid_search_custom( data = data, algorithm_name = algorithm_name
                                         , measures = measures, return_train_measures = return_train_measures)

    return run_summary

def grid_search_surprise( data : dict, algorithm_name : str, cv_iterator : str
                         , measures : list = ['mae', 'rmse', 'mse', 'fcp'], return_train_measures : bool = True, refit : bool = False):

    # define cross validation iterator
    cv_iterator_ms = { 'KFold' : KFold(**config['training']['model_selection']['KFold'] )
                      , 'RepeatedKFold' : RepeatedKFold(**config['training']['model_selection']['RepeatedKFold'] ) }

    grid_search = GridSearchCV(
        algo_class = algorithms[algorithm_name]
        , param_grid = config['training']['params_algo'][algorithm_name]
        , measures = measures
        , cv = cv_iterator_ms[cv_iterator]
        , refit = refit
        , return_train_measures = return_train_measures
    )

    grid_search.fit(data['train']['model'])
    run_summary = pd.DataFrame.from_dict(grid_search.cv_results)

    run_summary.drop(columns = [col for col in run_summary.columns if any(col.startswith(drop_col) for drop_col in ['mean_', 'std_', 'rank_', 'params'])], inplace = True )
    run_summary = run_summary[ [col for col in run_summary.columns if col.startswith('param_')] + [col for col in run_summary.columns if not col.startswith('param_')] ]
    run_summary.columns = run_summary.columns.str.replace('param_', "", regex=False)

    return run_summary

def grid_search_custom( data : dict, algorithm_name : str
                       , measures : list = ['mae', 'rmse', 'mse', 'fcp'], return_train_measures : bool = True):

    algo_class = algorithms[algorithm_name]
    params_algo = params_grid(algorithm_name = algorithm_name)
    run_summary = []
    
    for params_i in params_algo:
        run_summary_i = params_i.copy()
        for split_i in range(0, len(data['ms_mdl'].keys()), 2):
            
            algo_class_i = algo_class(**params_i)
            algo_class_i.fit( data['ms_mdl'][list(data['ms_mdl'].keys())[split_i]] )

            predictions = algo_class_i.test( data['ms_mdl'][list(data['ms_mdl'].keys())[split_i+1]] )

            # run_summary_i['split'+str(split_i//2)+'_test_mae'] = accuracy.mae(predictions, verbose = False)
            measure_results = m23_evaluation.eval_measures(predictions, measures)
            for m in measures:
                run_summary_i['split'+str(split_i//2)+'_test_'+m] = measure_results[m]
            
            if return_train_measures:
                trainset_raw_model = m11_load_data.load_from_df( data['ms'][list(data['ms_mdl'].keys())[split_i]] )
                trainset_raw_model = [ trainset_raw_model.df.iloc[i].to_list() for i in range(len(trainset_raw_model.df)) ]
                predictions_train = algo_class_i.test( trainset_raw_model )
                
                # run_summary_i['split'+str(split_i//2)+'_train_mae'] = accuracy.mae(predictions_train, verbose = False)
                measure_results_train = m23_evaluation.eval_measures(predictions_train, measures)
                for m in measures:
                    run_summary_i['split'+str(split_i//2)+'_train_'+m] = measure_results_train[m]

        run_summary.append(run_summary_i)
    
    return pd.DataFrame(run_summary)

def params_grid(algorithm_name : str):
    """
    Transform parameters values into pairs of coordinates.
    It has the read_yaml function as a dependency.

    Args:
        algorithm : algorithm to be tested
    Returns
        params    : parameters to be used in GridSearch
        
    """
    
    params_algo = config['training']['params_algo'][algorithm_name]
    keys, values = list(params_algo.keys()), list(params_algo.values())

    # generating all possible combinations - pairs of coordinates
    all_combinations = product(*values)

    # creating a list of dictionaries with the parameter's name and value
    dict_list = [{keys[i]: combination[i] for i in range(len(keys))} for combination in all_combinations]
    
    return dict_list
