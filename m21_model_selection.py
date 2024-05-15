import m11_load_data
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, TimeSeriesSplit

def cross_validation(df_ratings : pd.DataFrame, cv_iterator : str):
    """
    Split ratings dataframe into training, validation and test sets.
    It has the read_yaml function as a dependency.

    Args:
        df_ratings : pd.DataFrame with given ratings of users per movie
        hold_out   : indicating whether holdout method should be used
    Returns
        
    """

    # split data into training and test sets
    data = train_test_split(df_ratings)
    
    if cv_iterator in ['hold_out', 'ts_split']:
        config_ms = m11_load_data.read_yaml()['training']['model_selection']
        data = train_test_split(df_ratings)
        data['ms'], data['ms_mdl'] = {}, {}    
    
        if cv_iterator=='hold_out':
            # getting start date for validation set
            validation_date = config_ms['hold_out']['validation_date']
    
            # split train set into actual training and validation sets
            data['ms']['ho_train'] = data['train']['raw'][ data['train']['raw']['date'] < validation_date ]
            data['ms']['ho_valid'] = data['train']['raw'][ data['train']['raw']['date'] >= validation_date ]
    
        if cv_iterator=='ts_split':
            # partition train set according to time series cross validation
            tscv = TimeSeriesSplit()
            for i, (train_index, valid_index) in enumerate(tscv.split(data['train']['raw'])):
                data['ms']['tscv'+str(i+1)+'_train'] = data['train']['raw'].iloc[train_index,:]
                data['ms']['tscv'+str(i+1)+'_valid'] = data['train']['raw'].iloc[valid_index,:]
    
        # format data for the algorithm
        for df_name in data['ms'].keys():
            data['ms_mdl'][df_name] = m11_load_data.load_from_df(data['ms'][df_name])
            if df_name.endswith("train"):
                data['ms_mdl'][df_name] = data['ms_mdl'][df_name].build_full_trainset()
            else:
                data['ms_mdl'][df_name] = [ data['ms_mdl'][df_name].df.iloc[i].to_list() for i in range(len(data['ms_mdl'][df_name].df)) ]        
        
    return data

def train_test_split(df_ratings : pd.DataFrame):

    test_date = config_ms = m11_load_data.read_yaml()['training']['test_date']
    data = {'train' : {}, 'test' : {} }
    data['train']['raw'], data['test']['raw'] = df_ratings[ df_ratings['date'] < test_date ], df_ratings[ df_ratings['date'] >= test_date ]
    data['train']['model'], data['test']['model'] = [ m11_load_data.load_from_df(df) for df in [ data['train']['raw'], data['test']['raw'] ]]
    data['test']['model'] = [ data['test']['model'].df.iloc[i].to_list() for i in range(len(data['test']['model'].df)) ]

    return data
    
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
    