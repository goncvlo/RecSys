def model_selection(df_ratings : pd.DataFrame, hold_out : bool = False, kfold : bool = False, rep_kfold : bool = False, ts_split : bool = False):
    """
    Split ratings dataframe into training, validation and test sets.
    It has the read_yaml function as a dependency.

    Args:
        df_ratings : pd.DataFrame with given ratings of users per movie
        hold_out   : indicating whether holdout method should be used
    Returns
        
    """
    
    config_ms = config['training']['model_selection']
    data = {'original':{}, 'model':{}}
    
    # predefine train and test sets
    test_date = config['training']['test_date']
    train_set = df_ratings[ df_ratings['date'] < test_date ]
    test_set  = df_ratings[ df_ratings['date'] >= test_date ]
    del df_ratings
    
    if kfold:
        # partition train set according to k-fold method
        kf = KFold(n_splits=config_ms['kfold']['n_splits'], shuffle=config_ms['kfold']['shuffle'], random_state=config_ms['kfold']['random_state'])
        for i, (train_index, valid_index) in enumerate(kf.split(train_set)):

            data['original']['kf'+str(i+1)+'_train'] = train_set.iloc[train_index,:]
            data['original']['kf'+str(i+1)+'_valid'] = train_set.iloc[valid_index,:]

    if rep_kfold:
        # partition train set according to repeated k-fold method
        rkf = RepeatedKFold(n_splits=config_ms['rep_kfold']['n_splits'], n_repeats=config_ms['rep_kfold']['n_repeats'], random_state=config_ms['rep_kfold']['random_state'])
        for i, (train_index, valid_index) in enumerate(rkf.split(train_set)):
            i1, i2 = i//config_ms['rep_kfold']['n_splits']+1, i%config_ms['rep_kfold']['n_splits']+1

            data['original']['rkf'+str(i1)+str(i2)+'_train'] = train_set.iloc[train_index,:]
            data['original']['rkf'+str(i1)+str(i2)+'_valid'] = train_set.iloc[valid_index,:]

    # add test set to data variable
    data['original']['test_set'] = test_set
    # format data for the algorithm
    for df_name in data['original'].keys():
        data['model'][df_name] = m11_load_data.load_from_df(data['original'][df_name])
        if df_name.endswith("train"):
            data['model'][df_name] = data['model'][df_name].build_full_trainset()
        else:
            data['model'][df_name] = [ data['model'][df_name].df.iloc[i].to_list() for i in range(len(data['model'][df_name].df)) ]  
        
    return data