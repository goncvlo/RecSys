from surprise import accuracy

def eval_measures( predictions, measures : list = ['mae', 'rmse', 'mse', 'fcp'] ):

    measure_results = {
        'mae' : accuracy.mae(predictions, verbose = False)
        , 'rmse' : accuracy.rmse(predictions, verbose = False)
        , 'mse' : accuracy.mse(predictions, verbose = False)
        , 'fcp' : accuracy.fcp(predictions, verbose = False)
    }
        
    return { key : measure_results[key] for key in measures }
    