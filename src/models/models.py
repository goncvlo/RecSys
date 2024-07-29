from surprise import KNNWithMeans, SVD, NMF, CoClustering
import yaml
from src.models.cv_iterator import KFold, RepeatedKFold, HoldOut
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from src.data.data_loader import Dataset_custom
from surprise.model_selection import GridSearchCV

# define algorithm objects and read its param grid
algo_classes={'KNNWithMeans':KNNWithMeans, 'SVD':SVD, 'NMF':NMF, 'CoClustering':CoClustering}
with open('src/models/algo_params.yml', 'r') as file:
    param_grid=yaml.load(file, Loader=yaml.SafeLoader)

# define cross-validation iterator and its params
cv_iterators={'KFold':KFold(n_splits=5, random_state=0, shuffle=False)
              , 'RepeatedKFold':RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
              , 'TimeSeriesSplit':TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=None, gap=0)
              , 'HoldOut':HoldOut(validation_date='19980312')}

class grid_search():

    def __init__(self, algo_class:str, measures:list, cv:str, return_train_measures:bool):
        self.algo_class=algo_classes[algo_class]
        self.param_grid=param_grid[algo_class]
        self.measures=measures
        self.cv=cv_iterators[cv]
        self.return_train_measures=return_train_measures
    
    def fit(self, data:pd.DataFrame):
        # prepare ingestion into surprise models
        data=Dataset_custom.from_df(data)
        # set grid search params and fit data
        gs=GridSearchCV(
            algo_class=self.algo_class
            , param_grid=self.param_grid
            , measures=self.measures
            , cv=self.cv
            , return_train_measures=self.return_train_measures
            )
        gs.fit(data=data)

        cv_results=pd.DataFrame.from_dict(gs.cv_results)
        cv_results.drop(
            columns=[col for col in cv_results.columns if any(col.startswith(drop_col) for drop_col in ['mean_', 'std_', 'rank_', 'params'])]
            , inplace=True
            )
        #cv_results.columns=cv_results.columns.str.replace('param_', "", regex=False)
        return cv_results