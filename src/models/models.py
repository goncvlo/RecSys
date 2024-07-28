import yaml
from surprise.model_selection import KFold, RepeatedKFold, GridSearchCV
from surprise import KNNWithMeans, SVD, NMF, CoClustering
from src.data.data_loader import load_from_df
import pandas as pd

# define algorithm objects and read its param grid
algo_classes={'KNNWithMeans':KNNWithMeans, 'SVD':SVD, 'NMF':NMF, 'CoClustering':CoClustering}
with open('src/models/algo_params.yml', 'r') as file:
    param_grid=yaml.load(file, Loader=yaml.SafeLoader)

# define cross-validation iterator and its params
cv_iterators={'KFold':KFold(n_splits=5, random_state=0, shuffle=False)
              , 'RepeatedKFold':RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)}

class grid_search():

    def __init__(self, algo_class:str, measures:list, cv:str, return_train_measures:bool):
        self.algo_class=algo_classes[algo_class]
        self.param_grid=param_grid[algo_class]
        self.measures=measures
        self.cv=cv_iterators[cv]
        self.return_train_measures=return_train_measures
    
    def fit(self, data:pd.DataFrame):

        gs=GridSearchCV(
            algo_class=self.algo_class
            , param_grid=self.param_grid
            , measures=self.measures
            , cv=self.cv
            , return_train_measures=self.return_train_measures
            )
        # prepare ingestion into surprise models
        data=load_from_df(data)
        
        # fit data into grid search and return results
        gs.fit(data=data)
        cv_results=pd.DataFrame.from_dict(gs.cv_results)
        cv_results.drop(
            columns=[col for col in cv_results.columns if any(col.startswith(drop_col) for drop_col in ['mean_', 'std_', 'rank_', 'params'])]
            , inplace=True
            )
        #cv_results.columns = cv_results.columns.str.replace('param_', "", regex=False)
        return cv_results
