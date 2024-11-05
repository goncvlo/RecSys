from surprise import KNNWithMeans, SVD, NMF, CoClustering, accuracy
import yaml
from src.models.cv_iterator import KFold, RepeatedKFold, HoldOut #TimeSeriesSplit 
import pandas as pd
from src.data.data_loader import Dataset_custom
from surprise.model_selection import GridSearchCV
from surprise import Dataset, Reader
from src.utils.utils import get_top_n

# define algorithm objects and read its param grid
algo_classes={'KNNWithMeans':KNNWithMeans, 'SVD':SVD, 'NMF':NMF, 'CoClustering':CoClustering}
with open('../src/models/algo_params.yml', 'r') as file:
    param_grid=yaml.load(file, Loader=yaml.SafeLoader)

# define cross-validation iterator and its params
cv_iterators={'KFold':KFold(n_splits=5, random_state=0, shuffle=False)
              , 'RepeatedKFold':RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
              #, 'TimeSeriesSplit':TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=None, gap=0)
              , 'HoldOut':HoldOut(validation_date='19980312')}

# define accuracy metrics to evaluate predictions
pred_metrics={'rmse':accuracy.rmse, 'mse':accuracy.mse, 'mae':accuracy.mae, 'fcp':accuracy.fcp}


class grid_search():
    def __init__(self, algo_class:str, measures:list, cv:str, return_train_measures:bool):
        self.algo_class=algo_classes[algo_class]
        self.param_grid=param_grid[algo_class]
        self.measures=measures
        self.cv=cv_iterators[cv]
        self.return_train_measures=return_train_measures
    
    def fit(self, train_set:pd.DataFrame):
        # prepare ingestion into surprise models
        train_set=Dataset_custom.from_df(train_set)
        # set grid search params and fit data
        gs=GridSearchCV(
            algo_class=self.algo_class
            , param_grid=self.param_grid
            , measures=self.measures
            , cv=self.cv
            , return_train_measures=self.return_train_measures
            )
        gs.fit(data=train_set)
        # get cross validation results
        cv_results=pd.DataFrame.from_dict(gs.cv_results)
        # define best score, params and model
        self.best_score=gs.best_score[self.measures[0]]
        self.best_params=gs.best_params[self.measures[0]]
        self.best_estimator=gs.best_estimator[self.measures[0]]
        return cv_results


class modeling():
    def __init__(self, algo_class:str, params:dict, metrics:list=['rmse', 'mse', 'mae', 'fcp']):
        self.algo_class=algo_classes[algo_class](**params)
        self.metrics=metrics

    def fit(self, train_set:pd.DataFrame):
        # prepare ingestion into surprise models
        reader = Reader(rating_scale=(1, 5))
        train_set=Dataset.load_from_df(train_set[["userId", "itemId", "rating"]], reader)
        train_set=train_set.build_full_trainset()
        self.train_set=train_set
        # train algorithm
        self.algo_class.fit(train_set)
    
    def evaluate(self, test_set:pd.DataFrame, train_set:pd.DataFrame):
        # metrics to compute
        eval_metrics=dict()
        # prepare ingestion into surprise models
        train_set, test_set=[list(df[["userId", "itemId", "rating"]].itertuples(index=False, name=None)) for df in [train_set, test_set]]
        train_pred, test_pred = [self.algo_class.test(df) for df in [train_set, test_set]]
        # log metrics
        for metric in self.metrics:
            eval_metrics[metric+'_test']=pred_metrics[metric](predictions=test_pred,verbose=False)
            eval_metrics[metric+'_train']=pred_metrics[metric](predictions=train_pred,verbose=False)
        self.metrics=eval_metrics

    def inference(self, n:int=3):
        # prepare ingestion into surprise package
        test_set=self.train_set.build_anti_testset()
        predictions=self.algo_class.test(test_set)
        top_n=get_top_n(predictions=predictions, n=n)
        top_n=pd.DataFrame([(uid, iid, est) for uid, user_ratings in top_n.items() for iid, est in user_ratings]
                           ,columns=['userId', 'itemId', 'est_rating'])\
                           .sort_values(by=['userId', 'est_rating'], ascending=[True, False])
        return top_n