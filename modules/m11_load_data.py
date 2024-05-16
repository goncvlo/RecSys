import yaml
import pandas as pd
from surprise import Dataset, Reader

def read_yaml( location : str = 'config.yaml' ):
    
    with open(location, 'r') as file:
        file = yaml.load(file, Loader=yaml.SafeLoader)
    return file

def load_data():
    """
    Fetch data from Data folder. It has the read_yaml function as a dependency.

    Args:
        None
    Returns
        df_users   : pd.DataFrame with users' personal info
        df_items   : pd.DataFrame with movies' details
        df_ratings : pd.DataFrame with given ratings of users per movie
    """

    config_load_data = read_yaml()['load_data']
    kwargs = {'encoding' : 'latin-1', 'header' : None}
    
    df_users   = pd.read_csv( config_load_data['users']
                             , names = ['userId', 'age', 'gender', 'job', 'zip_code']
                             , sep="|"
                             , dtype = {'gender' : 'string', 'job' : 'string', 'zip_code' : 'string'}
                             , **kwargs )
    
    df_items  = pd.read_csv( config_load_data['items']
                            , names = ['itemId', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action'
                                       , 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy'
                                       , 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                            , sep="|"
                            , dtype = {'movie_title' : 'string', 'IMDb_URL' : 'string'}
                            , **kwargs )
    
    df_ratings = pd.read_csv( config_load_data['ratings']
                             , names = ['userId', 'itemId', 'rating', 'timestamp']
                             , sep="\t"
                             , **kwargs )

    return df_users, df_items, df_ratings

def load_from_df( df_ratings : pd.DataFrame ):
    """
    Convert dataframe into readable object to use in surprise models.

    Args:
        df_ratings : pd.DataFrame with given ratings of users per movie
    Returns
        
    """
    reader = Reader(rating_scale = (0,5))
    return Dataset.load_from_df(df_ratings[["userId", "itemId", "rating"]], reader)
