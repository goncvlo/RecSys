import pandas as pd
#from surprise import Dataset, Reader

def load_data(config:dict)->dict:

    dataframes=dict()
    kwargs = {'encoding':'latin-1', 'header':None}
    for df_name in list(config.keys()):
        sep="|"
        if df_name=='ratings':
            sep="\t"
        dataframes[df_name]=pd.read_csv(
            filepath_or_buffer=config[df_name]['path']
            , names=config[df_name]['columns']
            , sep=sep
            , **kwargs 
        )
    return dataframes

def load_from_df(df:pd.DataFrame):
    """
    Convert dataframe into readable object to use in surprise models.

    Args:
        df_ratings : pd.DataFrame with given ratings of users per movie
    Returns   
    """

    reader = Reader(rating_scale = (0,5))
    return Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)