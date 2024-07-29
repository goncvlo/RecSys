import pandas as pd

def prepare_data(dataframes:dict):

    for df_name in list(dataframes.keys()):
        
        # col range is [0,len(col)]
        for col in ['userId', 'itemId']:
            if col in dataframes[df_name].columns:
                dataframes[df_name][col]=dataframes[df_name][col] - 1
    
    # transform users dataframe
    dataframes['users']=dataframes['users']\
        .astype(dtype={'gender':'string', 'job':'string', 'zip_code':'string'})
    
    # transform items dataframe
    dataframes['items']=dataframes['items']\
        .drop(columns=['video_release_date'])\
        .astype(dtype={'movie_title':'string', 'IMDb_URL':'string'})
    
    # transform ratings dataframe
    dataframes['ratings']['date']=pd.to_datetime(dataframes['ratings']['timestamp'],unit='s').dt.date.astype('string').str.replace('-','')
    dataframes['ratings']['time']=pd.to_datetime(dataframes['ratings']['timestamp'],unit='s').dt.time.astype('string').str.replace(':','')
    
    return dataframes