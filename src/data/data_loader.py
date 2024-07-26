import pandas as pd

def load_data(config):

    dataframes=dict()
    kwargs = {'encoding' : 'latin-1', 'header' : None}
    for df_name in list(config.keys()):
        dataframes[df_name]=pd.read_csv(
            filepath_or_buffer=config[df_name]['path']
            , names=config[df_name]['columns']
            , sep="|"
            , **kwargs 
        )
    return dataframes