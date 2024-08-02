import pandas as pd

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

class Dataset_custom:
    def __init__(self, raw_ratings):
        self.raw_ratings = raw_ratings

    @classmethod
    def from_df(cls, df):
        """Convert dataframe into a list of ratings and return a Dataset instance."""
        df = df[["userId", "itemId", "rating", "date"]]
        raw_ratings = [
            (uid, iid, float(r), date, None)
            for uid, iid, r, date in df.itertuples(index=False)
        ]
        return cls(raw_ratings)