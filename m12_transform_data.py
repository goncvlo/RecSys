import m11_load_data
import pandas as pd

def transform_data():
    
    df_users, df_movies, df_ratings = m11_load_data.load_data()

    # Transform users' dataframe
    df_users['userId'] = df_users['userId'] - 1

    # Transform movies' dataframe
    df_movies['itemId'] = df_movies['itemId'] - 1
    df_movies.drop(columns = ['video_release_date'], inplace = True)
    
    # Transform ratings' dataframe
    df_ratings['userId'] = df_ratings['userId'] - 1
    df_ratings['itemId'] = df_ratings['itemId'] - 1
    df_ratings['date'] = pd.to_datetime(df_ratings['timestamp'],unit='s').dt.date.astype('string').str.replace('-','')
    df_ratings['time'] = pd.to_datetime(df_ratings['timestamp'],unit='s').dt.time.astype('string').str.replace(':','')

    return df_users, df_movies, df_ratings

def matrix_ui():
    
    df_users, df_movies, df_ratings = transform_data()
    del df_users, df_movies

    return df_ratings[['userId', 'new_movieId', 'rating']].pivot( index = 'userId', columns = 'new_movieId', values = 'rating')
