'''
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

def create_user_item_matrix(file_ratings, file_movies, n_rows):
    '''
    '''
    ratings = pd.read_csv(file_ratings, nrows=n_rows) #dataset to big --> kernel crashs everytime just use first 1_000_000 data

    #make a dict with movieId and title
    movies = pd.read_csv(file_movies, index_col=0)
    movies.drop(columns='genres', inplace=True)

    df_ratings_movie = pd.merge(ratings, movies, how='inner', on='movieId')
    df_ratings_movie.drop(columns=['timestamp','movieId'], inplace=True)
    return df_ratings_movie


def model_recommender(df):
    '''
    '''
    R = df.pivot(index='userId',
            columns='title',
            values='rating'
    )


    R= R.fillna(2.5)

    nmf = NMF(n_components = 150, 
            max_iter=5_000, 
            #alpha = 0.2, 
            l1_ratio= 0.5) # instantiate model
    nmf.fit(R) #fit R to the model

    #create Q: item feature matrix
    Q = pd.DataFrame(nmf.components_, columns=R.columns)

    #create P: user feature matrix
    P = pd.DataFrame(nmf.transform(R), index=R.index)

    #create R_hat: Matrixmultiplication of Q and P
    R_hat = pd.DataFrame(np.dot(P,Q), columns=R.columns, index=R.index)

    #evaluate error: delta(R, R_hat)
    nmf.reconstruction_err_
    return R, P, Q, nmf




def user_recommendation(input_dict, model_function):
    '''
    '''
    R, P, Q, nmf = model_function

    ranking = []
    for i in list(range(0,5)):
        ranking.append(input_dict[sorted(input_dict.keys())[i]])

    titel = []
    for i in list(range(5,10)):
        titel.append(input_dict[sorted(input_dict.keys())[i]])

    dict_user = {titel[i]:ranking[i] for i in range(len(titel))}
    
    new_user = pd.DataFrame(dict_user, index=['new_user'], columns=R.columns)
    new_user = new_user.fillna(2.54)

    user_P = nmf.transform(new_user)

    user_R = pd.DataFrame(np.dot(user_P, Q), columns=R.columns, index=['new_user'])

    recommendations = user_R.drop(columns=titel)

    recommendations_user = list(recommendations.sort_values(axis=1, by='new_user', ascending=False))

    return recommendations_user[:5]


