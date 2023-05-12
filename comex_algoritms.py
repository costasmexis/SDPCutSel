import pandas as pd
import json
import ast
import pickle
import os
import scipy.spatial

# Machine Learning Imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from kmodes.kmodes import KModes
from sklearn.decomposition import TruncatedSVD
import gower

''' Define constants'''
N_CLUSTERS = 100

def _save_pickle_ranklist(rank_list, cut_round):
    
    ''' SAVE rank_list TO pickle '''
    FILENAME= 'temp_files/rank_list_{}.pickle'.format(cut_round)
    with open(FILENAME, 'wb') as f:
        pickle.dump(rank_list, f)
    
def _save_data_csv(df, df_sparse, cut_round):
    ''' SAVE main dataset TO pickle'''
    FILENAME= 'temp_files/data_main_{}.csv'.format(cut_round)
    df.to_csv(FILENAME)

    ''' SAVE main dataset TO pickle'''
    FILENAME= 'temp_files/data_sparse_{}.csv'.format(cut_round)
    df_sparse.to_csv(FILENAME)

def _read_data(cut_round):
    df = pd.read_csv('temp_files/data_main_{}.csv'.format(cut_round), index_col=0)
    df_sparse = pd.read_csv('temp_files/data_sparse_{}.csv'.format(cut_round), index_col=0)
    return df, df_sparse

def _preprocess_df(df):
    df[['A', 'B', 'C']] = df[1].apply(lambda x: pd.Series(x))
    return df

def _simple_sorting(df, _rank_list):
    df.sort_values(by=2, ascending=False, inplace=True)
    cuts_idx=df[:100].index.values

    rank_list = [_rank_list[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list

def _simple_kmeans(df, df_sparse, _rank_list, n_clusters=N_CLUSTERS):
    # print shape of dataset
    print('Dataset shape:', df_sparse.shape)
    kmeans=KMeans(n_clusters=n_clusters).fit(df_sparse)
    df['cluster']=kmeans.labels_

    n_elements = int(100/n_clusters)
    SELECTED_CUTS = [df[df['cluster'] == cls].sort_values(by=2, ascending=False).index[:n_elements].values for cls in range(n_clusters)]
    SELECTED_CUTS = [sublst for arr in SELECTED_CUTS for sublst in arr]
    cuts_idx = SELECTED_CUTS
    rank_list = [_rank_list[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list

def _simple_kmodes(df, _rank_list, n_clusters=N_CLUSTERS):
    df=_preprocess_df(df)

    # print shape of dataset
    print('Dataset shape:', df.shape)
    kmodes=KModes(n_clusters=N_CLUSTERS)
    df['cluster']=kmodes.fit_predict(df[['A','B','C',2,5]])

    n_elements = int(100/n_clusters)
    SELECTED_CUTS = [df[df['cluster'] == cls].sort_values(by=2, ascending=False).index[:n_elements].values for cls in range(n_clusters)]
    SELECTED_CUTS = [sublst for arr in SELECTED_CUTS for sublst in arr]
    cuts_idx = SELECTED_CUTS
    rank_list = [_rank_list[i] for i in cuts_idx] # return element list based on their cuts

    return rank_list

def _train_dec_tree(df, df_sparse, cut_round):
    
    _df, _df_sparse = _read_data(1)

    train = _df_sparse.copy()
    train['target'] = _df['2'].copy()
    X_train = train.drop('target',axis=1)
    y_train = train['target'].copy()

    test = df_sparse.copy()
    test['target'] = df[2].copy()
    X_test = test.drop('target',axis=1)
    y_test = test['target'].copy()

    model = XGBRegressor()
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    
    df[2] = y_pred
    
    return df

def _dimensionality_reduction(df_sparse):
    svd = TruncatedSVD(n_components=5)
    # Fit the SVD model to the dataset and transform the dataset
    svd.fit(df_sparse)
    return svd

