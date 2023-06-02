import pandas as pd
import numpy as np
import json
import ast
import pickle
import os
import scipy.spatial
from scipy.spatial.distance import pdist, squareform
import glob

# Machine Learning Imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from kmodes.kmodes import KModes
from sklearn.decomposition import TruncatedSVD
import gower
import random

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

def _random_selection(df, _rank_list):
    print(df.shape)
    cuts_idx = [random.randint(0, len(df)-1) for _ in range(100)]
    rank_list = [_rank_list[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list

def _simple_sorting(df, _rank_list):
    print(df.shape)
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

def _simple_kmodes(df, _rank_list, cut_round, n_clusters=N_CLUSTERS, last_round=1000):
    file_path = 'temp_files/kmodes.pickle'
    if cut_round <= last_round:
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
        if cut_round == last_round:
            # Open the file in write mode
            with open(file_path, 'wb') as file:
                # Use pickle.dump() to write the model object to the file
                pickle.dump(kmodes, file)
            print("Model saved...")
    else:
        print('Using trained model...')
        df=_preprocess_df(df)
        # print shape of dataset
        print('Dataset shape:', df.shape)
        # Open the file in read mode
        with open(file_path, 'rb') as file:
            # Use pickle.load() to load the model object from the file
            kmodes = pickle.load(file)

        df['cluster']=kmodes.predict(df[['A','B','C',2,5]])
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

    model = DecisionTreeRegressor()
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    
    df[2] = y_pred
    
    return df

def _create_sparse_df(df):
    # # ***** create sparse one-hot encoded dataset *******
    X = pd.DataFrame(0, index=range(df.shape[0]), columns=['col_{}'.format(i) for i in range(100)])

    def set_values(row):
        A = row['A']
        B = row['B']
        C = row['C']
        X.loc[row.name, 'col_{}'.format(A)] = 1
        X.loc[row.name, 'col_{}'.format(B)] = 1
        X.loc[row.name, 'col_{}'.format(C)] = 1

    df.apply(set_values, axis=1)

    X = pd.get_dummies(df[['A', 'B', 'C']], columns=['A', 'B', 'C'], prefix='col')
    X = X.groupby(level=0, axis=1).max()

    return X

def _rank_list_to_sparse(rank_list):
    rank_list=pd.DataFrame(rank_list)
    rank_list['A'] = rank_list[1].apply(lambda x: x[0])
    rank_list['B'] = rank_list[1].apply(lambda x: x[1])
    rank_list['C'] = rank_list[1].apply(lambda x: x[2])
    rank_list_sparse = _create_sparse_df(rank_list)
    return rank_list_sparse

def _similarity_matrix(df, df_sparse):
    # jaccard = scipy.spatial.distance.cdist(df_sparse, df_sparse, metric='jaccard')
    
    # similarity_matrix = 1-pd.DataFrame(data=jaccard, columns=df_sparse.index.values,  
    #                          index=df_sparse.index.values)
    # df['similarity']=similarity_matrix.sum(axis=1)

    # return similarity_matrix, df

    df_sparse_matrix = df_sparse.values
    similarity_matrix = pdist(df_sparse_matrix, metric='cosine')
    similarity_matrix = squareform(similarity_matrix)
    similarity_matrix = pd.DataFrame(similarity_matrix, index=df_sparse.index, columns=df_sparse.index)
    df['similarity'] = np.mean(similarity_matrix, axis=1)
    return similarity_matrix, df

def _read_rank_list(filename):
    with open(filename, 'rb') as f:
        rank_list = pickle.load(f)
        
    rank_list = pd.DataFrame(rank_list)
    return rank_list

def _read_all_rank_lists():
    # Get all CSV files in the folder starting with "rank"
    pickle_files = glob.glob("temp_files/rank_list_*.pickle")
    # Concatenate all DataFrames into a single DataFrame
    rank_list = pd.concat(_read_rank_list(f) for f in pickle_files)
    rank_list.reset_index(inplace=True, drop=True)
    rank_list['cut_round'] = (rank_list.index // 100) + 1 
    return rank_list

def _previously_selected_triplets(df):
    rank_list = _read_all_rank_lists()
    unique_triplets = pd.DataFrame(rank_list[1].astype(str).value_counts()).reset_index()
    unique_triplets.rename(columns={1: 'count', 'index':'triplet'}, inplace=True)
    rounds = []
    for row in range(len(unique_triplets)):
        triplet = str(unique_triplets['triplet'].iloc[row])
        rounds.append(rank_list[rank_list[1].astype(str)==triplet]['cut_round'].values)
        
    unique_triplets['rounds']=rounds

    df['prev_selected'] = df[1].astype(str).isin(unique_triplets['triplet'].astype(str)).astype(int)

    selected_rows = df['prev_selected'] == 1
    triplets = df.loc[selected_rows, 1].astype(str)
    matching_counts = unique_triplets.loc[unique_triplets['triplet'].astype(str).isin(triplets), 'count'].values
    df.loc[selected_rows, 'prev_selected'] = df.loc[selected_rows, 'prev_selected'] * matching_counts
    
    return df, unique_triplets

def _jaccard_similarity(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

# 1 means complete similar (the same)
def _compute_similarity(df, rank_list):
    cuts_a = df[1].values
    cuts_b = rank_list[1].values

    average_similarities = []

    for element in cuts_a:
        set_a = set(element)
        element_similarities = []
        
        for other_element in cuts_b:
            set_b = set(other_element)
            similarity = _jaccard_similarity(set_a, set_b)
            element_similarities.append(similarity)
        
        average_similarity = sum(element_similarities) / len(element_similarities)
        average_similarities.append(average_similarity)

    df['avg_similarity'] = average_similarities

    return df

def _memory_kmodes(df, _rank_list, n_clusters=N_CLUSTERS):
    df=_preprocess_df(df)
    
    # print shape of dataset
    print('Dataset shape:', df.shape)
    kmodes=KModes(n_clusters=N_CLUSTERS)
    df['cluster']=kmodes.fit_predict(df[['A','B','C',2,5]])

    n_elements = int(100/n_clusters) * 10
    SELECTED_CUTS = [df[df['cluster'] == cls].sort_values(by=2, ascending=False).index[:n_elements].values for cls in range(n_clusters)]
    selected_indices = np.concatenate(SELECTED_CUTS)
    df = df.iloc[selected_indices]   

    cuts_idx = df.sort_values(by='avg_similarity', ascending=True).head(100).index
    rank_list = [_rank_list[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list


def _similarity_measure(df, rank_list):
    rank_list_sparse= _rank_list_to_sparse(rank_list)
    dims = rank_list_sparse.sum().sort_values(ascending=False).index
    dims=pd.DataFrame(dims)
    dims[0]=dims[0].apply(lambda x: int(x.split('_')[1]))

    elements = []
    for row in range(len(df)):
        count = 0
        for d in dims[0][:10].values:
            if d in df[1].iloc[row]:
                count+=1
        elements.append(count)

    df['times'] = elements

    return df




