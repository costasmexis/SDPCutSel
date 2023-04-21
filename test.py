import pandas as pd
import numpy as np
import json
import ast
from kmodes.kmodes import KModes
import gower
from itertools import combinations
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

df = pd.read_csv("temp_df_pop.csv", index_col=0)
df.reset_index(drop=True, inplace=True)

df['1'] = df['1'].apply(lambda x: json.loads(x))
df['3'] = df['3'].apply(lambda x: ast.literal_eval(x))
df['3'] = [list(t) for t in df['3']]
df[['A', 'B', 'C']] = df['1'].apply(lambda x: pd.Series(x))

df.drop(columns=['1','3','4'], inplace=True)

def _create_sparse_df(df):
    # # ***** create sparse one-hot encoded dataset *******
    X = pd.DataFrame(0, index=range(df.shape[0]), columns=['col_{}'.format(i) for i in range(70)])

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

    X['feasibility'] = df['2'].copy()
    X['optimality'] = df['5'].copy()
    return X

'''KMEANS'''
def _kmeans_clustering(df, df_pop, n_clusters=100):

    # KModes for Categorical Variables
    N_CLUSTERS = n_clusters
    kmeans = KMeans(n_clusters=N_CLUSTERS)

    kmeans.fit(df)
    df['kmeans'] = kmeans.labels_

    NUM_ELEMENTS = int(100/N_CLUSTERS)
    try:
        SELECTED_CUTS = [df[df['kmeans'] == cls].sort_values(by='optimality', ascending=False).index[:NUM_ELEMENTS].values for cls in range(N_CLUSTERS)]
    except KeyError:
        SELECTED_CUTS = [df[df['kmeans'] == cls].sort_values(by='2', ascending=False).index[:NUM_ELEMENTS].values for cls in range(N_CLUSTERS)]
    SELECTED_CUTS = [sublst for arr in SELECTED_CUTS for sublst in arr]

    # get the elements of the initial list based on the index
    # cuts_idx = df[:100].index # get index of selected cuts
    cuts_idx = SELECTED_CUTS
    rank_list = [df_pop[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list


def _agglomerative_clustering(df, df_pop, n_clusters=100):
    N_CLUSTERS = n_clusters
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    clustering.fit(df)
    df['clustering'] = clustering.labels_

    NUM_ELEMENTS = int(100/N_CLUSTERS)
    SELECTED_CUTS = [df[df['clustering'] == cls].sort_values(by='optimality', ascending=False).index[:NUM_ELEMENTS].values for cls in range(N_CLUSTERS)]
    SELECTED_CUTS = [sublst for arr in SELECTED_CUTS for sublst in arr]

    # get the elements of the initial list based on the index
    # cuts_idx = df[:100].index # get index of selected cuts
    cuts_idx = SELECTED_CUTS
    rank_list = [df_pop[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list

def _spectral_clustering(m, df, df_pop, n_clusters=100):
    
    N_CLUSTERS = n_clusters
    clustering = SpectralClustering(n_clusters=N_CLUSTERS, affinity='precomputed')
    clustering.fit(m)
    df['clustering'] = clustering.labels_

    NUM_ELEMENTS = int(100/N_CLUSTERS)
    try:
        SELECTED_CUTS = [df[df['clustering'] == cls].sort_values(by='optimality', ascending=False).index[:NUM_ELEMENTS].values for cls in range(N_CLUSTERS)]
    except KeyError:
        SELECTED_CUTS = [df[df['clustering'] == cls].sort_values(by='2', ascending=False).index[:NUM_ELEMENTS].values for cls in range(N_CLUSTERS)]
    SELECTED_CUTS = [sublst for arr in SELECTED_CUTS for sublst in arr]

    # get the elements of the initial list based on the index
    # cuts_idx = df[:100].index # get index of selected cuts
    cuts_idx = SELECTED_CUTS
    rank_list = [df_pop[i] for i in cuts_idx] # return element list based on their cuts
    return rank_list


df.sort_values(by='2', ascending=False, inplace=True)
SELECTED_CUTS = df[:100].index.values



