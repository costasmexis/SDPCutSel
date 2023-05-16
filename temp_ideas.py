import pandas as pd
import numpy as np
import json
import ast
import pickle
import glob
import comex_algoritms as cm

def _read_rank_list(filename):
    with open(filename, 'rb') as f:
        rank_list = pickle.load(f)
        
    rank_list = pd.DataFrame(rank_list)
    return rank_list

def _read_data(filename):
    return pd.read_csv(filename, index_col=0)

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

# Get all CSV files in the folder starting with "rank"
pickle_files = glob.glob("temp_files/rank_list_*.pickle")
dfs = [_read_rank_list(f) for f in pickle_files]
rank_list = pd.concat(dfs)
rank_list.reset_index(inplace=True, drop=True)

data = glob.glob("temp_files/data_main_*.csv")
dfs = [_read_data(f) for f in data]
df = pd.concat(dfs)

data = glob.glob("temp_files/data_sparse_*.csv")
dfs = [_read_data(f) for f in data]
df_sparse = pd.concat(dfs)

rank_list['A'] = rank_list[1].apply(lambda x: x[0])
rank_list['B'] = rank_list[1].apply(lambda x: x[1])
rank_list['C'] = rank_list[1].apply(lambda x: x[2])
rank_list_sparse = _create_sparse_df(rank_list)

'''Calculate a SIMILARITY matrix'''
sim_matrix = cm._similarity_matrix(df_sparse[4000:4005])
print(sim_matrix)
