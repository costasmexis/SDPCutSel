import pandas as pd
import numpy as np
import json
import ast
import pickle
import glob
from sklearn.tree import DecisionTreeRegressor


def _read_rank_list(filename):
    with open(filename, 'rb') as f:
        rank_list = pickle.load(f)
        
    rank_list = pd.DataFrame(rank_list)
    return rank_list

def _read_data(filename):
    return pd.read_csv(filename, index_col=0)


# Get all CSV files in the folder starting with "rank"
pickle_files = glob.glob("temp_files/rank_list_*.pickle")
dfs = []
for f in pickle_files:
    temp_rl = _read_rank_list(f)
    dfs.append(temp_rl)
rank_list = pd.concat(dfs)

# Get all CSV files in the folder starting with "rank"
data = glob.glob("temp_files/data_main_*.csv")
dfs = []
for f in data:
    temp = _read_data(f)
    dfs.append(temp)
df = pd.concat(dfs)

# Get all CSV files in the folder starting with "rank"
data = glob.glob("temp_files/data_sparse_*.csv")
dfs = []
for f in data:
    temp = _read_data(f)
    dfs.append(temp)
df_sparse = pd.concat(dfs)


# data
dataset = df_sparse.copy()
dataset['target'] = df['2'].copy()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

X=dataset.drop('target',axis=1)
y=dataset['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

tree=DecisionTreeRegressor()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
error = mean_absolute_error(y_test,y_pred)
print(error)

results = pd.DataFrame()
results['true'] = y_test
results['pred'] = y_pred


