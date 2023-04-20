import pandas as pd
import numpy as np
import json
import ast
from kmodes.kmodes import KModes
import gower
from itertools import combinations
from tqdm import tqdm

df = pd.read_csv("test.csv", index_col=0)
1/0
df.sort_values(by='2', ascending=False, inplace=True)

df['1'] = df['1'].apply(lambda x: json.loads(x))
df['3'] = df['3'].apply(lambda x: ast.literal_eval(x))
df['3'] = [list(t) for t in df['3']]
df[['A', 'B', 'C']] = df['1'].apply(lambda x: pd.Series(x))

df_toselect = df[['A','B','C']][:100].copy()

def dissimilarity(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set == b_set:
        return 0
    else:
        return len(a_set.union(b_set)) - len(a_set.intersection(b_set))

n = len(df_toselect)
dissimilarity_matrix = np.zeros((n,n))
# Compute dissimilarity between all pairs of rows
for i in tqdm(range(n)):
    for j in range(i+1, n):
        listA = [df_toselect['A'].iloc[i], df_toselect['B'].iloc[i], df_toselect['C'].iloc[i]]
        listB = [df_toselect['A'].iloc[j], df_toselect['B'].iloc[j], df_toselect['C'].iloc[j]]

        dissimilarity_matrix[i, j] = dissimilarity(listA, listB)
        dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

# Print the resulting dissimilarity matrix
dissimilarity_matrix = pd.DataFrame(dissimilarity_matrix)
dissimilarity_matrix['sum'] = dissimilarity_matrix.sum(axis=1)
print(dissimilarity_matrix)

# idx = []
# for col in dissimilarity_matrix.columns:
#     if len(idx) == 100: break
#     if dissimilarity_matrix[col].iloc[0] > 4: idx.append(col)
