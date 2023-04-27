import pandas as pd
import numpy as np
import json
import ast
from kmodes.kmodes import KModes
import gower
from itertools import combinations
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

from keras.layers import Dense, Input
from keras.models import Model, Sequential

from sklearn.manifold import TSNE

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


'''EMBEDDING USING VAC'''     
def _autoencoder(X):
    n_inputs = X.shape[1]

    autoencoder = Sequential([
        Dense(50, activation='relu', input_shape = (n_inputs,), name='input_layer'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(10,  activation='relu', name='latent_layer'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(50, activation='relu'),
        Dense(n_inputs, activation='relu')
    ])

    autoencoder.compile(optimizer = 'adam', loss = 'mse')
    h = autoencoder.fit(X, X, epochs=2, batch_size=248, 
                        validation_split=0.30,verbose=1)

    # create a new model that outputs the predictions of the latent_layer
    latent_layer_model = Model(inputs=autoencoder.input, 
                            outputs=autoencoder.get_layer('latent_layer').output)

    # get the predictions of the latent_layer for the input X
    latent_layer_predictions = latent_layer_model.predict(X)

    result_df = pd.DataFrame(latent_layer_predictions)

    return result_df


# df.sort_values(by='2', ascending=False, inplace=True)
# SELECTED_CUTS = df[:100].index.values

# X = _create_sparse_df(df)

# vac_df = _autoencoder(X)

# def _learn_manifold(df):
#     df_embedded = TSNE(n_components=2, learning_rate=200,
#                        init='random', perplexity=10).fit_transform(df)
#     return df_embedded

# from sklearn.cluster import KMeans

# embedded = _learn_manifold(vac_df)
# kmeans = KMeans(n_clusters=20).fit(embedded)
# df['kmeans'] = kmeans.labels_
# SELECTED_CUTS = [df[df['kmeans'] == cls].sort_values(by='2', ascending=False).index[:100].values for cls in range(20)]

rank_list = pd.read_csv('rank_list.csv')