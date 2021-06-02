import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import chain
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, Normalizer
scaler = StandardScaler()
normalizer = Normalizer()
#%%
with open('data/imageFeaturesNoSIFT.pkl', 'rb') as f:
    image_pickle = pickle.load(f)
    
#%%
image_pickle.columns=['imdb_title_id','n_people', 'alphanumeric_chars', 'most_com_colors', 'dom_color', 'color_hist']
df_image = image_pickle.drop(columns=['imdb_title_id']).sort_index()

#%%
n_people = normalizer.fit_transform(np.array(df_image['n_people']).reshape(-1, 1))
chars = scaler.fit_transform(np.array(df_image['alphanumeric_chars']).reshape(-1,1))
#%%

com_colors = np.array(df_image['most_com_colors'].apply(pd.Series))
com_colors_array = np.divide(np.array([[np.asarray(tup) for tup in row] for row in com_colors]).reshape(com_colors.shape[0], -1), 255)
#%%
dom_colors = np.array(df_image['dom_color'].apply(pd.Series))
dom_colors_array = np.divide(np.array([[np.asarray(tup) for tup in row] for row in dom_colors]).reshape(dom_colors.shape[0], -1), 255)

#%%
color_hist = np.array(df_image['color_hist'].apply(pd.Series))

#%%
image_matrix = np.hstack((n_people, chars, com_colors_array, dom_colors_array, color_hist))

#%%
with open('data/image_matrix.pkl', 'wb') as outfile:
    pickle.dump(image_matrix, outfile)
