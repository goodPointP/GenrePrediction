import pandas as pd
import numpy as np
from collections import Counter
import re
import operator
import string
import pickle
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

#%% Prefiltered dataset

data = pd.read_csv("data/IMDb_movies.csv")
df = data.drop(['imdb_title_id','original_title', "date_published", "language", "votes", "actors", "director", "writer", "production_company", "metascore", "reviews_from_users", "reviews_from_critics", 'worlwide_gross_income'], axis = 1).dropna()

#%% Transform titles to TD-IDF vectors

bagoftitles = set(reduce(operator.concat, [i.split(' ') for i in df['title']]))
t_cv = CountVectorizer(analyzer= 'word', stop_words='english')
t_tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
t_count = t_cv.fit_transform(bagoftitles)
t_tfidf.fit(t_count)

#%% Transform descriptions to TD-IDF vectors

bagofwords = set(reduce(operator.concat, [i.split(' ') for i in df['description']]))
bow_stripped = [i.translate(str.maketrans('', '', string.punctuation)) for i in bagofwords]
d_cv = CountVectorizer(analyzer= 'word', stop_words='english')
d_tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
d_count = d_cv.fit_transform(bow_stripped)
d_tfidf.fit(d_count)

vec = [d_tfidf.transform(d_cv.transform([df['description'].iloc[i]])) for i in range(len(df['description']))]

#%% Maps genres to numbers determined by their frequency in the corpus

genres = list(df['genre'])
genres_stripped = [str(genre).split(',') for genre in genres]
genres_div = []
for genre in genres_stripped:
    helper = []
    for g in genre:
        helper.append(g.strip())
    genres_div.append(helper)
genres_flat = reduce(operator.concat, genres_div)
genres_numbered = dict([tuple((j[0], i)) for i, j in enumerate(Counter(genres_flat).most_common())])
genres_mapped = []
for genre in genres_div:
    gen = [genres_numbered.get(g) for g in genre]
    genres_mapped.append(gen)

#%% Maps countries to numbers determined by their frequency in the corpus

countries = list(df['country'])
countries_stripped = [str(i).split(',', 1)[0] for i in countries]
countries_numbered = dict([tuple((j[0], i)) for i, j in enumerate(Counter(countries_stripped).most_common())])

#%% Preprocess dataframe to only contain numerics

df['title'] = [t_tfidf.transform(t_cv.transform([df['title'].iloc[i]])).toarray() for i in range(len(df['title']))]
df['description'] = [d_tfidf.transform(d_cv.transform([df['description'].iloc[i]])).toarray() for i in range(len(df['description']))]
df['genre'] = genres_mapped
df['country'] = [countries_numbered.get(i) for i in countries_stripped]
df['budget'] = [re.sub("[^0-9]", "", i) for i in df['budget']] #remove non-numeric chars
df['usa_gross_income'] = [i[2:] for i in df['usa_gross_income']] #remove currency chars

#%%

feat_matrix = np.array(df)
with open('feat_matrix.pkl', 'wb') as f:
      pickle.dump(feat_matrix,f)