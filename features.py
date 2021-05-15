import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%% "Homebrew" dataset

def read_data():
    entries = 1000000
    name_basics = pd.read_csv("data/title.basics.tsv", sep='\t', nrows = entries)
    title_basics = pd.read_csv("data/title.basics.tsv", sep='\t', nrows = entries)
    title_crew = pd.read_csv("data/title.crew.tsv", sep='\t', nrows = entries)
    title_ratings = pd.read_csv("data/title.ratings.tsv", sep='\t', nrows = entries)
    
    return name_basics, title_basics, title_crew, title_ratings

name_basics, title_basics, title_crew, title_ratings = read_data()
dataframe = pd.concat([name_basics, title_basics.iloc[:,4:], title_crew.iloc[:,1:], title_ratings.iloc[:,1:]], axis = 1)
movies = df[df['titleType'] == 'movie']

#%% Prefiltered dataset

data = pd.read_csv("data/IMDb_movies.csv")
data['imdb_title_id'] = [float(i[2:]) for i in data['imdb_title_id']]


#%%
def split_data(data, truth):
    X_train, X_test, y_train, y_test = train_test_split(
    data, truth, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test


df = data.drop(['avg_vote', 'original_title', "date_published", "language"], axis = 1)
truth = data['avg_vote']


#%%
X_train, X_test, y_train, y_test = split_data(data, truth)

lr = LinearRegression()

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = metrics.roc_auc_score(y_test, lr_pred)
lr_rep = metrics.classification_report(y_test, lr_pred)


