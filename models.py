from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pickle
from sklearn.linear_model import Perceptron
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#%%
with open('df_short.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)

feat_matrix = feat_matrix.sample(frac=1, random_state=420)
vectorizer = CountVectorizer()
label_matrix = vectorizer.fit_transform(feat_matrix['genre']).toarray()
description_matrix = feat_matrix['description']
#%% tf_idf

d_tfidf = TfidfVectorizer(analyzer='word', max_features=10000)
d_count = d_tfidf.fit_transform(description_matrix)

#%% PCA

pca = PCA(n_components = 1000)
desc_reduced = pca.fit_transform(d_count.toarray())

#%%
X_train, X_test, y_train, y_test = split_data(desc_reduced, label_matrix)


#%% Multi-label-out-of-the-box

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)
rf_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

mlp = MLPClassifier()

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.f1_score(y_test, mlp_pred, average = "micro")

mul_clf = ClassifierChain(LogisticRegression())
mul_clf.fit(X_train, y_train)
mul_pred = mul_clf.predict(X_test)
mul_acc = metrics.f1_score(y_test, mul_pred, average = "micro")
#%%

f = metrics.hamming_loss(y_test, mul_pred)
#%% Meta-modelling approach - Uses PCA 


desc_clf = RandomForestClassifier()
desc_clf.fit(X_train, y_train)
desc_clf_prob = desc_clf.predict_proba(X_test)

threshold, upper, lower = 0.5, 1, 0
combined_probs = np.hstack([np.delete((np.add(title_clf_prob[i], desc_clf_prob[i]) / 2), 0,1) for i in range(len(y_test.T))])
comb_pred = np.where(combined_probs > threshold, upper, lower)
comb_acc = metrics.f1_score(y_test, comb_pred, average = "micro")


