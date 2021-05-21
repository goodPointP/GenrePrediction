from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pickle

#%%
with open('feat_matrix.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)
 
test_matrix = np.array(feat_matrix[:,7:])
labels = [list(i) for i in feat_matrix[:, 2]]

#creates a samples X set(labels) array of zeros and inserts ones in places of label mappings
label_matrix = np.zeros((len(labels), 22))
for i, label in enumerate(labels):
    for l in label:
        label_matrix[i][l] = 1

X_train, X_test, y_train, y_test = split_data(test_matrix, label_matrix)

#%%

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)
rf_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

mlp = MLPClassifier()

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

#%% FeatureUnion Approach

#%% PCA approach
titles = np.concatenate([i for i in feat_matrix[:,0]])
desc = np.concatenate([i for i in feat_matrix[:,5]])

pca = PCA(n_components = 100)
titles_reduced = pca.fit_transform(titles)
desc_reduced = pca.fit_transform(desc)

features = np.hstack((titles_reduced, desc_reduced))

#%% Meta-modelling approach
t_X_train, t_X_test, t_y_train, t_y_test = split_data(titles_reduced, label_matrix)
d_X_train, d_X_test, d_y_train, d_y_test = split_data(desc_reduced, label_matrix)

title_clf = RandomForestClassifier()
title_clf.fit(t_X_train, t_y_train)
title_clf_prob = title_clf.predict_proba(t_X_test)

desc_clf = RandomForestClassifier()
desc_clf.fit(d_X_train, d_y_train)
desc_clf_prob = desc_clf.predict_proba(d_X_test)

#%%

combined_preds = []
for movie in y_test:
    pred_comb = np.zeros(len(y_test.T))
    for i, y in enumerate(pred_comb):
        pred_comb[i] = (title_clf_prob[i][1] + desc_clf_prob[i][1]) / 2
    combined_preds.append(pred_comb)
        
