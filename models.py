from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

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

#%% Multi-label-out-of-the-box

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)
rf_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

mlp = MLPClassifier()

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.f1_score(y_test, rf_pred, average = "micro")


#%% PCA
titles = np.concatenate([i for i in feat_matrix[:,0]])
desc = np.concatenate([i for i in feat_matrix[:,5]])

pca = PCA(n_components = 100)
titles_reduced = pca.fit_transform(titles)
desc_reduced = pca.fit_transform(desc)

features = np.hstack((titles_reduced, desc_reduced))

#%% Meta-modelling approach - Uses PCA 
t_X_train, t_X_test, t_y_train, t_y_test = split_data(titles_reduced, label_matrix)
d_X_train, d_X_test, d_y_train, d_y_test = split_data(desc_reduced, label_matrix)

title_clf = RandomForestClassifier()
title_clf.fit(t_X_train, t_y_train)
title_clf_prob = title_clf.predict_proba(t_X_test)

desc_clf = RandomForestClassifier()
desc_clf.fit(d_X_train, d_y_train)
desc_clf_prob = desc_clf.predict_proba(d_X_test)

threshold, upper, lower = 0.5, 1, 0
combined_probs = np.hstack([np.delete((np.add(title_clf_prob[i], desc_clf_prob[i]) / 2), 0,1) for i in range(len(y_test.T))])
comb_pred = np.where(combined_probs > threshold, upper, lower)
comb_acc = metrics.f1_score(y_test, comb_pred, average = "micro")

#%% Classifier chains approach

mul_X_train, mul_X_test, mul_y_train, mul_y_test = split_data(features, label_matrix)

mul_clf = ClassifierChain(LogisticRegression())
mul_clf.fit(mul_X_train, mul_y_train)
mul_pred = mul_clf.predict(mul_X_test)
mul_acc = metrics.f1_score(mul_y_test, mul_pred, average = "micro")

