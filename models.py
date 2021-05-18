from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, HalvingGridSearchCV
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
rf_acc = metrics.accuracy_score(y_test, rf_pred)

mlp = MLPClassifier()

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.accuracy_score(y_test, mlp_pred)
mlp_rep = metrics.classification_report(y_test, mlp_pred)