from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import Stemmer
from sklearn.model_selection import GridSearchCV


#%% Opening data
with open('data/dataset_final.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)

with open('data/image_matrix.pkl', 'rb') as f:
    image_matrix = pickle.load(f)
    
#### YOUR CODE HERE "WITH OPEN...."

#%% Creating X and y

label_matrix = CountVectorizer().fit_transform(feat_matrix['genre']).toarray()
description_matrix = feat_matrix['description']

#%% Preprocessing

stemmer = Stemmer.Stemmer('english')
texts = remove_punc_stop(description_matrix)
texts_stemmed = [' '.join(stemmer.stemWords(text)) for text in texts]

#%% Train-test split for FITTING purposes

X_train_fitter, X_test_fitter, y_train_fitter, y_test_fitter = split_data(texts_stemmed, label_matrix)

tfidf = TfidfVectorizer(analyzer='word', max_features=1000)
X_train_tfidf_fitter = tfidf.fit_transform(X_train_fitter).toarray()

#%% Creating the combined feature matrix

X_word2vec = word2vec_matrix(texts_stemmed)
X_tfidf = tfidf.transform(texts_stemmed).toarray()
X_text_combined = np.hstack((X_tfidf, X_word2vec))

#### YOUR CODE HERE (BELOW) ADD NEW MATRIX TO X_IMAGE_TEXT_COMBINED
X_image_text_combined = np.hstack((X_tfidf, X_word2vec, image_matrix))

#%% Train-test for PCA

X_train_temp, X_test_temp, y_train, y_test = split_data(X_image_text_combined, label_matrix)

#%% PCA - choosing the minimum number of components to express 99% variance

pca = PCA().fit(X_train_temp)
cumsum = np.cumsum(pca.explained_variance_ratio_)
pca_X_train = pca.transform(X_train_temp)[:,:np.argmax(cumsum > 0.99)]
pca_X_test = pca.transform(X_test_temp)[:,:np.argmax(cumsum > 0.99)]


#%% Setting final X Train/Test
X_train = pca_X_train
X_test = pca_X_test
#%% Running classifiers, at this point ignoring KNN and RF as they are generally performing worse
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)
# knn_acc = metrics.f1_score(y_test, knn_pred, average='micro')

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
# rf_prob = rf.predict_proba(X_test)
# rf_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

mlp = MLPClassifier(activation='relu', alpha=0.0001, learning_rate_init=0.001, hidden_layer_sizes=(200,), solver='sgd', max_iter=1000)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.f1_score(y_test, mlp_pred, average = "micro")

mul_clf = ClassifierChain(LogisticRegression(C=1, solver='lbfgs', max_iter=500))
mul_clf.fit(X_train, y_train)
mul_pred = mul_clf.predict(X_test)
mul_acc = metrics.f1_score(y_test, mul_pred, average = "micro")

# KNN, MLP, MUL, RF
#chance (around) = 44 
#tfidf-100: 50, 54, 57, 48
#tfidf-1000: 49, 64, 67, 34
#tfidf-10000: 59, 63, 67, 18
#word2vec: 64, 63, 69, 63 
#n_people: 41, 41, 51, 41
#domi_clrs: 52, 54, 49, 51
#como_clrs: 49, 53, 50, 49
#clr_hist: 47, 53, 55, 39
#text data: #, 62, 67, 63
#image data: #, 53, 57, 48
#combined: 64, 71.5, 72, 47
#%% Hyperparam tuning
# hyper_X_train_temp, hyper_X_test_temp, hyper_y_train, hyper_y_test = split_data(X_image_text_combined, label_matrix)

# hyper_pca = PCA().fit(hyper_X_train_temp)
# hyper_cumsum = np.cumsum(pca.explained_variance_ratio_)

# hyper_X_train = pca.transform(X_train_temp[:500])[:,:np.argmax(hyper_cumsum > 0.99)]
# hyper_X_test = pca.transform(X_test_temp[:500])[:,:np.argmax(hyper_cumsum > 0.99)]

# mlp_params = {'hidden_layer_sizes':[50, 100, 200], 'activation':('identity', 'logistic', 'tanh','relu'),'solver':('lbfgs', 'sgd', 'adam'), 'alpha':[0.0001, 0.001, 0.00001], 'max_iter':[1000]}
# mlp_tuner = MLPClassifier()
# mlp_gscv = GridSearchCV(mlp_tuner, mlp_params)
# mlp_gscv.fit(hyper_X_train, hyper_y_train[:500])
# f = ClassifierChain(LogisticRegression(C=1, solver='lbfgs', max_iter=500))
#{'estimator__C':[0.75, 1, 1.25], 'classifier__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
# f.fit(hyper_X_train, y_train[:500])
# g = f.score(hyper_X_test, y_test[:500])

#%% KFOLD
# X = np.vstack((X_train, X_test))
# y = np.vstack((y_train, y_test))
# mlp_res = []
# lr_res = []

# kf = KFold(n_splits=5)
# for train_index, test_index in kf.split(X, y):
#     X_tr, X_te = X[train_index], X[test_index]
#     y_tr, y_te = y[train_index], y[test_index]
    
#     mlp_kfold = MLPClassifier(activation='relu', alpha=0.0001, learning_rate_init=0.001, hidden_layer_sizes=(200,), solver='sgd', max_iter=1000)
#     mlp_kfold.fit(X_tr, y_tr)
#     mlp_res.append(metrics.f1_score(y_te, mlp_kfold.predict(X_te), average = "micro"))
    
#     lr_kfold = ClassifierChain(LogisticRegression(max_iter=500))
#     lr_kfold.fit(X_tr, y_tr)
#     lr_res.append(metrics.f1_score(y_te, lr_kfold.predict(X_te), average = "micro"))
    
#lr: 0.725, 0.718, 0.711, 0.729, 0.729; mean = 0.723
#mlp: 0.709, 0.703, 0.697, 0.721, 0.716; mean = 0.709