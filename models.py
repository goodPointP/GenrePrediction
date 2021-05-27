from functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import pickle
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import Stemmer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#%% Opening data
with open('data/dataset_final.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)

with open('data/image_matrix.pkl', 'rb') as f:
    image_matrix = pickle.load(f)
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
X_image_text_combined = np.hstack((X_tfidf, X_word2vec, image_matrix))
X_text_combined = np.hstack((X_tfidf, X_word2vec))
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

# mul_clf = ClassifierChain(LogisticRegression(max_iter=500))
# mul_clf.fit(X_train, y_train)
# mul_pred = mul_clf.predict(X_test)
# mul_acc = metrics.f1_score(y_test, mul_pred, average = "micro")

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
hyper_X_train_temp, hyper_X_test_temp, hyper_y_train, hyper_y_test = split_data(X_image_text_combined, label_matrix)

hyper_pca = PCA().fit(hyper_X_train_temp)
hyper_cumsum = np.cumsum(hyper_pca.explained_variance_ratio_)
#%%
hyper_X_train = hyper_pca.transform(hyper_X_train_temp[:500])[:,:np.argmax(hyper_cumsum > 0.90)]
hyper_X_test = hyper_pca.transform(hyper_X_test_temp[:500])[:,:np.argmax(hyper_cumsum > 0.90)]





mlp_params = {'hidden_layer_sizes':[50, 100, 200], 'activation':('identity', 'logistic', 'tanh','relu'),'solver':('lbfgs', 'sgd', 'adam'), 'alpha':[0.0001, 0.001, 0.00001], 'max_iter':[1000]}
mlp_tuner = MLPClassifier()
mlp_gscv = GridSearchCV(mlp_tuner, mlp_params)
mlp_gscv.fit(hyper_X_train, hyper_y_train[:500])





#%%
counter = 0
alsocounter = 0

for i, (row, alsorow) in enumerate(zip(mul_pred, y_test)):
    if False in (row == alsorow):
        alsocounter += 1
    else:
        counter += 1
        

# f = metrics.hamming_loss(y_test, mul_pred)
#%% Meta-modelling approach - Uses PCA 
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import ExtraTreeClassifier
# extra_tree = ExtraTreeClassifier(random_state=0)
# extra_tree.fit(X_train, y_train)
# extra_pred = extra_tree.predict(X_test)
# extra_acc = metrics.f1_score(y_test, extra_pred, average = "micro")

# # etb = BaggingClassifier(extra_tree, random_state=0)
# # etb_acc = etb.predict(X_test)
# #etb_acc = metrics.f1_score(y_test, etb_pred, average = "micro")

# desc_clf = RandomForestClassifier()
# desc_clf.fit(X_train, y_train)
# desc_clf_prob = desc_clf.predict_proba(X_test)

# threshold, upper, lower = 0.5, 1, 0
# combined_probs = np.hstack([np.delete((np.add(title_clf_prob[i], desc_clf_prob[i]) / 2), 0,1) for i in range(len(y_test.T))])
# comb_pred = np.where(combined_probs > threshold, upper, lower)
# comb_acc = metrics.f1_score(y_test, comb_pred, average = "micro")


