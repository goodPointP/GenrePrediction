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
import Stemmer
from gensim.models import Word2Vec
from sklearn.preprocessing import Normalizer
#%%
with open('data/dataset_final.pkl', 'rb') as f:
    feat_matrix = pickle.load(f)
    
#%% Creating X and y

label_matrix = CountVectorizer().fit_transform(feat_matrix['genre']).toarray()
description_matrix = feat_matrix['description']
#%% Preprocessing

stemmer = Stemmer.Stemmer('english')
texts = remove_punc_stop(description_matrix)
texts_stemmed = [' '.join(stemmer.stemWords(text)) for text in texts]
#%% Train-test split

X_train_temp, X_test_temp, y_train, y_test = split_data(texts_stemmed, label_matrix)

#%%

X_train_word2vec, X_test_word2vec = word2vec_matrix(X_train_temp), word2vec_matrix(X_test_temp)


#%% tf_idf
tfidf = TfidfVectorizer(analyzer='word', max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train_temp).toarray()
X_test_tfidf = tfidf.transform(X_test_temp).toarray()

combined_X_train = np.hstack((X_train_tfidf, X_train_word2vec))
combined_X_test = np.hstack((X_test_tfidf, X_test_word2vec))

normalized_X_train = Normalizer().fit_transform(combined_X_train)
normalized_X_test = Normalizer().transform(combined_X_test)

#%% PCA - choosing the minimum number of components to express 99% variance

pca = PCA().fit(normalized_X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
pca_X_train = pca.transform(normalized_X_train)[:,:np.argmax(cumsum > 0.99)]
pca_X_test = pca.transform(normalized_X_test)[:,:np.argmax(cumsum > 0.99)]


#%% Multi-label-out-of-the-box
X_train = pca_X_train
X_test = pca_X_test

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)
rf_acc = metrics.f1_score(y_test, rf_pred, average = "micro")

mlp = MLPClassifier(max_iter=1000)

mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = metrics.f1_score(y_test, mlp_pred, average = "micro")

mul_clf = ClassifierChain(LogisticRegression(max_iter=500))
mul_clf.fit(X_train, y_train)
mul_pred = mul_clf.predict(X_test)
mul_acc = metrics.f1_score(y_test, mul_pred, average = "micro")
#%%

f = metrics.hamming_loss(y_test, mul_pred)
#%% Meta-modelling approach - Uses PCA 


# desc_clf = RandomForestClassifier()
# desc_clf.fit(X_train, y_train)
# desc_clf_prob = desc_clf.predict_proba(X_test)

# threshold, upper, lower = 0.5, 1, 0
# combined_probs = np.hstack([np.delete((np.add(title_clf_prob[i], desc_clf_prob[i]) / 2), 0,1) for i in range(len(y_test.T))])
# comb_pred = np.where(combined_probs > threshold, upper, lower)
# comb_acc = metrics.f1_score(y_test, comb_pred, average = "micro")


