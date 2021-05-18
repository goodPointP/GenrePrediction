### import.py
import pandas as pd
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import string
import re
from nltk.corpus import stopwords
from collections import Counter

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punc(corpus):
    texts = []
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    for text in corpus:
        texts.append(str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key)))
    return texts

def remove_stop(corpus):
    texts = []
    stop_words = stopwords.words('english')
    stopwords_dict = set(stop_words)
    for text in corpus:
        texts.append(' '.join([word for word in text.split() if word.lower() not in stopwords_dict]))
    return texts

def remove_punc_stop(corpus):
    stop_words = stopwords.words('english')
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    stopwords_dict = set(stop_words)
    texts = []
    for text in corpus:
        text = str(re.sub("(?<!s)'\B|\B'\s*", "", text.replace('"', "'")).translate(key))
        texts.append(' '.join([word for word in text.split() if word.lower() not in stopwords_dict]))    
    return texts

def remove_spaces(text):
    return text.replace(' ','')

def tokenizer(text):
    word_tokenizer = RegexpTokenizer(r"\w+")
    w = word_tokenizer.tokenize(text)
    s = sent_tokenize(text)
    return w,s

def normalize(data):
    normalized = StandardScaler()
    norm_data = normalized.fit_transform(data)
    return norm_data

def split_data(data, truth):
    X_train, X_test, y_train, y_test = train_test_split(
    data, truth, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test
