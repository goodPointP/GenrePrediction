import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import Stemmer
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example

stemmer = Stemmer.Stemmer('english')

#%% Preprocessing

def remove_punc_stop(corpus):
    stop_words = stopwords.words('english')
    key = str.maketrans('', '', string.punctuation.replace("'", "").replace('"', ''))
    stopwords_dict = set(stop_words)
    texts = []
    for text in corpus:
        text = str(re.sub("(?<!s)'\B|\B'\s*", "", text.lower().replace('"', "'")).translate(key))
        texts.append([word for word in text.split() if word.lower() not in stopwords_dict])    
    return texts

#%% Prefiltered dataset

data = pd.read_csv("data/IMDb_movies.csv")
df = data[['imdb_title_id', 'year', 'genre','description']].dropna() #Including language here
#%%
keywords = ['Drama','Comedy','Horror','Romance','Comedy, Drama','Drama, Romance','Comedy, Romance', 'Comedy, Drama, Romance', 'Comedy, Horror', 'Drama, Horror'] 
values = [700, 700, 2200, 415, 500, 1000, 1000, 700, 570, 229]
list_unshortened_df = [df[df['genre'] == key] for key in keywords]
list_shortened_df = []
for dataframe, value in zip(list_unshortened_df, values):
    list_shortened_df.append(dataframe.sample(frac=1, random_state=420)[:value])
df_sorted = pd.concat(list_shortened_df)


#%% Text stemming
texts = df_sorted['description']
texts = remove_punc_stop(texts)
texts_stemmed = [" ".join(stemmer.stemWords(text)) for text in texts]

#%% TF-IDF

d_tfidf = TfidfVectorizer(analyzer='word', max_features=1000)
d_count = d_tfidf.fit_transform(texts_stemmed)

test = d_count.toarray()

#%%
labelset = set(" ".join(df_sorted['genre']).replace(",", "").split())
train_data = []
size = 2000
for text, labels in zip(texts_stemmed[:size], df_sorted['genre'][:size]):
    categories = {}
    labeldict = {i: False for i in labelset}
    if len(labels) == 1:
        labels = labels.replace(",","")
        labeldict[labels] == True
    else:
        labels = labels.split()
        for label in labels:
            label = label.replace(",","")
            labeldict[label] = True
    categories['cats'] = labeldict
    
    # for lab in enumerate(label):
    #     temp['category'+str(i)] = lab.replace(",","")
    # # if len(label) == 1:
    # #     temp['category0'] = " ".join(label)
    # #     categories['cats'] = temp
    # # if len(label) > 1:
    #     for i, lab in enumerate(labels):
    #         temp['category'+str(i)] = lab.replace(",","")
    #         categories['cats'] = temp
    train_data.append((text, categories))

#%%

nlp = spacy.blank('en')
config = {"threshold": 0.8}
nlp.add_pipe('textcat_multilabel', config=config)
textcat = nlp.get_pipe("textcat_multilabel")

labels = set(" ".join(df_sorted['genre']).replace(",", "").split())
for key in labeldict.keys():
    textcat.add_label(key)

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat_multilabel']

# Only train the textcat pipe
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize()
    for i in range(1):
        losses = {}
        for batch in minibatch(train_data, size=compounding(2., 8., 1.5)):
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(str(text))
                examples.append(Example.from_dict(doc, annotations))
            nlp.update(examples, sgd=optimizer, losses=losses, drop=0.2)
        #print('{0:.3f}'.format(losses['textcat']))

docs = list(nlp.pipe(texts_stemmed[:500]))
preds = textcat.predict(docs)
