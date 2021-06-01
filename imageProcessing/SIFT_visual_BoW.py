import cv2
import numpy as np
import os
import pandas as pd
import csv

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

img_path = '../imageScraping/posters/highres/'
# train = pd.read_csv('../input/train.csv')
# species = train.species.sort_values().unique()

dico = []

sift = cv2.SIFT_create()
# kmeans = MiniBatchKMeans()

#%%
import pickle
with open('../data/dataset_final.pkl', 'rb') as f:
    data = pickle.load(f)
IDs = data.imdb_title_id
genre = data.genre
#%%
def step1():
    for leaf in IDs:
        # print(img_path + str(leaf) + ".jpg")
        img = cv2.imread(img_path + str(leaf) + ".jpg")
        # print(img)
        kp, des = sift.detectAndCompute(img, None)

        for d in des:
            dico.append(d)
            
        return dico
            

def step2():
    k = np.size(genre) * 10

    batch_size = np.size(os.listdir(img_path)) * 3
    kmeans = MiniBatchKMeans(n_clusters=40, batch_size=batch_size, verbose=1).fit(dico)
    return kmeans


def step3():
    kmeans.verbose = False
    k = 40
    histo_list = []

    for leaf in IDs:
        img = cv2.imread(img_path + str(leaf) + ".jpg")
        kp, des = sift.detectAndCompute(img, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

        histo_list.append(histo)
    
    return histo_list

#%%
import pickle
with open('imageFeaturesNoSIFT.pkl', 'wb') as outfile:
    pickle.dump(SIFThistogram, outfile)