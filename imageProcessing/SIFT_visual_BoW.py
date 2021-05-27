from sklearn.decomposition import PCA
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import itertools
import pandas as pd

#%%
import pickle
with open('../data/dataset_final.pkl', 'rb') as f:
    data = pickle.load(f)
IDs = data.imdb_title_id

#%%
# training_paths = ['tt9825006.jpg', 'tt9779516.jpg', 'tt9426210.jpg']
sift = cv2.SIFT_create()
siftResults = []

for ID in IDs:
    image = cv2.imread('posters/highres/' + ID + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, dsc = sift.detectAndCompute(gray, None)
    tempResults = list(zip([kps.response for kps in kp], dsc))
    tempResultsSorted = sorted (tempResults, key = lambda x: x[0])[-10:]
    
    sortedSIFTDescs = []
    for row in tempResultsSorted:
        sortedSIFTDescs.append(row[1])
    
    merged = np.array(list(itertools.chain.from_iterable(sortedSIFTDescs)))
    siftResults.append(merged)

siftResults = np.array(siftResults)

#%%
import pickle
with open('imageFeaturesOnlySIFT.pkl', 'wb') as outfile:
    pickle.dump(siftResults, outfile)
