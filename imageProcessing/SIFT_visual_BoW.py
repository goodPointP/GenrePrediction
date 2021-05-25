import cv2
from sklearn.cluster import KMeans, build_histogram

# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

kmeans = KMeans(n_clusters = 800)
kmeans.fit(descriptor_list)

preprocessed_image = []
for image in images:
      image = gray(image)
      keypoint, descriptor = features(image, extractor)
      if (descriptor is not None):
          histogram = build_histogram(descriptor, kmeans)
          preprocessed_image.append(histogram)