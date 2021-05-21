import cv2
import pytesseract # https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20210506.exe tesseract.exe needs to be in C:\Program Files\Tesseract-OCR\
import face_recognition # 1. install https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16 2. pip install dlib 3. pip install face_recognition
from colorthief import ColorThief #pip install colorthief
from image_slicer import slice # pip install image-slicer
import numpy as np

# get SIFT features
def getImageFeaturesSIFT(imagePath):
    image = cv2.imread(imagePath)
    sift = cv2.SIFT_create()
    kp, SIFTfeatures = sift.detectAndCompute(image, None)
    return SIFTfeatures

# get number of people on the poster
def getImageFeaturesNumberPeople(imagePath):
    image = face_recognition.load_image_file(imagePath)
    face_locations = face_recognition.face_locations(image)
    return len(face_locations)
    # cascPath = 'supplements/haarcascade_frontalface_default.xml'
    # faceCascade = cv2.CascadeClassifier(cascPath)
    # image = cv2.imread(imagePath)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags = cv2.CASCADE_SCALE_IMAGE
    # )
    # nPeople = len(faces)
    # return nPeople

# get number alphanumeric characters on the poster
def getImageFeaturesNumberCharacters(imagePath):
    image = cv2.imread(imagePath)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    chars = pytesseract.image_to_string(image)
    
    chars = chars.replace('\n', '')
    chars = chars.replace('x0c', '')
    chars = chars.replace(' ', '')
    
    nChars = len(chars)
    return nChars

# average color by tiles
def getImageFeaturesTilesMostCommonColors(imagePath, numOfPatches = 25):
    image = cv2.imread(imagePath)
    slices = slice(imagePath, numOfPatches)
    
    mostCommonColors = []
    for patch in slices:
        patch.image
        patchArray = np.array(patch.image)
        unique, counts = np.unique(patchArray.reshape(-1, 3), axis=0, return_counts=True)
        R, G, B = unique[np.argmax(counts)]
        mostCommonColors.append((R, G, B))
        
    return mostCommonColors

def getImageFeaturesDominantColors(imagePath, numOfColors = 5):
    colorThief = ColorThief(imagePath)
    palette = colorThief.get_palette(color_count=numOfColors)
    return palette

# color histogram
def getImageFeaturesHistogram(imagePath):
    img = cv2.imread(imagePath) #mode could also be HSV
    histogram = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    features = cv2.normalize(histogram, histogram).flatten()
    return features

# testing

highResPath = '../imageScraping/posters/highres/'
lowResPath = '../imageScraping/posters/lowres/'

imageName = 'tt8290698.jpg'
imagePathHighRes, imagePathLowRes  = highResPath + imageName, lowResPath + imageName
# a = getImageFeaturesNumberPeople(imagePath)
# a = getImageFeaturesSIFT(imagePath)
# resultHighRes = getImageFeaturesHistogram(imagePathHighRes)
# resultLowRes = getImageFeaturesHistogram(imagePathLowRes)

# resultHighRes = getImageFeaturesDominantColors(imagePathHighRes, 5)
# resultLowRes = getImageFeaturesDominantColors(imagePathLowRes, 5)
a = getImageFeaturesTilesMostCommonColors(imagePathLowRes)

