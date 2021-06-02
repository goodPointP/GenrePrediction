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
    slices = slice(imagePath, numOfPatches, save=False)
    
    mostCommonColors = []
    for patch in slices:
        # patch.image
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

#%%
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print('Texts:')

    for text in texts:
        rawText = text.description
        break
        # print('\n"{}"'.format(text.description))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
                    # for vertex in text.bounding_poly.vertices])

        # print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    rawText = rawText.replace('\n', '')
    rawText = rawText.replace(' ', '')
    
    # print(rawText)

    return len(rawText)


#%%
import pickle
with open('../data/dataset_final.pkl', 'rb') as f:
    data = pickle.load(f)
IDs = data.imdb_title_id

#%%
import timeit
starttime = timeit.default_timer()

errorList = []
imageFeatures = []
for image in IDs:
    imageName = 'posters/highres/' + image + '.jpg'
    currentImageFeatures = []
    try:
        currentImageFeatures.append(image)
        # currentImageFeatures.append(getImageFeaturesSIFT(imageName))
        # currentImageFeatures.append(getImageFeaturesNumberPeople(imageName))
        currentImageFeatures.append(detect_text(imageName))
        # currentImageFeatures.append(getImageFeaturesTilesMostCommonColors(imageName, 25))
        # currentImageFeatures.append(getImageFeaturesDominantColors(imageName, 5))
        # currentImageFeatures.append(getImageFeaturesHistogram(imageName))
    except:
        errorList.append(imageName)
    imageFeatures.append(currentImageFeatures)
    
print("It took: ", timeit.default_timer() - starttime, 'seconds')

#%%
import pickle
with open('imageFeaturesNoSIFT.pkl', 'wb') as outfile:
    pickle.dump(imageFeaturesOriginal, outfile)
    
#%%
import pickle
with open('../data/imageFeaturesNoSIFT.pkl', 'rb') as f:
    imageFeaturesOriginal = pickle.load(f)
