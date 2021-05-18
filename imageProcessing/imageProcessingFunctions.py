import cv2
import pytesseract
import face_recognition # 1. install https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16 2. pip install dlib 3. pip install face_recognition

# get SIFT features
def getImageFeaturesSIFT(imagePath):
    return SIFTfeatures

# get number of people on the poster
def getImageFeaturesNumberPeople(imagePath):
    # image = face_recognition.load_image_file(imagePath)
    # face_locations = face_recognition.face_locations(image)
    # return len(face_locations)
    cascPath = 'supplements/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    nPeople = len(faces)
    return nPeople

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

# is it in color or not?
def getImageFeaturesColor(imagePath):
    return isColored

# average color by tiles
def getImageFeaturesTilesAverageColors(imagePath):
    return tilesAverageColors

# color histogram
def getImageFeaturesHistogram(imagePath):
    return histogram

# testing
imagePath = '../imageScraping/posters/tt0031381.jpg'
a = getImageFeaturesNumberPeople(imagePath)
