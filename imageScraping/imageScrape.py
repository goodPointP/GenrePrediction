import urllib.request
import requests
from bs4 import BeautifulSoup as BS
import shutil

def downloadHTML(ImdbID):
  url = 'https://www.imdb.com/title/'+str(ImdbID)+'/'
  response = urllib.request.urlopen(url)
  html = response.read()
  return html

def extractImageLink(html):
  soup = BS(html) #on_duplicate_attribute='ignore')
  imageLink = soup.find('img')['src']
  #OR
  imageLink = soup.img['src']
  return imageLink

def downloadImage(link):
  response = requests.get(link, stream = True)
  if response.status_code == 200:
    response.raw.decode_content = True
    return response

def saveImage(image, ImdbID):
  fileName = str(ImdbID) + '.jpg'
  fullPath = 'posters/' + fileName
  try:
    with open(fullPath, "wb") as f:
      shutil.copyfileobj(image.raw, f)
    return True
  except:
    print(f'Did not save the image {fileName} to {fullPath}.')
    return False

def getImages(ImdbIDs):
    for ImdbID in ImdbIDs:
        html = downloadHTML(ImdbID)
        imageLink = extractImageLink(html)
        image = downloadImage(imageLink)
        saveImage(image, ImdbID)

# test data
# ImdbID = 'tt0120737'

ids = ['tt0084302', 'tt0000003', 'tt0383846', 'tt0167260', 'tt2221420', 'tt3155794']
getImages(ids)