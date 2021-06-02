import urllib.request
import re
import requests
from bs4 import BeautifulSoup as BS
import shutil
import pandas as pd
from multiprocessing import Pool

def downloadHTML(ImdbID):
  url = 'https://www.imdb.com/title/'+str(ImdbID)+'/'
  response = urllib.request.urlopen(url)
  html = response.read()
  return html

def downloadMediaviewerHTML(mediaViewerlink):
  response = urllib.request.urlopen(mediaViewerlink)
  return response

def extractImageLink(html):
  soup = BS(html) #on_duplicate_attribute='ignore')
  posterDIV = soup.find_all('div', {'class': 'poster'})
  match = re.search('<a href="(.*)">', str(posterDIV))
  posterDIV = match.group(1)

  mediaViewerlink = 'https://www.imdb.com'+posterDIV
  mediaViewerHTML = downloadMediaviewerHTML(mediaViewerlink)
  soup = BS(mediaViewerHTML)
  imageLink = soup.find_all('div', {'style': 'max-height:493px;max-width:600px;left:calc(50% + 0px)'})
  print(f'Image link: {str(imageLink)}\n')
  match = re.search(' src="(.*?)"', str(imageLink)).group(1)
  # imageLink = soup.find_all('div', {'class': 'poster'})
  #OR
  # imageLink = soup.img['src']
  return match

def downloadAndSaveImage(link, ImdbID):
  print(f"Link: {link}")
  fileName = str(ImdbID) + '.jpg'
  fullPath = 'posters/' + fileName
  response = requests.get(link, stream = True)
  if response.status_code == 200:
    response.raw.decode_content = True
    with open(fullPath, "wb") as f:
      shutil.copyfileobj(response.raw, f)
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
    return image

def getImages(ImdbIDs):
    for ImdbID in ImdbIDs:
        print(ImdbID)
        html = downloadHTML(ImdbID)
        imageLink = extractImageLink(html)
        downloadAndSaveImage(imageLink, ImdbID)
        # image = downloadAndSaveImage(imageLink)
        # saveImage(image, ImdbID)

# test data
ids = ['tt0000003', 'tt0084302', 'tt0383846', 'tt0167260', 'tt2221420', 'tt3155794']
a = getImages(ids)

#%%

data = pd.read_csv("../data/IMDb_movies.csv")
# df = data.drop(['original_title', "date_published", "language", "votes", "actors", "director", "writer", "production_company", "metascore", "reviews_from_users", "reviews_from_critics"], axis = 1).dropna()
IDs = data.imdb_title_id

#%%
# start getting images
getImages(IDs)