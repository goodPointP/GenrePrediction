import urllib.request
from bs4 import BeautifulSoup as BS

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
  response = requests.get(link)
  image = response.content
  return image

def saveImage(image, ImdbID):
  fileName = str(ImdbID) + '.jpg'
  fullPath = 'posters/' + fileName
  try:
    with open(fullPath, "w") as f:
      f.write(image)
    return True
  except:
    print(f'Did not save the image {fileName} to {fullPath}.')
    return False
 
# test data
ImdbID = 'tt0120737'

html = downloadHTML(ImdbID)
imageLink = extractImageLink(html)
image = downloadImage(imageLink)
saveImage(image, ImdbID)
