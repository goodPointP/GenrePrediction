from selenium import webdriver
from PIL import Image
from io import BytesIO
import re
import urllib.request
from bs4 import BeautifulSoup as BS
from selenium.webdriver.chrome.options import Options
import pandas as pd


def getPoster(ImdbID):
    url = 'https://www.imdb.com/title/'+str(ImdbID)+'/'
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BS(html) #on_duplicate_attribute='ignore')
    posterDIV = soup.find_all('div', {'class': 'poster'})
    match = re.search('<a href="(.*)">', str(posterDIV))
    posterDIV = match.group(1)

    mediaViewerlink = 'https://www.imdb.com'+posterDIV
    
    driver.get(mediaViewerlink)
    elements = driver.find_elements_by_tag_name('img')
    element = elements[0]
    i = 1
    
    while element.is_displayed() == False:    
        element = elements[i]
        i += 1
        
    clickElement = driver.find_element_by_xpath("//*[@aria-label='Close']")
    clickElement.click()
    
    location = element.location
    size = element.size
    png = driver.get_screenshot_as_png()
    
    im = Image.open(BytesIO(png))
    
    left = location['x']
    top = location['y']
    right = location['x'] + size['width']
    bottom = location['y'] + size['height']
    
    # driver.quit()
    
    im = im.crop((left, top, right, bottom))
    im = im.convert("RGB")
    
    fileName = str(ImdbID) + '.jpg'
    fullPath = 'posters/' + fileName
    im.save(fullPath)
    
#%%
DRIVER_PATH = 'chromedriver.exe'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
ImdbIDs = ['tt0000003', 'tt0084302', 'tt0383846', 'tt0167260', 'tt2221420', 'tt3155794']
for ImdbID in ImdbIDs:
    getPoster(ImdbID)

#%%
data = pd.read_csv("../data/IMDb_movies.csv")
df = data.drop(['original_title', "date_published", "language", "votes", "actors", "director", "writer", "production_company", "metascore", "reviews_from_users", "reviews_from_critics"], axis = 1).dropna()
IDs = df.imdb_title_id

#%%
DRIVER_PATH = 'chromedriver.exe'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)

for ImdbID in IDs:
    getPoster(ImdbID)