def downloadHTML(ImdbID):
  return html

def extractImageLink(html):
  return imageLink

def downloadImage(link):
  return image

def saveImage(image):
  try:
    # save image
    return True
  except:
    print('An error occured')
    return False
 

ImdbID = 'tt0120737'

html = downloadHTML(ImdbID)
imageLink = extractImageLink(html)
image = downloadImage(imageLink)
saveImage(image)
