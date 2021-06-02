# GenrePrediction
Cog Sci 2 Project

- DONE scrape images from IMDB
- DONE country mapping
- DONE how to represent description and title (word2vec ...)?
- DONE genre mapping
- DONE numeric representation of features

# IMDB database features:
- rating
- title
- year
- duration
- country
- director(s) (?)
- writer(s) (?)
- IMDB descriptions
- budget
- USA income
- worldwide income

# Other relevant features:
- posters:
   - visual bag of words with SIFT
   - number of people
   - number of charaters (using OCR)
   - ~~b&w or not~~
   - dominant color
   - divide images to separate tiles, anayze each tile's average color and use that as features
   - calculate color histograms (as seen in: https://medium.com/de-bijenkorf-techblog/image-vector-representations-an-overview-of-ways-to-search-visually-similar-images-3f5729e72d07)
