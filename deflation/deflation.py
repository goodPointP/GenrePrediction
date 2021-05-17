import pandas as pd

inflationData, inflationData['amount'] = pd.read_csv("inflation_data.csv"), pd.read_csv("inflation_data.csv")['amount']/100
# data source: https://www.in2013dollars.com/us/inflation/1920

#%%
data = pd.read_csv("../data/IMDb_movies.csv")
df = data.drop(['original_title', "date_published", "language", "votes", "actors", "director", "writer", "production_company", "metascore", "reviews_from_users", "reviews_from_critics"], axis = 1).dropna()

#%%

def deflate(amount, yearFrom):
    deflatedAmount = amount * (13.3527 / float(inflationData.loc[inflationData['year']==yearFrom]['amount']))
    return deflatedAmount

