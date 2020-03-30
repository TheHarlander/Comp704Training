from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
from pandas import DataFrame


data = pd.read_csv("MLDrivingData.csv")
X = data[['ObstacleXPos','ObstacleY','CoinXPos', 'CoinYPos']]
y = data['PlayerPos']

lr = linear_model.LinearRegression(fit_intercept=True)

lr.fit(X, y)

print('Intercept: ', lr.intercept_)
print('Coefficients: ', lr.coef_)

# Save trained model to pkl file
pickle.dump(lr, open('MLRtestTrainData.pkl', 'wb'))
# load the model from disk
loadedModel = pickle.load(open('MLRtestTrainData.pkl', 'rb'))

result = loadedModel.score(X, y)
print(result)


