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
#df = DataFrame(data, columns=['PlayerX', 'ObstacleXPos', 'CoinXPos'])

X = data[['ObstacleXPos', 'CoinXPos']]
y = data['PlayerPos']

lr = LinearRegression()
lr.fit(X,y)

print('Intercept: ', lr.intercept_)
print('Coefficients: ', lr.coef_)


