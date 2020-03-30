import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn import linear_model
from openpyxl import Workbook
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt


df = pd.read_csv("MLDrivingData.csv")
items = df[['ObstacleXPos','ObstacleY','CoinXPos', 'CoinYPos','ObstacleNear']]

# Create the X and Y arrays
X = items.values
y = df['PlayerPos'].values

# Split the data set in a training set 70/30
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state= 0)

# Fit regression model
model1 = ensemble.GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    subsample=0.5,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=2,
    loss='ls',
    random_state=0
)
model2 = ensemble.GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    subsample=0.5,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=2,
    loss='huber',
    random_state=0
)
print('Doing training.')

votereg = VotingRegressor(estimators=[('ls',model1),('lad', model2)])
votereg = votereg.fit(X,y)
print('Done training.')
pickle.dump(votereg, open('VRtestTrainData.pkl', 'wb'))
loadedModel = pickle.load(open('VRtestTrainData.pkl', 'rb'))

# Get information on how well the model did
mse = mean_absolute_error(y_train, votereg.predict(X_train))
print("Training set mean absolute error: %.4f" % mse)
mse = mean_absolute_error(y_test, votereg.predict(X_test))
print("Test set mean absolute error: %.4f" % mse)
score = loadedModel.score(X_test, y_test)
print('Accurucay of unseen data ' , score)
