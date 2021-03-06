import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn import linear_model
from openpyxl import Workbook
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt

# Load csv dataframe, use 5 variables for X
df = pd.read_csv("MLDrivingData.csv")
items = df[['ObstacleXPos','ObstacleY','CoinXPos', 'CoinYPos','ObstacleNear']]

# Create the X and Y arrays.0
X = items.values
# .values is used to convert the data into an array
y = df['PlayerPos'].values

# Split the data set in a training set (70%) and test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state= 0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.1,
    subsample=0.5,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=2,
    loss='ls',
    random_state=0
)
print('Doing training.')
# Fit model
model.fit(X_train, y_train)
print('Done training.')

# Find training error rate
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training set mean absolute error: %.4f" % mse)

# Find the test error rate
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test set mean absolute error: %.4f" % mse)

# Save trained model to pkl file the load
pickle.dump(model, open('GBRtestTrainData.pkl', 'wb'))
loadedModel = pickle.load(open('GBRtestTrainData.pkl', 'rb'))

score = loadedModel.score(X_test, y_test)

print('Accurucay of unseen data ' , score)