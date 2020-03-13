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

df = pd.read_csv("MLDrivingData.csv")

items_df = pd.get_dummies(df, columns=['ObstacleXPos', 'CoinXPos'])

# Create the X and Y arrays
X = items_df.values
y = df['PlayerPos'].values

# Split the data set in a trining set (70%) and test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state= 0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    max_depth=16,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
print('Doing training.')
model.fit(X_train, y_train)
print('Done training.')
# Find training error rate
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training set mean absolute error: %.4f" % mse)

# Find the test error rate
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test set mean absolute error: %.4f" % mse)


# Save trained model to pkl file
pickle.dump(model, open('testTrainData.pkl', 'wb'))


# load the model from disk
loadedModel = pickle.load(open('testTrainData.pkl', 'rb'))

result = loadedModel.score(X_test, y_test)
#outcome = model.predict(X=X_test)

print('Accurucay of unseen data ' , result)
#print('outcome : ' , outcome)

#plt.figure(figsize=(12,6))
#plt.title('Gradient boosting model')
#plt.scatter(X_train, y_train)
#plt.plot(X_train, model.predict(X_train), color='black')

#plt.show()