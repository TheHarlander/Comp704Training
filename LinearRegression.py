from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#now got 3 columns in excel sheet
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv("MLDrivingData.csv")

data.shape

print(data.describe())

X = data['CoinXPos'].values.reshape(-1,1)
y = data['PlayerPos'].values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Save trained model to pkl file
pickle.dump(regressor, open('LRtestTrainData.pkl', 'wb'))

# load the model from disk
loadedModel = pickle.load(open('LRtestTrainData.pkl', 'rb'))

result = loadedModel.score(X_test, y_test)
print(result)


'''
data.plot(x='CoinXPos', y ='PlayerPos', style='o')
plt.title('CoinXPos X vs Player X')
plt.xlabel('CoinXPos X')
plt.ylabel('Player X')
plt.show()
'''



'''
data.CoinXPos.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of coin pos', size=24)
plt.xlabel('Coin X Pos', size=18)
plt.ylabel('Frequency', size=18)


data.PlayerPos.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of player pos', size=24)
plt.xlabel('player X Pos', size=18)
plt.ylabel('Frequency', size=18)
plt.show()
'''


# scatter graph plot
'''
X = data.iloc[:,0].values.reshape(-1,1)
Y = data.iloc[:,1].values.reshape(-1,1)
model = LinearRegression()
model.fit(X,Y)
Y_pred = model.predict(X)

plt.scatter(X,Y)
plt.plot(X,Y_pred, color='red')
plt.show()
'''


'''
df = pd.read_csv("MLDrivingData.csv")
obstacle_df = pd.get_dummies(df, columns=['ObstacleXPos'])
player_df = pd.get_dummies(df, columns=['PlayerPos'])

predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=obstacle_df, y=player_df)

X_TEST = [[500,600,700]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))
'''

'''
df = pd.read_csv("MLDrivingData.csv")

target = pd.read_csv("MLDrivingData.csv", usecols=["PlayerPos"])

X = df
#y = target["PlayerPos"]

y = df['PlayerPos'].values

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
print(predictions)
score = lm.score(X,y)
print(score)
'''