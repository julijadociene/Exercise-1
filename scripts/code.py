import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# reproducibility
np.random.seed(10)

# number of observations
size = 500

# feature setup
nr_passengers = np.random.choice(a=range(1000, 1000000), size=size)
airport_size = np.random.exponential(scale=1.0, size=size)
nr_shops = np.random.choice(a=range(10), size=size)
nr_cafes = np.random.choice(a=range(10), size=size)
#y_airport_profit = np.random.choice(a=range(100000, 10000000), size=size)

#data
data = {'nr_passengers': nr_passengers,
        'nr_cafes' : nr_cafes,
        'nr_shops' : nr_shops,
        'airport_size' : airport_size}
df = pd.DataFrame(data)
print(df)

#y_airport_profit = a*nr_passengers+b*nr_cafes+c*nr_shops+d*airport_size+errror
a = 0.3
b = 0.7
c = 0.5
for i in range(3):
    d = np.random.normal(loc=1, scale=.5)
    error = np.random.normal(loc=0, scale=np.random.choice([0.1, 0.5, 1]), size=size)
    y = a*df['nr_passengers']+b*df['nr_cafes']+c*df['nr_shops']+d*df['airport_size']+error
    plt.scatter(df['airport_size'], y)

#model
y = a*df['nr_passengers']+b*df['nr_cafes']+c*df['nr_shops']+d*df['airport_size']+error

#train, test datasets
train_size=0.8 #split the data in 80:10:10 for train:valid:test dataset

X = df.copy()
y = y
 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#Online metrika?

#Offline metrika
reg = LinearRegression().fit(xtrain, ytrain)

#print the coefficients
print(reg.intercept_)
print(reg.coef_)

#predictions based on model
ypred = reg.predict(xtest)

#R^2 also accuracy?
#Accuracy is a measure for the closeness of the measurements to a specific value,
print(reg.score(X, y))
print(reg.score(xtest, ytest, sample_weight=None)) 


