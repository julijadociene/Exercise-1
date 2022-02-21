import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from src import data

#y_airport_profit = a*nr_passengers+b*nr_cafes+c*nr_shops+d*airport_size+errror
a = 0.3
b = 0.7
c = 0.5
for i in range(3):
    d = np.random.normal(loc=1, scale=.5)
    error = np.random.normal(loc=0, scale=np.random.choice([0.1, 0.5, 1]), size=data.size)
    y = a*data.df['nr_passengers']+b*data.df['nr_cafes']+c*data.df['nr_shops']+d*data.df['airport_size']+error
    plt.scatter(data.df['airport_size'], y)

#model
y = a*data.df['nr_passengers']+b*data.df['nr_cafes']+c*data.df['nr_shops']+d*data.df['airport_size']+error

#train, test datasets
train_size=0.8 #split the data in 80:10:10 for train:valid:test dataset

X = data.df.copy()
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


