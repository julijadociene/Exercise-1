import numpy as np
from datetime import datetime
from typing import Tuple
from src import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linear_function(x_1: np.ndarray, x_2: np.ndarray, x_3: np.ndarray, x_4:np.ndarray, a: float, b: float, c: float, d: float, error: float) -> np.ndarray:
    return a*x_1 + b*x_2 + c*x_3 + d*x_4 + error

def sample_coefficients(a_loc: float, b_loc: float, c_loc: float, d_loc: float, minutes_normalised: float) -> Tuple[float, float]:
    """_summary_

    Args:
        minutes_normalised (float): normalised value of minutes. Say currently it is 13.15
        so we will get 0.15.

    Returns:
        Tuple[float, float]: _description_
    """
    param_a = np.sin(minutes_normalised)
    param_b = np.cos(minutes_normalised)
    param_c = np.sin(minutes_normalised)
    param_d = np.cos(minutes_normalised)
    print(a_loc*param_a, b_loc*param_b, c_loc*param_c, d_loc*param_d)
    a = np.random.normal(loc=a_loc*param_a, scale=0.1)
    b = np.random.normal(loc=b_loc*param_b, scale=0.1)
    c = np.random.normal(loc=c_loc*param_c, scale=0.1)
    d = np.random.normal(loc=d_loc*param_d, scale=0.1)
    return a, b, c, d

def rescale(val, in_min=0, in_max=59, out_min=-1, out_max=1):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

def main() :
    a_loc = 1.5
    b_loc = 1.5
    c_loc = 1.5
    d_loc = 1.5
    now = datetime.now()
    minutes = rescale(int(now.strftime("%M")))
    a, b, c, d = sample_coefficients(a_loc, b_loc, c_loc, d_loc, minutes)
    error = np.random.normal(loc=0, scale=np.random.choice([0.1, 0.5, 1]), size=data.size)
    y = linear_function(data.df['nr_passengers'], data.df['nr_cafes'], data.df['nr_shops'], data.df['airport_size'], a, b, c, d, error)

if __name__ == "__main__":
    main()

def true_offline_to_online_relationship(offline_value: float) -> float:
    if offline_value < 0.5:
        return 0.2*np.random.normal(0, 0.001)
    if offline_value >= 0.5:
        return 1*offline_value

#train, test datasets
train_size=0.8 #split the data in 80:10:10 for train:valid:test dataset

X = data.df.copy()
#???????
y = y
 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#Offline metrika
reg = LinearRegression().fit(xtrain, ytrain)

#print the coefficients
print(reg.intercept_)
print(reg.coef_)

#predictions based on model
ypred = reg.predict(xtest)

#R^2 also accuracy; Accuracy is a measure for the closeness of the measurements to a specific value,
offline_value = reg.score(X, y)
print(offline_value)
 
print(true_offline_to_online_relationship(offline_value))