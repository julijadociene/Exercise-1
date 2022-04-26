import numpy as np
import pandas as pd
from src.data import (
    generate_data,
    true_offline_to_online_relationship,
    generate_y,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from typing import List


def main(size: int, test_size: float, true_parameter_means: List[int]):

    data = generate_data(size)
    y = generate_y(
        true_parameter_means[0],
        true_parameter_means[1],
        true_parameter_means[2],
        true_parameter_means[3],
        size,
        data,
    )

    # train, test datasets
    X = data.copy()
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Offline metric
    reg = LinearRegression().fit(xtrain, ytrain)

    # print the coefficients
    print(reg.intercept_)
    print(reg.coef_)

    # R^2 also accuracy; Accuracy is a measure for the closeness of the measurements to a specific value,
    offline_value = reg.score(X, y)
    print("Offline value " + str(offline_value))

    print("Online value " + str(true_offline_to_online_relationship(offline_value)))


if __name__ == "__main__":

    # reproducibility
    np.random.seed(10)
    main(size=500, test_size=0.2, true_parameter_means=[1.5 for _ in range(4)])
