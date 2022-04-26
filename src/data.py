import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime


def generate_data(size: int) -> pd.DataFrame:
    """Data generating function

    Args:
        size (int): Size of the dataset

    Returns:
        pd.DataFrame: simulated dataset dataframe
    """
    nr_passengers = np.random.choice(a=range(1000, 1000000), size=size)
    airport_size = np.random.exponential(scale=1.0, size=size)
    nr_shops = np.random.choice(a=range(10), size=size)
    nr_cafes = np.random.choice(a=range(10), size=size)
    data = {
        "nr_passengers": nr_passengers,
        "nr_cafes": nr_cafes,
        "nr_shops": nr_shops,
        "airport_size": airport_size,
    }
    df = pd.DataFrame(data)
    return df


def sample_coefficients(
    a_loc: float, b_loc: float, c_loc: float, d_loc: float, minutes_normalised: float
) -> Tuple[float, float]:
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
    a = np.random.normal(loc=a_loc * param_a, scale=0.1)
    b = np.random.normal(loc=b_loc * param_b, scale=0.1)
    c = np.random.normal(loc=c_loc * param_c, scale=0.1)
    d = np.random.normal(loc=d_loc * param_d, scale=0.1)
    return a, b, c, d


def linear_function(
    x_1: np.ndarray,
    x_2: np.ndarray,
    x_3: np.ndarray,
    x_4: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    error: float,
) -> np.ndarray:
    return a * x_1 + b * x_2 + c * x_3 + d * x_4 + error


def rescale(val, in_min=0, in_max=59, out_min=-1, out_max=1):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def true_offline_to_online_relationship(offline_value: float) -> float:
    if offline_value < 0.5:
        return 0.2 * np.random.normal(0, 0.001)
    if offline_value >= 0.5:
        return 1 * offline_value


def generate_y(
    a_loc, b_loc, c_loc, d_loc, size: int, data: pd.DataFrame
) -> pd.DataFrame:
    now = datetime.now()
    minutes = rescale(int(now.strftime("%M")))
    a, b, c, d = sample_coefficients(a_loc, b_loc, c_loc, d_loc, minutes)
    error = np.random.normal(loc=0, scale=np.random.choice([0.1, 0.5, 1]), size=size)
    y = linear_function(
        data["nr_passengers"],
        data["nr_cafes"],
        data["nr_shops"],
        data["airport_size"],
        a,
        b,
        c,
        d,
        error,
    )
    return y
