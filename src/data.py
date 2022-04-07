import numpy as np
import pandas as pd

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

