import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
import timeit

import os



data_name = 'BTCUSDT'

data_path = f'./data/{data_name}.csv'

df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)
df.sort_index(inplace=True)


os.makedirs(f'./data/{data_name}', exist_ok=True)

train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)
train_data.to_feather(f'./data/{data_name}/train_data.feather')
val_data.to_feather(f'./data/{data_name}/val_data.feather')
test_data.to_feather(f'./data/{data_name}/test_data.feather')
