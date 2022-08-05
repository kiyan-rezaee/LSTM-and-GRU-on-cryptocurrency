import pandas as pd
import numpy as np
from matplotlib import pyplot
from collections import deque
from sklearn import preprocessing
import random

main_dataframe = pd.DataFrame()

currencies = ['ETH-USD', 'LTC-USD']
for c in currencies:
    dataset = 'data/' + c + '.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={'close': c + '_close', 'volume': c + '_volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[[c + '_close', c + '_volume']]
    if len(main_dataframe) == 0:
        main_dataframe = df
    else:
        main_dataframe = main_dataframe.join(df)

# print(main_dataframe.head())
