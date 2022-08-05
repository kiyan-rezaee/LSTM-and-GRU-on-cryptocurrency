import pandas as pd
import numpy as np
from matplotlib import pyplot
from collections import deque
from sklearn import preprocessing
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def compare(current, future):
    if future > current:
        return 1
    else:
        return 0

def preprocess_main_dataframe(df):
    df = df.drop('future', axis=1)
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    sequences = []
    prev_days = deque(maxlen=30)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == 30:
            sequences.append([np.array(prev_days), i[-1]])
    random.shuffle(sequences)
    buys = []
    sells = []
    for seq, target in sequences:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)
        

main_dataframe = pd.DataFrame()

currencies = ['BCH-USD', 'BTC-USD', 'ETH-USD', 'LTC-USD']
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

print(main_dataframe.head())
# print(main_dataframe.isnull().sum())
main_dataframe.fillna(method='ffill', inplace=True)
# print(main_dataframe.isnull().sum())
main_dataframe.fillna(method='bfill', inplace=True)
main_dataframe['future'] = main_dataframe['LTC-USD_close'].shift(-3)
# print(main_dataframe.head())


main_dataframe['target'] = list(map(compare, main_dataframe['LTC-USD_close'], main_dataframe['future']))
times = sorted(main_dataframe.index.values)
last_10pct = sorted(main_dataframe.index.values)[-int(0.1 * len(times))]
main_dataframe_test = main_dataframe[(main_dataframe.index >= last_10pct)]
main_dataframe_train = main_dataframe[(main_dataframe.index < last_10pct)]
main_dataframe['BTC-USD_close'].pct_change()
train_X, train_y = preprocess_main_dataframe(main_dataframe_train)
test_X, test_y = preprocess_main_dataframe(main_dataframe_test)
# print(np.unique(train_y, return_counts=True))

model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1:]), return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

md = model.fit(train_X, train_y, batch_size=100, epochs=100, validation_data=(test_X, test_y))