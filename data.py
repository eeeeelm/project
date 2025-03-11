import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from model import MyModel

stocks = pd.read_pickle('data/stock_code.pkl')
stocks = stocks['stock'].values

train_days = 60
predict_days = 5


for stock in tqdm(stocks):
    x = []
    y= []

    if os.path.exists('part_data/' + str(stock) + '.pkl') == False:
        continue
    data = pd.read_pickle('part_data/' + str(stock) + '.pkl')
    length = len(data)
    data = data.values
    if length < predict_days + train_days:
        continue
    for i in range(length - predict_days - train_days):
        tmp = np.array(data[i: i + train_days], dtype=np.float32)
        x.append(tmp[:, :6])
        tmp = np.array(data[i: i + train_days + predict_days], dtype=np.float32)
        y.append(tmp[:, 6])
    x = np.array(x)
    y = np.array(y)
    pd.to_pickle(x, 'train/' + str(stock) + '.pkl')
    pd.to_pickle(y, 'pred/' + str(stock) + '.pkl')


