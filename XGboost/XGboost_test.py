import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
import time
from util import *
from readdb import *

#读取原始股票序列
with open(r'F:\WorkplaceSzy\ipcamaster\all_stocks.pkl', 'rb') as f:
    all_stocks = pickle.load(f)

#得到时间频率列表和Barra因子
x_barra = get_barra(all_stocks, '2010-03-01', '2021-12-31')
frequency = get_trading_day_list('2010-03-01', '2021-12-31', frequency='month')
x_barra['date'] = x_barra['date'].apply(lambda x: str(x)[:10])
x_barra = x_barra[list(x_barra['date'].apply(lambda x: (x in frequency)))]

#读取截距数据
residual = pd.read_csv(r'F:\WorkplaceSzy\XGboost\残差.csv')
day_list = residual.columns[1:len(residual.columns)].values.tolist()
residual_copy = residual.copy()
x = (pd.Series(residual_copy.columns.values).apply(lambda x: str(x)[0:len(residual.columns)]))
residual_copy.columns = x.tolist()
residual_copy.iloc[:,[True]+list(x.apply(lambda a: (a in frequency)))[1:]]