import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
import time
from util import *
from readdb import *

class BoostModel:

    def __init__(self, max_depth=3, subsample=0.95, num_round=2000, early_stoppping_rounds=50):
        self.params = {'max_depth': 3, 'eta': 0.01, 'silent': 1, 'alpha': 0.5, 'lambda': 0.5, 'eval_metric': 'rmse', 'subsample': subsample, 'objective':'reg:linear', 'colsample_bytree':0.8}
        self.num_round = num_round
        self.early_stoppping_rounds = early_stoppping_rounds

    def fit(self, train_data, train_label, val_data, val_label):
        dtrain = xgb.DMatrix(train_data, label=train_label)
        deval = xgb.DMatrix(val_data, label=val_label)

        boost_model = xgb.train(self.params, dtrain, num_boost_round=self.num_round, evals=[(dtrain, 'train'), (deval, 'eval')], early_stopping_rounds=self.early_stoppping_rounds, verbose_eval=False)
        print('get best eval : %s, in step %s'%(boost_model.best_score, boost_model.best_iteration))
        self.boost_model = boost_model
        return boost_model

    def predict(self, test_data):

        dtest = xgb.DMatrix(test_data)
        predict_score = self.boost_model.predict(dtest, ntree_limit=self.boost_model.best_ntree_limit)

        return predict_score


def get_train_val_test_data(df, train_pct=0.8, train_start_date=None, train_end_date=None, test_start_date=None, test_end_date=None):
    '''
    参数：train_pct:在训练集中作为验证集的比例
    '''
    train_val_df = df[(df['trade_date'] >= train_start_date) & (df['trade_date'] <= train_end_date)]
    train_val_df = train_val_df.sample(frac=1, random_state=30).reset_index(drop=True)

    #拆分训练集，验证集
    train_df = train_val_df.iloc[0:int(len(train_val_df)*train_pct)]
    val_df = train_val_df.iloc[int(len(train_val_df)*train_pct):]

    test_df = df[df['trade_date'] > train_end_date]
    if test_start_date and test_end_date:
        test_df = test_df[(test_df['trade_date']>= test_start_date ) & (test_df['trade_date'] <= test_end_date)]
    print("[data verify], train, %s to %s" % (min(train_df.trade_date.values), max(train_df.trade_date.values)))
    print("[data verify], val, %s to %s" % (min(val_df.trade_date.values), max(val_df.trade_date.values)))
    print("[data verify], test, %s to %s" % (min(test_df.trade_date.values), max(test_df.trade_date.values)))
    return train_df, val_df, test_df

def predict_factors(train_df, val_df, test_df, factor_names):
    for tcount in range(len(factor_names)):
        factor_name = factor_names[tcount]
        print("predicting %s..." % factor_name)
        boost_model = BoostModel()



#读取原始股票序列
with open(r'F:\WorkplaceSzy\ipcamaster\all_stocks.pkl', 'rb') as f:
    all_stocks = pickle.load(f)

#得到时间频率列表和Barra因子
x_barra = get_barra(all_stocks, '2010-03-01', '2021-01-04')
frequency = get_trading_day_list('2010-03-01', '2021-01-04', frequency='month')
x_barra['date'] = x_barra['date'].apply(lambda x: str(x)[:10])
x_barra = x_barra[list(x_barra['date'].apply(lambda x: (x in frequency)))]

ID = dict(zip(np.unique(x_barra.code).tolist(), np.arange(1, len(np.unique(x_barra.code)) + 1)))
x_barra.code = x_barra.code.apply(lambda x: ID[x])


#计算残差
x_barra_coef = pd.read_csv(r'F:\WorkplaceSzy\XGboost\coef_barra(用于计算残差).csv')

N = len(np.unique(x_barra.code))

def barra_each_stock_return(x_barra, x_barra_coef):
    for t in range(0, 131):
        barra_return = x_barra.iloc[t,2:].values* x_barra_coef.iloc[1:,t+1].values + x_barra_coef.iloc[0:1,t+1].values
    return barra_return

x_barra_test = pd.pivot_table(x_barra.copy(), index = ['code'], values=x_barra.copy().columns.drop(['code']), aggfunc=barra_each_stock_return(x_barra.copy(), x_barra_coef.copy()))

#读取截距数据
residual = pd.read_csv(r'F:\WorkplaceSzy\XGboost\残差.csv')
day_list = residual.columns[1:len(residual.columns)].values.tolist()
residual_copy = residual.copy()
x = (pd.Series(residual_copy.columns.values).apply(lambda x: str(x)[0:len(residual.columns)]))
residual_copy.columns = x.tolist()
residual_copy.iloc[:,[True]+list(x.apply(lambda a: (a in frequency)))[1:]]

