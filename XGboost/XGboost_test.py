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
    train_val_df = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
    train_val_df = train_val_df.sample(frac=1, random_state=30).reset_index(drop=True)

    #拆分训练集，验证集
    train_df = train_val_df.iloc[0:int(len(train_val_df)*train_pct)]
    val_df = train_val_df.iloc[int(len(train_val_df)*train_pct):]

    test_df = df[df['date'] > train_end_date]
    if test_start_date and test_end_date:
        test_df = test_df[(test_df['date'] >= test_start_date) & (test_df['date'] <= test_end_date)]
    print("[data verify], train, %s to %s" % (np.min(train_df.date.values), np.max(train_df.date.values)))
    print("[data verify], val, %s to %s" % (np.min(val_df.date.values), np.max(val_df.date.values)))
    print("[data verify], test, %s to %s" % (np.min(test_df.date.values), np.max(test_df.date.values)))
    return train_df, val_df, test_df

def format_feature_label(data_df, y_marker, factor_names):
    '''

    Parameters
    ----------
    data_df:原始输入数据
    y_marker： ‘residual’截距column的名字
    factor_names：除去residual之外的columns 名字

    Returns
    -------
    feature：（type:np.array）
    label_lists:

    '''
    df = data_df.copy()
    if 'date' in df.columns:
        del df['date']
    if 'code' in df.columns:
        del df['code']
    end_tag_length = len(y_marker)
    valid_feature_col = [x for x in df.columns if x[-end_tag_length:] != y_marker]
    feature = np.array(df[valid_feature_col])

    label_lists = []
    for factor_name in factor_names:
        label = np.array(df[y_marker])
        label_lists.append(label)
    return feature, label_lists



#读取原始股票序列
with open(r'F:\WorkplaceSzy\ipcamaster\all_stocks.pkl', 'rb') as f:
    all_stocks = pickle.load(f)

x_barra_coef = pd.read_csv(r'F:\WorkplaceSzy\XGboost\coef_barra(用于计算残差).csv')
#得到时间频率列表和Barra因子
x_barra = get_barra(all_stocks, '2010-03-01', '2021-01-04')
frequency = get_trading_day_list('2010-03-01', '2021-01-04', frequency='month')
x_barra['date'] = x_barra['date'].apply(lambda x: str(x)[:10])
x_barra = x_barra[list(x_barra['date'].apply(lambda x: (x in frequency)))]

ID = dict(zip(np.unique(x_barra.code).tolist(), np.arange(1, len(np.unique(x_barra.code)) + 1)))
x_barra.code = x_barra.code.apply(lambda x: ID[x])

x_barra.to_csv(r'F:\WorkplaceSzy\XGboost\barra值.csv')
#standardize Barra
valuation_data = get_valuation(all_stocks, start_date='2010-03-01', end_date='2021-01-04', count=None)
market_cap = pd.DataFrame(valuation_data, columns=['date','market_cap'])
market_cap['date'] = market_cap['date'].apply(lambda x: str(x))
market_cap = market_cap[list(market_cap['date'].apply(lambda x: (x in frequency)))]

barra_mean = np.array(x_barra.iloc[:, 2:]) * np.array(market_cap.iloc[:, 1:2])#Barra因子乘market_cap
bs = [] #barra mean for each barra factor
for b in range(0, np.shape(barra_mean)[1]):
    bi = np.array(barra_mean[:, b]).sum()
    b_cap = bi / (np.array(market_cap['market_cap']).sum())
    bs.append(b_cap)

n = np.shape(np.array(x_barra.iloc[:, 2:]))[0]
m = np.shape(np.array(x_barra.iloc[:, 2:]))[1]
barra_standardized = []
for i in range(0, m):
    diff = np.array(x_barra.iloc[:, 2+i:3+i]) - np.array(bs[i])
    sqr_diff_sum = np.sum(np.square(diff))
    std = np.sqrt(sqr_diff_sum/n)
    bfs = []
    for j in range(0, n):
        nominator = np.array(x_barra.iloc[:, 2+i:3+i])[j, :] - np.array(bs[i])
        b_stand = nominator/std
        bfs.append(b_stand)
    barra_standardized.append(bfs)

d_test = np.array(barra_standardized)
barra_s_v = d_test.reshape(11004, 10)
barra_s_frame = pd.DataFrame(barra_s_v, columns=x_barra.columns[2:].values)
code_frame = pd.DataFrame(x_barra.iloc[:, 0:2])

b_s_test = np.append(np.array(code_frame), (np.array(barra_s_frame)), axis=1)

b_s_frame = pd.DataFrame(b_s_test, columns=x_barra.columns.values)
b_s_frame.to_csv(r'F:\WorkplaceSzy\XGboost\特殊标准化后的Barra.csv')
#计算残差

def aggfunc_return(barra, x_barra_coef):
    N = len(np.unique(barra.code))
    barra_return = []
    for n in range(1, N+1):
        #i_stock_barras = []
        for t in range(0, 131):
            i = (np.array(barra.iloc[0+(131*(n-1)):(131 * n), 2:]) * np.array(x_barra_coef.iloc[1:11, 1:]).T + np.array(x_barra_coef.iloc[0:1, 1:]).T)[t].sum()
            barra_return.append(i)
    return barra_return

cal_barra = aggfunc_return(b_s_frame, x_barra_coef.copy())
cal_barra = np.array(cal_barra)
cal_barra_test = np.append(np.array(x_barra['date'][0:11004]).reshape(11004, 1), np.array(cal_barra).reshape(11004,1), axis=1)
barra_result = pd.DataFrame(cal_barra_test, columns={'date', 'barra price'})
barra_result.to_csv(r'F:\WorkplaceSzy\XGboost\Barra标准化之后求到收盘价.csv')

close_price_base = get_price(all_stocks, '2010-03-01', '2021-01-04')
close_price_base['date'] = close_price_base['date'].apply(lambda x: str(x)[:10])
close_price_base = close_price_base[list(close_price_base['date'].apply(lambda x: (x in frequency)))]

c = np.array(close_price_base['close'])
close_price_ratio = []
for t in range(0, np.shape(np.array(close_price_base['close']))[0]-1):
    ci = c[t+1]/c[t]
    close_price_ratio.append(ci)

c_test = np.append(np.array(close_price_base['date'][0:11003]).reshape(11003, 1), np.array(close_price_ratio).reshape(11003, 1), axis=1)
close_price_ratio = pd.DataFrame(c_test, columns={'date', 'barra price'})
close_price_ratio.to_csv(r'F:\WorkplaceSzy\XGboost\收盘价.csv')
#得到残差
a_residual = np.array(close_price_ratio.iloc[:, 1:2] - barra_result.iloc[:11003, 1:2])
a_residual_t = np.append(np.array(close_price_base['date'][0:11003]).reshape(11003, 1), np.array(a_residual).reshape(11003, 1), axis=1)
barra_residual = pd.DataFrame(a_residual_t)
barra_residual_r = barra_residual.rename(columns={barra_residual.columns[0]: 'date', barra_residual.columns[1]: 'residual'})
barra_residual_r.to_csv(r'F:\WorkplaceSzy\XGboost\residual.csv')
#def barra_each_stock_return(x_barra, x_barra_coef):
    #for t in range(0, 131):
       # barra_return = x_barra.iloc[t,2:].values* x_barra_coef.iloc[1:,t+1].values + x_barra_coef.iloc[0:1,t+1].values
    #return barra_return

#x_barra_test = pd.pivot_table(x_barra.copy(), index = ['date'], columns =['code'], values=x_barra.copy().columns.drop(['code','date'])) #aggfunc=barra_each_stock_return(x_barra.copy(), x_barra_coef.copy()))

#读取截距得到训练集，测试集

boost_pane_array = np.append(np.array(b_s_frame.iloc[:11003,:]), np.array(barra_residual_r.iloc[:,1]).reshape(11003,1), axis=1)
boost_panel = pd.DataFrame(boost_pane_array, columns=b_s_frame.columns.values.tolist() + ['residual'])

train_df, val_df, test_df = get_train_val_test_data(boost_panel, train_pct=0.8, train_start_date='2010-03-01', train_end_date='2018-11-01', test_start_date='2018-12-03', test_end_date='2020-12-01')


factor_names_list = boost_panel.columns[2:-1].values.tolist()
def predict_factors(train_df_e, val_df_e, test_df_e, factor_names):
    '''
    输入参数：
        train_df, val_df, test_df: 训练集、验证集、测试集数据
    输出：
        预测值df，columns为 trade_date, 因子真实值（这两项是取自test_df中），因子的预测值

    '''
    train_feature, train_labels = format_feature_label(train_df_e, "residual", factor_names)
    val_feature, val_labels = format_feature_label(val_df_e, "residual", factor_names)
    test_feature, test_labels = format_feature_label(test_df_e, "residual", factor_names)

    test_date_values = list(test_df_e.date.values)
    result_dict = {"date": test_date_values}
    # xgboost模型训练，得到因子值输出
    for fcount in range(len(factor_names)):
        factor_name = factor_names[fcount]
        print("predicting %s..." % factor_name)
        boost_model = BoostModel()
        train_start = time.time()
        boost_model.fit(train_feature, train_labels[fcount], val_feature, val_labels[fcount])
        train_stop = time.time()
        print("train time cost:%s" % (train_stop - train_start))
        predict_score = boost_model.predict(test_feature)
        predict_stop = time.time()
        print("predict time cost:%s" % (predict_stop - train_stop))
        predict_score = list(predict_score)
        result_dict[factor_name + "_predict"] = predict_score
        fcount += 1
    return pd.DataFrame(result_dict)


boost_barra_result = predict_factors(train_df, val_df, test_df, factor_names_list)
boost_barra_result.to_csv(r"F:\WorkplaceSzy\XGboost\boost_barra_result.csv")



#读取截距数据
residual = pd.read_csv(r'F:\WorkplaceSzy\XGboost\残差.csv')
day_list = residual.columns[1:len(residual.columns)].values.tolist()
residual_copy = residual.copy()
x = (pd.Series(residual_copy.columns.values).apply(lambda x: str(x)[0:len(residual.columns)]))
residual_copy.columns = x.tolist()
residual_copy.iloc[:,[True]+list(x.apply(lambda a: (a in frequency)))[1:]]

