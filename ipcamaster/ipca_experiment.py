import numpy as np
import pickle
import progressbar
from ipca import *

from util import *
from readdb import *


with open(r'F:\WorkplaceSzy\ipcamaster\all_stocks.pkl', 'rb') as f:
    all_stocks = pickle.load(f)

periodlist = get_trading_day_list('2010-02-01', '2021-01-10', frequency='month')
m_excess_return = get_profit_depend_timelist(all_stocks, periodlist, 1, 1)
print(m_excess_return)

x_specifications = get_barra(all_stocks, '2010-03-01', '2021-01-04')

N = len(set(all_stocks))

ID = dict(zip(set(all_stocks), np.arange(1, N + 1)))

df_x = x_specifications.set_index('date')
# pd.pivot_table(df_x,index='date',columns='code',values=)
x111 = m_excess_return.copy()
x111['code'] = x111.index.tolist()
x111 = pd.melt(x111, ['code']).set_index(['code', 'variable'])
x111.index.names = ['code', 'date']
# 将x_specifications转为月频, L=10,N=84,T=131
x222 = x_specifications.copy()
x222['date'] = x222['date'].apply(lambda x: str(x)[:10])
x222 = x222[list(x222['date'].apply(lambda x: (x in periodlist)))]
x222 = x222.set_index(['code', 'date'])
# 合并excess_return and characteristics matrix
data = pd.concat([x111, x222], axis=1)

y = data['value']
x = data.iloc[:, 1:11]

from ipca import InstrumentedPCA

regr = InstrumentedPCA(n_factors=5, intercept=False)
regr = regr.fit(X=x, y=y)

# Gmma_alpha = 0 bootstrap test
regr_a = InstrumentedPCA(n_factors=5, intercept=True)
regr_a = regr_a.fit(X=x, y=y)
p_value = regr_a.BS_Walpha(ndraws=1000, n_jobs=1, backend='loky')

# get Gamma and Factors of regr
Gamma, Factors = regr.get_factors(label_ind=True)
x333 = x_specifications.copy()
x333['date'] = x333['date'].apply(lambda x: str(x)[:10])
x333 = x333[list(x333['date'].apply(lambda x: (x in periodlist)))]
Gamma.to_csv(r"C:\Users\Administrator\Desktop\ipca-master - 副本\Gamma.csv")
Factors.to_csv(r"C:\Users\Administrator\Desktop\ipca-master - 副本\Factors.csv")
indices = pd.concat([x333['code'], x333['date']], axis=1)
m_o_c = x.to_numpy()
indices_t = indices.to_numpy()
# 各种predict返回值
# TODO predict_portfolio
m = regr.predict_panel(m_o_c, indices_t, T=131, mean_factor=True)
predict_return = regr.predict(X=x222, mean_factor=False, label_ind=False)
pd.DataFrame(predict_return).to_csv(r"C:\Users\Administrator\Desktop\ipca-master - 副本\predicted_return.csv")
p_test = regr.predict(X=x222, mean_factor=False, label_ind=False)
pd.DataFrame(p_test).to_csv(r"C:\Users\Administrator\Desktop\ipca-master - 副本\IPCA结果第一版\p_test.csv")
# p_test=predicted_return
