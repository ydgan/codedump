import warnings
warnings.filterwarnings('ignore')

import akshare as ak
import baostock as bs

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

import time
import os, pdb
import datetime

def get_traday(freq='M'):
    is_traday = ak.tool_trade_date_hist_sina()
    is_traday['trade_date'] = pd.to_datetime(is_traday['trade_date'])

    resample = is_traday.set_index('trade_date')
    resample = resample.resample(freq).last()

    is_traday['trade_date'] = is_traday['trade_date'].apply(
        lambda x:x.strftime('%Y-%m-%d')
    )
    return is_traday, [x.strftime('%Y-%m-%d') for x in resample.index]

def get_stock_mat(zz500_list, sdate, edate):
    md_path = '/Users/ydgan/Documents/data/stock'
    sub_path = '/Users/ydgan/Documents/index_enhancement/sub_data'
    stock_mat = pd.DataFrame()
    for code in zz500_list['code']:
        try:
            mdata = pd.read_csv(os.path.join(md_path, code[3:]+'.csv'), index_col=0)
        except:
            sub_data = os.path.join(sub_path, code[3:]+'.csv')
            if os.path.exists(sub_data):
                mdata = pd.read_csv(sub_data, index_col=0)
            else:
                print(code)
                login = bs.login()
                rs = bs.query_history_k_data_plus(code,
                                                "date,open,high,low,close,volume,amount",
                                                start_date='1990-01-01', end_date='2023-09-01',
                                                frequency="d", adjustflag="1")
                mdata = rs.get_data()
                logout = bs.logout()
                if not mdata.empty:
                    mdata.to_csv(sub_data)

        if not mdata.empty:
            mdata = mdata[(mdata['date'] >= sdate) & (mdata['date'] < edate)]
            mdata = mdata.rename(columns={'close':code[3:]})
            if stock_mat.empty:
                stock_mat = mdata[['date', code[3:]]]
            else:
                stock_mat = pd.merge(stock_mat, mdata[['date', code[3:]]], on='date', how='outer')

    stock_mat.set_index('date', inplace=True)
    stock_mat = stock_mat.astype(float)
    return stock_mat

def cal_cov(df):
    return np.array(df.cov())

def cal_half_cov(df):
    xshape = df.shape[0]
    cov = cal_cov(df.iloc[0:int(xshape/4)]) * (1/10)
    for i in range(1, 4):
        sub_log_return = df.iloc[int(xshape/4)*i:int(xshape/4)*(i+1), :]
        sub_cov = cal_cov(sub_log_return)
        if i == 1:
            cov += sub_cov*(2/10)
        elif i == 2:
            cov += sub_cov*(3/10)
        else:
            cov += sub_cov*(4/10)
    return cov

def lasso_risk_parity(weight, cov):
    trc = cal_risk_contribution(weight, cov)
    target_error = sum([sum((i-trc)**2) for i in trc] + np.fabs(weight)*5)
    return target_error

def pca_risk_parity(weight, cov):
    trc = cal_risk_contribution_pca(weight, cov)
    target_error = sum([sum((i-trc)**2) for i in trc])
    return target_error

def cal_risk_contribution_pca(weight, cov):
    sigma = np.sqrt(np.dot(weight, np.dot(cov, weight)))
    u, d, v = np.linalg.svd(cov)
    a = np.dot(v, weight)
    b = np.dot(v, np.dot(cov, weight))
    trc = np.multiply(a,b)
    trc = trc / sigma
    return trc
    
def naive_risk_parity(weight, cov):
    trc = cal_risk_contribution(weight, cov)
    target_error = sum([sum((i-trc)**2) for i in trc])
    return target_error

def cal_risk_contribution(weight, cov):
    sigma = np.sqrt(np.dot(weight, np.dot(cov, weight)))
    mrc = np.dot(cov, weight) / sigma
    trc = weight * mrc
    return trc

def portfolio_weight(cov, risk_budget_objective):
    num = cov.shape[0]
    x0 = np.array([1/num]*num)
    bounds = tuple((0, 1) for _ in range(num))
    cons = ({'type':'eq', 'fun':lambda x:sum(x)-1})

    opt = minimize(
        risk_budget_objective, x0,
        args=(cov),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter':100}
    )
    return opt.x

def get_weight_matrix(df, method=None, half=False):
    if half:
        cov = cal_half_cov(df)
    else:
        cov = cal_cov(df)

    if method == 'pca':
        weight = portfolio_weight(cov, pca_risk_parity)
    elif method == 'lasso':
        weight = portfolio_weight(cov, lasso_risk_parity)
    else:
        weight = portfolio_weight(cov, naive_risk_parity)
    return weight.tolist()

def cal_weight(wt, log_return, date):
    print('%s %s'%(date, wt))
    weight_path = '/Users/ydgan/Documents/index_enhancement/base_hold'
    year = date.split('-')[0]
    month = date.split('-')[1]
    save_path = os.path.join(weight_path, wt, year, month)
    if not os.path.exists(save_path):   
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, date+'.csv')):
        return

    if wt == 'equal_weight':
        twt = 1 / log_return.shape[1]
        weight_list = [twt]*log_return.shape[1]
    elif wt == 'risk_parity':
        weight_list = get_weight_matrix(log_return)
    elif wt == 'pca_risk_parity':
        weight_list = get_weight_matrix(log_return, method='pca')
    elif wt == 'half_pca_risk_parity':
        weight_list = get_weight_matrix(log_return, method='pca', half=True)
    elif wt == 'half_risk_parity':
        weight_list = get_weight_matrix(log_return, half=True)
    elif wt == 'lasso_risk_parity':
        weight_list = get_weight_matrix(log_return, method='lasso')
    
    weight = pd.DataFrame({'weight':weight_list}, index=log_return.columns.tolist())
    weight.to_csv(os.path.join(save_path, date+'.csv'))

if __name__ == '__main__':
    start_date = pd.to_datetime('2016-01-01').date()
    end_date = pd.to_datetime('2023-09-01').date()

    weight_list = [
        'equal_weight',
        'risk_parity',
        #'pca_risk_parity',
        #'half_pca_risk_parity',
        'half_risk_parity',
        'lasso_risk_parity'
    ]

    traday_count = 20
    is_traday, resample = get_traday()
    for i in range((end_date - start_date).days + 1):
        date = start_date + datetime.timedelta(days=i)
        date = date.strftime('%Y-%m-%d')
        if not date in is_traday['trade_date'].values:
            continue
        if traday_count != 20:
            traday_count += 1
            continue
        traday_count = 0

        eindex = is_traday[is_traday['trade_date']==date].index
        sdate = is_traday.iloc[eindex-120, 0].iloc[0]

        login = bs.login()
        rs = bs.query_zz500_stocks(date)
        zz500 = rs.get_data()
        logout = bs.logout()

        stock_mat = get_stock_mat(zz500, sdate, date)
        pdb.set_trace()
        stock_mat = stock_mat.dropna(axis=1)
        stock_mat = stock_mat.loc[:, (stock_mat!=stock_mat.iloc[0]).any()]
        log_return = np.log(stock_mat/stock_mat.shift()).dropna()

        p1 = mp.Pool()
        for wt in weight_list:
            p1.apply_async(cal_weight, (wt, log_return, date,))
        p1.close()
        p1.join()
    