import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp

import statsmodels.api as sm
import scipy.stats as stats

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

import os, datetime
import akshare as ak

def err_call(err):
    print(str(err))

def get_traday(freq=5):
    is_traday = ak.tool_trade_date_hist_sina()
    is_traday['trade_date'] = pd.to_datetime(is_traday['trade_date'])
    is_traday['trade_date'] = is_traday['trade_date'].apply(
        lambda x:x.strftime('%Y-%m-%d')
    )

    rebal_date = is_traday.iloc[np.arange(0, is_traday.shape[0], freq)].reset_index(drop=True).rename(columns={'trade_date':'rebalance'})
    retra_date = is_traday.iloc[np.arange(1, is_traday.shape[0], freq)].reset_index(drop=True).rename(columns={'trade_date':'retrade'})
    return is_traday, pd.concat([rebal_date, retra_date], axis=1)

def cal_return(md_path, file, resample_date, queue):
    data = pd.read_csv(os.path.join(md_path, file), index_col=0)
    data = data[data.index.isin(resample_date['retrade'])].sort_index()

    data[file[:-4]] = data['open'].pct_change().shift(-1)
    queue.put(data[file[:-4]])

def ret_concat(queue):
    ret = list()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            ret.append(rec)
        else:
            return pd.concat(ret, axis=1).sort_index().dropna(axis=0, how='all')

def get_return(stock_path, resample_date):
    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool()
    for file in os.listdir(stock_path):
        p.apply_async(cal_return, (stock_path, file, resample_date, q,), error_callback=err_call)
    p.close()
    p.join()

    resample_ret = ret_concat(q)
    resample_ret = pd.merge(resample_date, resample_ret, left_on='retrade', right_index=True, how='right')
    resample_ret.set_index('rebalance', inplace=True)
    resample_ret.drop(columns=['retrade'], inplace=True)
    return resample_ret


if __name__ == '__main__':
    rebal_freq = 3
    stock_path = r'E:\data\stock'
    fac_path = r'E:\factors\single_factor\preprocessed'

    rep_path = r'E:\factors_eva\single_factor_eva\report'
    eva_path = r'E:\factors_eva\single_factor_eva'

    start_date = '2018-01-01'
    end_date = '2023-12-31'

    is_traday, resample_date = get_traday(freq=rebal_freq)
    resample_ret = get_return(stock_path, resample_date)

    factor_ic = dict()
    for root, dirs, files in os.walk(fac_path):

        for file in files:
            date = file[:-4]
            if date < start_date or date > end_date or date not in resample_ret.index:
                continue

            factor = root.split('\\')[-3]
            if not factor in factor_ic.keys():
                factor_ic[factor] = dict()
            if not date in factor_ic[factor].keys():
                factor_ic[factor][date] = dict()

            print('%s - %s'%(factor, date))

            tmp_fact = pd.read_csv(os.path.join(root, file), dtype={'Code':object, date:np.float32})
            tmp_ret = resample_ret.loc[date].to_frame().rename(columns={date:'Return'})

            tmp_indi = pd.merge(tmp_fact, tmp_ret, left_on='Code', right_index=True, how='inner')
            tmp_indi = tmp_indi.dropna()
            if not tmp_indi.empty:
                factor_ic[factor][date]['IC'] = stats.spearmanr(tmp_indi[date], tmp_indi['Return'])[0]
                factor_ic[factor][date]['t_value'] = sm.OLS(tmp_indi['Return'], tmp_indi[date]).fit().tvalues[0]

                fact_group = pd.qcut(tmp_indi[date].rank(method='first'),
                                      q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                        labels=[1, 2, 3, 4, 5])
                for group, value in fact_group.to_frame().groupby(date):
                    sub_fact = tmp_indi[tmp_indi.index.isin(value.index)]
                    factor_ic[factor][date]['group%s'%group] = sub_fact['Return'].mean()

    fact_list = []
    ic_list = []
    ir_list = []
    t_list = []
    mono_list = []

    for key, value in factor_ic.items():

        indi_tmp = pd.DataFrame(value, index=['IC', 't_value', 'group1', 'group2', 'group3', 'group4', 'group5']).T.sort_index()

        fact_list.append(key)
        ic_list.append(indi_tmp['IC'].mean())
        ir_list.append(indi_tmp['IC'].mean() / indi_tmp['IC'].std())
        t_list.append(abs(indi_tmp['t_value']).mean())

        indi_tmp.iloc[:, 2:] += 1
        indi_tmp.iloc[:, 2:] = indi_tmp.iloc[:, 2:].cumprod()
        indi_tmp.iloc[:, 2:] /= indi_tmp.iloc[0, 2:]

        mono = pd.DataFrame({'group':[1,2,3,4,5], 
                             'ret':[indi_tmp['group1'].iloc[-1],
                                    indi_tmp['group2'].iloc[-1], 
                                    indi_tmp['group3'].iloc[-1], 
                                    indi_tmp['group4'].iloc[-1], 
                                    indi_tmp['group5'].iloc[-1],]})
        mono_list.append(mono['group'].corr(mono['ret']))

        fig = plt.figure(figsize=(20,12))
        ax = plt.axes()
        plt.bar(np.arange(indi_tmp.shape[0]), indi_tmp['IC'], color='red')
        ax1 = plt.twinx()
        ax1.plot(np.arange(indi_tmp.shape[0]), indi_tmp['IC'].cumsum(), color='blue')

        xtick = np.arange(0, indi_tmp.shape[0], 50)
        xticklabel = pd.Series(indi_tmp.index[xtick])
        ax.set_xticks(xtick)
        ax.set_xticklabels(xticklabel)
        plt.title('%s  IC = %s IC_IR = %s'%(key, round(indi_tmp['IC'].mean(), 4), round(indi_tmp['IC'].mean() / indi_tmp['IC'].std(), 4)))

        p1 = os.path.join(eva_path, str(rebal_freq), key)
        if not os.path.exists(p1):
            os.makedirs(p1)

        indi_tmp.to_csv(os.path.join(p1, 'Rep-%s-%s.csv'%(start_date.replace('-', ''), end_date.replace('-', ''))))

        plt.savefig(os.path.join(p1, 'IC-%s-%s.png'%(start_date.replace('-', ''), end_date.replace('-', ''))), bbox_inches='tight')
        plt.close('all')

        indi_tmp.iloc[:, 2:].plot(figsize=(16,10))
        plt.savefig(os.path.join(p1, 'Ret-%s-%s.png'%(start_date.replace('-', ''), end_date.replace('-', ''))), bbox_inches='tight')
        plt.close('all')

    rep = pd.DataFrame({'Factor':fact_list, 'IC':ic_list, 'IC_IR':ir_list, 'T_value':t_list, 'Mono':mono_list})
    rep.to_csv(os.path.join(rep_path, '%s-%s-%s.csv'%(start_date.replace('-', ''), end_date.replace('-', ''), rebal_freq)))



    