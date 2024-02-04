import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import datetime, os
import akshare as ak
import pdb

def sharpe_ratio(series):
    series = series.pct_change()
    xreturn = series.mean()
    ret_std = series.std()
    risk_free = pow(1.03, 1/365) - 1
    return (xreturn - risk_free) / ret_std

def indus_sharpe(queue, row, date):
    sub_index = ak.index_hist_sw(row[2])
    if not sub_index.empty:
        print(row[1])
        date = pd.to_datetime(date).date() - datetime.timedelta(days=500)
        date = date.strftime('%Y-%m-%d')
        sub_index['日期'] = sub_index['日期'].apply(lambda x:x.strftime('%Y-%m-%d'))
        sub_index = sub_index[sub_index['日期'] > date]

        sub_index['year_sharpe'] = sub_index['收盘'].rolling(250).apply(lambda x:sharpe_ratio(x)) * np.sqrt(250)
        sub_index['half_year_sharpe'] = sub_index['收盘'].rolling(120).apply(lambda x:sharpe_ratio(x)) * np.sqrt(120)
        queue.put([
            row[1],
            sub_index[['日期', 'year_sharpe']],
            sub_index[['日期', 'half_year_sharpe']]
        ])

def sharpe_concat(queue):
    year = pd.DataFrame()
    half_year = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            name, year_sharpe, half_year_sharpe = rec[0], rec[1], rec[2]
            year_sharpe = year_sharpe.rename(columns={'year_sharpe':name})
            half_year_sharpe = half_year_sharpe.rename(columns={'half_year_sharpe':name})
           
            if year.empty:
                year = year_sharpe
            else:
                year = pd.merge(year, year_sharpe, on='日期', how='outer')
            if half_year.empty:
                half_year = half_year_sharpe
            else:
                half_year = pd.merge(half_year, half_year_sharpe, on='日期', how='outer')
        else:
            return year, half_year

def err_call(err):
    print(str(err))

if __name__ == '__main__':
    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']]

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])

    sharpe_path = '/Users/ydgan/Documents/sector_rotation/indus_sharpe'

    latest_date = '1990-01-01'
    if os.path.exists(os.path.join(sharpe_path, 'year_sharpe.csv')):
        ysharpe = pd.read_csv(os.path.join(sharpe_path, 'year_sharpe.csv'), index_col=0)
        hysharpe = pd.read_csv(os.path.join(sharpe_path, 'half_year_sharpe.csv'), index_col=0)
        latest_date_y = ysharpe.index[-1]
        latest_date_hy = hysharpe.index[-1]
        latest_date = min(latest_date_y, latest_date_hy)
    
    manager = mp.Manager()
    q = manager.Queue()
    p1 = mp.Pool()
    for row in indus.itertuples(name=None):
        p1.apply_async(indus_sharpe, (q, row, latest_date,), error_callback=err_call)
    p1.close()
    p1.join()

    year_sharpe, half_year_sharpe = sharpe_concat(q)
    year_sharpe = year_sharpe.set_index('日期')
    half_year_sharpe = half_year_sharpe.set_index('日期')

    if latest_date != '1990-01-01':
        year_sharpe = year_sharpe[year_sharpe.index > latest_date_y]
        year_sharpe = pd.concat([year_sharpe, ysharpe])
        half_year_sharpe = half_year_sharpe[half_year_sharpe.index > latest_date_hy]
        half_year_sharpe = pd.concat([half_year_sharpe, hysharpe])

    year_sharpe = year_sharpe.sort_index()
    half_year_sharpe = half_year_sharpe.sort_index()
    year_sharpe = year_sharpe.dropna(axis=0, how='all')
    half_year_sharpe = half_year_sharpe.dropna(axis=0, how='all')

    year_sharpe.to_csv(os.path.join(sharpe_path,'year_sharpe.csv'))
    half_year_sharpe.to_csv(os.path.join(sharpe_path, 'half_year_sharpe.csv'))

