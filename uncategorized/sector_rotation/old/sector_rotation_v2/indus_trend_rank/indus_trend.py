import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import os, pdb
import datetime
import akshare as ak

def abs_eng(series):
    abseng = (series.max() - series.min()) ** 2

    lefty = 1e5
    spread = 1e5
    if abseng == 0:
        minidx = [series.index[0]]
        maxidx = [series.index[-1]]
    else:
        maxidx = series[series == series.max()].index
        minidx = series[series == series.min()].index

    for minn in minidx:
        for maxx in maxidx:
            spd = abs(maxx - minn)
            if spd < spread:
                spread = spd
                lefty = min(maxx, minn)
            elif spd == spread:
                lefty = min(lefty, min(maxx, minn))

    if len(series) == 1:
        return abseng
    else:
        return abseng + abs_eng(series.loc[:lefty]) + abs_eng(series.loc[lefty+spread:])
    
def cal_direc_eng(series):
    stepp = series.cumsum()

    chg_idx = [series.index[0], series.index[-1]]
    chad_series = series[series != 0]
    chad = chad_series.rolling(2).sum()
    chg_idx += chad[chad == 0].index.tolist()
    chg_idx = list(set(chg_idx))
    chg_idx.sort()
    poteng = 0
    for i,j in zip(chg_idx[:-1], chg_idx[1:]):
        poteng += (stepp.loc[j] - stepp.loc[i]) ** 2

    abseng = abs_eng(stepp)
    return stepp.iloc[-1] * max(poteng, abseng)
    
def direng(queue, code, date):
    md_path = '/Users/ydgan/Documents/data/stock'
    md = os.path.join(md_path, code+'.csv')
    mdata = pd.read_csv(md, index_col=0) if os.path.exists(md) else pd.DataFrame()
    if not mdata.empty:
        print(code)
        date = pd.to_datetime(date).date() - datetime.timedelta(days = 30)
        date = date.strftime('%Y-%m-%d')

        mdata.reset_index(inplace=True)
        mdata = mdata[mdata['date'] > date]

        mdata['change'] = mdata['close'].pct_change()
        mdata['ma5'] = mdata['close'].rolling(5).mean()
        mdata['cloma'] = np.where(mdata['close'] <= mdata['ma5'], -1, 1)
        mdata['twochg'] = mdata['cloma'].rolling(2).sum()
        mdata = mdata.dropna(subset='ma5')

        mdata['trend'] = np.where(mdata['twochg'] == -2, np.where(mdata['change'] < 0, -1, 0),
                                    np.where(mdata['twochg'] == 2, np.where(mdata['change'] > 0, 1, 0),
                                            np.where(mdata['close'] > mdata['ma5'], 1, -1)))
        mdata[code] = mdata['trend'].rolling(5).apply(lambda x:cal_direc_eng(x))

        queue.put(
            [
                mdata[['date',code]]
            ]
        )

def err_call(err):
    print(str(err))

def direng_concat(queue):
    indus_df = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            indus_tmp = rec[0]

            if indus_df.empty:
                indus_df = indus_tmp
            else:
                indus_df = pd.merge(indus_df, indus_tmp, on='date', how='outer')
        else:
            return indus_df
        
def save_rank(cols, row, spath):
    date = row[0]
    print('saving %s...'%date)
    year = date.split('-')[0]
    month = date.split('-')[1]
    data = list(row[1:])

    srank = pd.DataFrame({'code':cols, 'ratio':data}).dropna()
    srank = srank.sort_values('ratio', ascending=False)
    spath = os.path.join(spath, year, month)
    if not os.path.exists(spath):
        os.makedirs(spath)
    srank.to_csv(os.path.join(spath, '%s.csv'%date))

if __name__ == '__main__':
    indus_path = '/Users/ydgan/Documents/sector_rotation_v2/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)

    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']][l1['行业名称'].isin(findus['indus'].unique())]
    indus['行业代码'] = indus['行业代码'].str[:-3]

    latest_date = '2017-01-01'
    '''
    if os.path.exists(os.path.join(trend_path, 'ratio.csv')):
        xratio = pd.read_csv(os.path.join(trend_path, 'ratio.csv'), index_col=0)
        xchange = pd.read_csv(os.path.join(trend_path, 'change.csv'), index_col=0)
        latest_date_r = xratio.index[-1]
        latest_date_c = xchange.index[-1]
        latest_date = min(latest_date_c, latest_date_r)
    '''

    indus_ratio = pd.DataFrame()
    indus_change = pd.DataFrame()
    for row in indus.itertuples(name=None):
        indus_comp = ak.index_component_sw(row[2])
        if not indus_comp.empty:
            print(row[1])

            manager = mp.Manager()
            q1 = manager.Queue()
            p1 = mp.Pool()
            for code in indus_comp['证券代码']:
                p1.apply_async(direng, (q1, code, latest_date,), error_callback=err_call)
            p1.close()
            p1.join()

            indus_df = direng_concat(q1)
            indus_df = indus_df.set_index('date').sort_index().dropna(axis=0, how='all')
            
            ratio = indus_df.apply(lambda x:sum(x>0) / x.dropna().shape[0], axis=1)
            change = indus_df.pct_change().dropna(axis=0, how='all')
            change = change.apply(lambda x:sum(x>0) / x.dropna().shape[0], axis=1)

            if indus_ratio.empty:
                indus_ratio = ratio.to_frame()
            else:
                indus_ratio = pd.merge(indus_ratio, ratio.to_frame(), left_index=True, right_index=True, how='outer')
            if indus_change.empty:
                indus_change = change.to_frame()
            else:
                indus_change = pd.merge(indus_change, change.to_frame(), left_index=True, right_index=True, how='outer')
            indus_ratio = indus_ratio.rename(columns={0:row[1]})
            indus_change = indus_change.rename(columns={0:row[1]})
    '''
    if latest_date != '1990-01-01':
        indus_ratio = indus_ratio[indus_ratio.index > latest_date]
        indus_change = indus_change[indus_change.index > latest_date]
        indus_ratio = pd.concat([xratio, indus_ratio])
        indus_change = pd.concat([xchange, indus_change])
    '''

    indus_ratio = indus_ratio.dropna(axis=0, how='all').sort_index()
    indus_change = indus_change.dropna(axis=0, how='all').sort_index()

    spathr = os.path.join(os.path.dirname(__file__), 'ratio_rank')
    colsr = indus_ratio.columns.tolist()
    spathc = os.path.join(os.path.dirname(__file__), 'change_rank')
    colsc = indus_change.columns.tolist()

    p2 = mp.Pool()
    for row in indus_ratio.itertuples(name=None):
        p2.apply_async(save_rank, (colsr, row, spathr,), error_callback=err_call)
    for row in indus_change.itertuples(name=None):
        p2.apply_async(save_rank, (colsc, row, spathc,), error_callback=err_call)
    p2.close()
    p2.join()