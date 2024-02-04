import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import datetime, os, pdb
import akshare as ak

def percentile(series):
    a,b = 0,1
    x = series.iloc[-1]
    while True:
        p = (a + b) / 2
        if np.percentile(series, p*100) > x:
            b = p
        elif np.percentile(series, p*100) < x:
            a = p
        elif np.percentile(series, p*100) == x:
            return p
        
        if b - a < 1e-5:
            break
    return p

def cal_tmt(series):
    bias = (series - series.mean()) / series.mean()
    return percentile(bias)

def indus_tmt(queue, row, date):
    sub_index = ak.index_hist_sw(row[2])
    date = pd.to_datetime(date).date() - datetime.timedelta(days=500)
    date = date.strftime('%Y-%m-%d')

    if not sub_index.empty:
        print(row[1])
        sub_index['日期'] = sub_index['日期'].apply(lambda x:x.strftime('%Y-%m-%d'))
        sub_index = sub_index[sub_index['日期'] > date]

        for lb in [60, 120, 250]:
            sub_index['%s_%stmt'%(row[1], lb)] = sub_index['收盘'].rolling(lb).apply(lambda x:cal_tmt(x))
        sub_index[row[1]] = sub_index['%s_60tmt'%row[1]] * 0.5 + sub_index['%s_120tmt'%row[1]] * 0.3 + sub_index['%s_250tmt'%row[1]] * 0.2

        queue.put([
            sub_index[['日期',row[1]]],
            sub_index[['日期','%s_60tmt'%row[1]]],
            sub_index[['日期','%s_120tmt'%row[1]]],
            sub_index[['日期','%s_250tmt'%row[1]]]
        ])

def err_call(err):
    print(str(err))

def tmt_concat(queue):
    tmt = pd.DataFrame()
    hftmt = pd.DataFrame()
    lftmt = pd.DataFrame()
    elftmt = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            tmt_tmp, hf_tmp, lf_tmp, elf_tmp = rec[0], rec[1], rec[2], rec[3]

            if tmt.empty:
                tmt = tmt_tmp
            else:
                tmt = pd.merge(tmt, tmt_tmp, on='日期', how='outer')

            if hftmt.empty:
                hftmt = hf_tmp
            else:
                hftmt = pd.merge(hftmt, hf_tmp, on='日期', how='outer')

            if lftmt.empty:
                lftmt = lf_tmp
            else:
                lftmt = pd.merge(lftmt, lf_tmp, on='日期', how='outer')

            if elftmt.empty:
                elftmt = elf_tmp
            else:
                elftmt = pd.merge(elftmt, elf_tmp, on='日期', how='outer')
        else:
            return tmt, hftmt, lftmt, elftmt

if __name__ == '__main__':
    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']]

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])

    tmt_path = '/Users/ydgan/Documents/sector_rotation/indus_tmt'

    latest_date = '1990-01-01'
    if os.path.exists(os.path.join(tmt_path, 'indus_tmt.csv')):
        xtmt = pd.read_csv(os.path.join(tmt_path, 'indus_tmt.csv'), index_col=0)
        xhf = pd.read_csv(os.path.join(tmt_path, 'hf_tmt.csv'), index_col=0)
        xlf = pd.read_csv(os.path.join(tmt_path, 'lf_tmt.csv'), index_col=0)
        xelf = pd.read_csv(os.path.join(tmt_path, 'elf_tmt.csv'), index_col=0)
        latest_date = min(xtmt.index[-1], xhf.index[-1], xlf.index[-1], xelf.index[-1])
    
    manager = mp.Manager()
    q = manager.Queue()
    p1 = mp.Pool()
    for row in indus.itertuples(name=None):
        p1.apply_async(indus_tmt, (q, row, latest_date,), error_callback=err_call)
    p1.close()
    p1.join()

    indus_tmt, hf_tmt, lf_tmt, elf_tmt = tmt_concat(q)
    indus_tmt = indus_tmt.set_index('日期')
    hf_tmt = hf_tmt.set_index('日期')
    lf_tmt = lf_tmt.set_index('日期')
    elf_tmt = elf_tmt.set_index('日期')


    if latest_date != '1990-01-01':
        indus_tmt = indus_tmt[indus_tmt.index > latest_date]
        indus_tmt = pd.concat([indus_tmt, xtmt])

        hf_tmt = hf_tmt[hf_tmt.index > latest_date]
        hf_tmt = pd.concat([hf_tmt, xhf])

        lf_tmt = lf_tmt[lf_tmt.index > latest_date]
        lf_tmt = pd.concat([lf_tmt, xlf])

        elf_tmt = elf_tmt[elf_tmt.index > latest_date]
        elf_tmt = pd.concat([elf_tmt, xelf])

    indus_tmt = indus_tmt.sort_index()
    indus_tmt = indus_tmt.dropna(axis=0, how='all')

    hf_tmt = hf_tmt.sort_index()
    hf_tmt = hf_tmt.dropna(axis=0, how='all')

    lf_tmt = lf_tmt.sort_index()
    lf_tmt = lf_tmt.dropna(axis=0, how='all')

    elf_tmt = elf_tmt.sort_index()
    elf_tmt = elf_tmt.dropna(axis=0, how='all')
    
    indus_tmt.to_csv(os.path.join(tmt_path,'indus_tmt.csv'))
    hf_tmt.to_csv(os.path.join(tmt_path,'hf_tmt.csv'))
    lf_tmt.to_csv(os.path.join(tmt_path,'lf_tmt.csv'))
    elf_tmt.to_csv(os.path.join(tmt_path,'elf_tmt.csv'))
