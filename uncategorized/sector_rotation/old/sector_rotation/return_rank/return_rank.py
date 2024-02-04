import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp
import os,pdb

def return_cal(queue, stock, path, date):
    print(stock)
    look_back = [5, 20]
    df = pd.read_csv(os.path.join(path, stock), index_col=0)
    for lb in look_back:
        df['return%s'%lb] = df['close'] / df['close'].shift(lb)
    df = df[df['date'] > date]

    queue.put([
        stock[:-4],
        df['date'],
        df['return5'],
        df['return20']
    ])

def rank_concat(queue):
    cols = list()
    dict5 = dict()
    dict20 = dict()
    iter_len = 0

    while True:
        if not queue.empty():
            rec = queue.get(True)
            code, date, tmp5, tmp20 = rec[0], rec[1], rec[2], rec[3]

            cols.append(code)
            for dt, r5, r20 in zip(date, tmp5, tmp20):
                if not dt in dict5.keys():
                    dict5[dt] = [np.nan] * iter_len
                    dict5[dt] += [r5]
                else:
                    if len(dict5[dt]) < iter_len:
                        dict5[dt] += ([np.nan] * (iter_len - len(dict5[dt])))
                    dict5[dt] += [r5]            

                if not dt in dict20.keys():
                    dict20[dt] = [np.nan] * iter_len
                    dict20[dt] += [r20]
                else:
                    if len(dict20[dt]) < iter_len:
                        dict20[dt] += ([np.nan] * (iter_len - len(dict20[dt])))
                    dict20[dt] += [r20] 

            iter_len += 1
        else:
            for k,v in dict5.items():
                if len(v) < iter_len:
                    dict5[k] += ([np.nan] * (iter_len - len(v)))
            for k,v in dict20.items():
                if len(v) < iter_len:
                    dict20[k] += ([np.nan] * (iter_len - len(v)))

            return cols, dict5, dict20
        
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

def err_call(err):
    print(str(err))

if __name__ == '__main__':
    md_path = '/Users/ydgan/Documents/data/stock'
    #md_path = 'D:\data\stock'
    save_path = '/Users/ydgan/Documents/sector_rotation/return_rank'
    #save_path = 'D:\\sector_rotation\\return_rank'

    lrank = '1990-01-01'
    for root, dirs, files in os.walk(save_path):
        for file in files:
            if file.endswith('.csv') and file[:-4] > lrank:
                lrank = file[:-4]

    manager = mp.Manager()
    q = manager.Queue() 
    p1 = mp.Pool()
    for stock in os.listdir(md_path):
        p1.apply_async(return_cal, (q, stock, md_path, lrank,))
    p1.close()
    p1.join()

    cols, d5, d20 = rank_concat(q)

    rank5 = pd.DataFrame(d5, index=cols).T.sort_index()
    rank20 = pd.DataFrame(d20, index=cols).T.sort_index()
    rank5 = rank5.dropna(axis=0, how='all')
    rank20 = rank20.dropna(axis=0, how='all')

    spath5 = os.path.join(save_path, '5_days_return_rank')
    cols5 = rank5.columns.tolist()
    spath20 = os.path.join(save_path, '20_days_return_rank')
    cols20 = rank20.columns.tolist()

    p2 = mp.Pool()
    for row in rank5.itertuples(name=None):
        p2.apply_async(save_rank, (cols5, row, spath5,), error_callback=err_call)
    for row in rank20.itertuples(name=None):
        p2.apply_async(save_rank, (cols20, row, spath20,), error_callback=err_call)
    p2.close()
    p2.join()


