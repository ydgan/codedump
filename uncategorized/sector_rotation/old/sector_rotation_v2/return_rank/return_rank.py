import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp
import os,pdb

def return_cal(queue, stock, path, date):
    print(stock)
    look_back = [1, 3, 5]
    df = pd.read_csv(os.path.join(path, stock), index_col=0)
    for lb in look_back:
        df['return%s'%lb] = df['close'] / df['close'].shift(lb)
    df = df[df.index > date]

    queue.put([
        stock[:-4],
        df.index.tolist(),
        df['return1'],
        df['return3'],
        df['return5']
    ])

def rank_concat(queue):
    cols = list()
    dict1 = dict()
    dict3 = dict()
    dict5 = dict()
    iter_len = 0

    while True:
        if not queue.empty():
            rec = queue.get(True)
            code, date, tmp1, tmp3, tmp5 = rec[0], rec[1], rec[2], rec[3], rec[4]

            cols.append(code)
            for dt, r1, r3, r5 in zip(date, tmp1, tmp3, tmp5):
                if not dt in dict1.keys():
                    dict1[dt] = [np.nan] * iter_len
                    dict1[dt] += [r1]
                else:
                    if len(dict1[dt]) < iter_len:
                        dict1[dt] += ([np.nan] * (iter_len - len(dict1[dt])))
                    dict1[dt] += [r1]            

                if not dt in dict3.keys():
                    dict3[dt] = [np.nan] * iter_len
                    dict3[dt] += [r3]
                else:
                    if len(dict3[dt]) < iter_len:
                        dict3[dt] += ([np.nan] * (iter_len - len(dict3[dt])))
                    dict3[dt] += [r3] 

                if not dt in dict5.keys():
                    dict5[dt] = [np.nan] * iter_len
                    dict5[dt] += [r5]
                else:
                    if len(dict5[dt]) < iter_len:
                        dict5[dt] += ([np.nan] * (iter_len - len(dict5[dt])))
                    dict5[dt] += [r5] 

            iter_len += 1
        else:
            for k,v in dict1.items():
                if len(v) < iter_len:
                    dict1[k] += ([np.nan] * (iter_len - len(v)))
            for k,v in dict3.items():
                if len(v) < iter_len:
                    dict5[k] += ([np.nan] * (iter_len - len(v)))
            for k,v in dict5.items():
                if len(v) < iter_len:
                    dict5[k] += ([np.nan] * (iter_len - len(v)))

            return cols, dict1, dict3, dict5
        
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

    lrank = '2017-01-01'
    '''
    for root, dirs, files in os.walk(save_path):
        for file in files:
            if file.endswith('.csv') and file[:-4] > lrank:
                lrank = file[:-4]
    '''

    manager = mp.Manager()
    q = manager.Queue() 
    p1 = mp.Pool()
    for stock in os.listdir(md_path):
        p1.apply_async(return_cal, (q, stock, md_path, lrank,))
    p1.close()
    p1.join()

    cols, d1, d3, d5 = rank_concat(q)

    pdb.set_trace()
    rank1 = pd.DataFrame(d1, index=cols).T.sort_index()
    rank3 = pd.DataFrame(d3, index=cols).T.sort_index()
    rank5 = pd.DataFrame(d5, index=cols).T.sort_index()
    rank1 = rank1.dropna(axis=0, how='all')
    rank3 = rank3.dropna(axis=0, how='all')
    rank5 = rank5.dropna(axis=0, how='all')

    spath1 = os.path.join(os.path.dirname(__file__), '1_days_return_rank')
    cols1 = rank1.columns.tolist()
    spath3 = os.path.join(os.path.dirname(__file__), '3_days_return_rank')
    cols3 = rank3.columns.tolist()
    spath5 = os.path.join(os.path.dirname(__file__), '5_days_return_rank')
    cols5 = rank5.columns.tolist()

    p2 = mp.Pool()
    for row in rank1.itertuples(name=None):
        p2.apply_async(save_rank, (cols1, row, spath1,), error_callback=err_call)
    for row in rank3.itertuples(name=None):
        p2.apply_async(save_rank, (cols3, row, spath3,), error_callback=err_call)
    for row in rank5.itertuples(name=None):
        p2.apply_async(save_rank, (cols5, row, spath5,), error_callback=err_call)
    p2.close()
    p2.join()


