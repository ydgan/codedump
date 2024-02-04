import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp

import os, pdb
import math
import akshare as ak

def get_ratio(queue, root, file, indus, l1):
    look_back = root.split('/')[-3].split('_')[0]
    date = file[:-4]
    print(date)

    rank = pd.read_csv(os.path.join(root, file), index_col=0, dtype=object)
    rank = pd.merge(rank, indus, on='code', how='left')
    rank = rank.dropna()

    bestish = math.ceil(rank.shape[0] * 0.1)
    gstock = rank.iloc[:bestish,:]

    i1 = gstock['indus'].value_counts().to_frame()
    i1 = pd.merge(i1, l1[['行业名称', '成份个数']], left_index=True, right_on='行业名称', how='left')
    i1['ratio'] = i1['count'] / i1['成份个数']

    ratio = i1[['行业名称', 'ratio']]
    ratio = ratio.rename(columns={'ratio':date})
    ratio = ratio.set_index('行业名称').T
    queue.put([look_back, ratio])

def ratio_concat(queue):
    df1 = pd.DataFrame()
    df3 = pd.DataFrame()
    df5 = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            look_back, ratio = rec[0], rec[1]
            print('concat %s...'%ratio.index[0])
            if look_back == '1':
                df1 = pd.concat([df1, ratio])
            elif look_back == '3':
                df3 = pd.concat([df3, ratio])
            else:
                df5 = pd.concat([df5, ratio])
        else:
            return df1, df3, df5
    
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
    l1 = ak.sw_index_first_info()
    st = ak.stock_zh_a_st_em

    indus_path = '/Users/ydgan/Documents/sector_rotation_v2/sw_industry'
    indus = pd.read_csv(os.path.join(indus_path,'sw_industry.csv'), dtype=object, index_col=0)
    return_path = '/Users/ydgan/Documents/sector_rotation_v2/return_rank'
    
    latest_date = '2017-01-01'
    '''
    if os.path.exists(os.path.join(indus_ratio_path, '5_days_industry_ratio.csv')):
        xratio5 = pd.read_csv(os.path.join(indus_ratio_path, '5_days_industry_ratio.csv'), index_col=0)
        xratio20 = pd.read_csv(os.path.join(indus_ratio_path, '20_days_industry_ratio.csv'), index_col=0)
        latest_date_5 = xratio5.index[-1]
        latest_date_20 = xratio20.index[-1]
        latest_date = min(latest_date_5, latest_date_20)
    '''

    manager = mp.Manager()
    q = manager.Queue()
    p1 = mp.Pool()
    for root, dirs, files in os.walk(return_path):
        for file in files:
            if file.endswith('.csv') and file[:-4] > latest_date:
                p1.apply_async(get_ratio, (q, root, file, indus, l1,))
    p1.close()
    p1.join()

    df1, df3, df5 = ratio_concat(q)

    '''
    if latest_date != '1990-01-01':
        df5 = df5[df5.index > latest_date_5]
        df5 = pd.concat([xratio5, df5])
        df20 = df20[df20.index > latest_date_20]
        df20 = pd.concat([xratio20, df20])
    '''
    df1 = df1.fillna(0).sort_index()
    df3 = df3.fillna(0).sort_index()
    df5 = df5.fillna(0).sort_index()

    spath1 = os.path.join(os.path.dirname(__file__), '1_days_indus_ratio_rank')
    cols1 = df1.columns.tolist()
    spath3 = os.path.join(os.path.dirname(__file__), '3_days_indus_ratio_rank')
    cols3 = df3.columns.tolist()
    spath5 = os.path.join(os.path.dirname(__file__), '5_days_indus_ratio_rank')
    cols5 = df5.columns.tolist()

    p2 = mp.Pool()
    for row in df1.itertuples(name=None):
        p2.apply_async(save_rank, (cols1, row, spath1,), error_callback=err_call)
    for row in df3.itertuples(name=None):
        p2.apply_async(save_rank, (cols3, row, spath3,), error_callback=err_call)
    for row in df3.itertuples(name=None):
        p2.apply_async(save_rank, (cols5, row, spath5,), error_callback=err_call)
    p2.close()
    p2.join()

    print('done^^')
