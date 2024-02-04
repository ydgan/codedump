import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import multiprocessing as mp

import os, pdb
import math
import akshare as ak

def get_ratio(queue, root, file, indus, l1):
    look_back = root.split('/')[-3].split('_')[0]
    date = file[:-4]
    print(date)

    rank = pd.read_csv(os.path.join(root, file), index_col=0, dtype='object')

    bestish = math.ceil(rank.shape[0] * 0.1)
    gstock = rank.iloc[:bestish,0].to_frame()
    
    gstock = pd.merge(gstock, indus, on='code', how='left')
    i1 = gstock['indus1'].value_counts().to_frame()

    i1 = pd.merge(i1, l1[['行业名称', '成份个数']], left_index=True, right_on='行业名称', how='left')
    i1['ratio'] = i1['count'] / i1['成份个数']

    ratio = i1[['行业名称', 'ratio']]
    ratio = ratio.rename(columns={'ratio':date})
    ratio = ratio.set_index('行业名称').T
    queue.put([look_back, ratio])

def ratio_concat(queue):
    df5 = pd.DataFrame()
    df20 = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            look_back, ratio = rec[0], rec[1]
            print('concat %s...'%ratio.index[0])
            if look_back == '5':
                df5 = pd.concat([df5, ratio])
            else:
                df20 = pd.concat([df20, ratio])
        else:
            return df5, df20   

if __name__ == '__main__':
    l1 = ak.sw_index_first_info()

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    indus = pd.read_csv(os.path.join(indus_path,'sw_industry.csv'), dtype=object, index_col=0)

    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    return_path = '/Users/ydgan/Documents/sector_rotation/return_rank'
    
    latest_date = '1990-01-01'
    if os.path.exists(os.path.join(indus_ratio_path, '5_days_industry_ratio.csv')):
        xratio5 = pd.read_csv(os.path.join(indus_ratio_path, '5_days_industry_ratio.csv'), index_col=0)
        xratio20 = pd.read_csv(os.path.join(indus_ratio_path, '20_days_industry_ratio.csv'), index_col=0)
        latest_date_5 = xratio5.index[-1]
        latest_date_20 = xratio20.index[-1]
        latest_date = min(latest_date_5, latest_date_20)

    manager = mp.Manager()
    q = manager.Queue()
    p1 = mp.Pool()
    for root, dirs, files in os.walk(return_path):
        for file in files:
            if file.endswith('.csv') and file[:-4] > latest_date:
                p1.apply_async(get_ratio, (q, root, file, indus, l1,))
    p1.close()
    p1.join()

    df5, df20 = ratio_concat(q)

    if latest_date != '1990-01-01':
        df5 = df5[df5.index > latest_date_5]
        df5 = pd.concat([xratio5, df5])
        df20 = df20[df20.index > latest_date_20]
        df20 = pd.concat([xratio20, df20])
    df5 = df5.fillna(0).sort_index()
    df20 = df20.fillna(0).sort_index()

    df5.to_csv(os.path.join(indus_ratio_path, '5_days_industry_ratio.csv'))
    df20.to_csv(os.path.join(indus_ratio_path, '20_days_industry_ratio.csv'))