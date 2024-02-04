import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import os, pdb
import akshare as ak

def get_indus():
    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']]

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])
    return indus

def err_call(err):
    print(str(err))

if __name__ == '__main__':
    indus = get_indus()

    indus_yinyang = pd.DataFrame()
    for row in indus.itertuples(name=None):
        sub_index = ak.index_hist_sw(row[2])
        if not sub_index.empty:
            print(row[1])
            sub_index[row[1]] = (sub_index['收盘'] - sub_index['开盘']) / sub_index['开盘']
            
            if indus_yinyang.empty:
                indus_yinyang = sub_index[['日期', row[1]]]
            else:
                indus_yinyang = pd.merge(indus_yinyang, sub_index[['日期',row[1]]], on='日期', how='outer')
    
    indus_yinyang.set_index('日期', inplace=True)
    indus_yinyang.to_csv('/Users/ydgan/Documents/sector_rotation/yin_yang/yinyang.csv')
