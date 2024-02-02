import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import multiprocessing as mp

from alphak import factors
import os

def cal_factor(alpha, factor):
    fpath = r'E:\factors\single_factor\raw'
    if os.path.exists(os.path.join(fpath, factor, '2024')):
        return

    print(factor)
    fac_res = getattr(alpha, factor)()
    fac_res = fac_res.interpolate(limit_area='inside')

    for row in fac_res.dropna(axis=0, how='all').itertuples():
        if row[0] < '2010-01-01':
            continue

        fac_tmp = pd.DataFrame({'Code':fac_res.columns, row[0]:row[1:]}).dropna()

        year = row[0].split('-')[0]
        month = row[0].split('-')[1]
        save_path = os.path.join(r'E:\factors\single_factor', 'raw', factor, year, month)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fac_tmp.to_csv(os.path.join(save_path, row[0]+'.csv'), index=False)

def err_call(err):
    print(str(err))


if __name__ == '__main__':

    alpha = factors.ALPHA()

    open_tuple = tuple()
    high_tuple = tuple()
    low_tuple = tuple()
    close_tuple = tuple()
    volume_tuple = tuple()
    amount_tuple = tuple()
    turnover_tuple = tuple()
    for market_data in os.listdir(r'E:\data\stock'):
        mdata = pd.read_csv(os.path.join(r'E:\data\stock', market_data), index_col=0)

        alpha.data_feed(market_data[:-4], mdata)
        open_tuple = open_tuple.__add__((mdata.iloc[:, 0].to_frame().rename(columns={'open':market_data[:-4]}),))
        high_tuple = high_tuple.__add__((mdata.iloc[:, 1].to_frame().rename(columns={'high':market_data[:-4]}),))
        low_tuple = low_tuple.__add__((mdata.iloc[:, 2].to_frame().rename(columns={'low':market_data[:-4]}),))
        close_tuple = close_tuple.__add__((mdata.iloc[:, 3].to_frame().rename(columns={'close':market_data[:-4]}),))
        volume_tuple = volume_tuple.__add__((mdata.iloc[:, 4].to_frame().rename(columns={'volume':market_data[:-4]}),))
        amount_tuple = amount_tuple.__add__((mdata.iloc[:, 5].to_frame().rename(columns={'amount':market_data[:-4]}),))
        turnover_tuple = turnover_tuple.__add__((mdata.iloc[:, 6].to_frame().rename(columns={'turn':market_data[:-4]}),))
    alpha.cols_feed(open_tuple, high_tuple, low_tuple, close_tuple, volume_tuple, amount_tuple, turnover_tuple)

    fac_list = [x for x in dir(alpha) if not x.startswith('_') 
                 and not x in ['data_feed', 'cols_feed', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'vwap'] 
                 and not 'tuple' in x]
    
    #f2 = [x for x in fac_list if not os.path.exists(r'E:\factors\single_factor\raw\%s'%(x))]

    for factor in fac_list:
        cal_factor(alpha, factor)
        #try:
            #cal_factor(alpha, factor)
        #except:
            #continue