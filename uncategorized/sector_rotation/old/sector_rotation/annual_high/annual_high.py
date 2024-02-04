import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import os, datetime, pdb
import akshare as ak

def get_indus():
    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']]

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])
    return indus

def annual_high(queue, code, date):
    md_path = '/Users/ydgan/Documents/data/stock'
    md = os.path.join(md_path, code+'.csv')
    mdata = pd.read_csv(md, index_col=0) if os.path.exists(md) else pd.DataFrame()
    date = pd.to_datetime(date).date() - datetime.timedelta(days=500)
    date = date.strftime('%Y-%m-%d')
    if not mdata.empty:
        mdata = mdata[mdata['date'] > date]
        print(code)
        mdata[code] = mdata['close'].rolling(250).apply(lambda x:x.iloc[-1] == max(x))
        mdata[code] = mdata[code].rolling(5).apply(lambda x:np.where(sum(x>0)>0, 1, 0))
        queue.put([
            mdata[['date', code]]
        ])
        
def annual_high_concat(queue):
    indus_high = pd.DataFrame()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            high_tmp = rec[0]

            if indus_high.empty:
                indus_high = high_tmp
            else:
                indus_high = pd.merge(indus_high, high_tmp, on='date', how='outer')
        else:
            return indus_high
        
def err_call(err):
    print(str(err))

if __name__ == '__main__':
    indus = get_indus()

    ah_path = '/Users/ydgan/Documents/sector_rotation/annual_high'

    latest_date = '1990-01-01'
    if os.path.exists(os.path.join(ah_path, 'annual_high.csv')):
        xah = pd.read_csv(os.path.join(ah_path, 'annual_high.csv'), index_col=0)
        latest_date = xah.index[-1]

    indus_annual_high = pd.DataFrame()
    for row in indus.itertuples(name=None):
        try:
            indus_comp = ak.index_component_sw(row[2])
        except:
            continue

        if not indus_comp.empty:
            print(row[1])

            manager = mp.Manager()
            q = manager.Queue()
            p = mp.Pool()
            for code in indus_comp['证券代码']:
                p.apply_async(annual_high, (q, code, latest_date,), error_callback=err_call)
            p.close()
            p.join()

            indus_high = annual_high_concat(q)
            indus_high = indus_high.set_index('date').sort_index()
            indus_high = indus_high.dropna(how='all', axis=0)

            indus_high[row[1]] = indus_high.apply(
                lambda x:x[x==1].shape[0] / x.dropna().shape[0], axis=1
            )
        
            if indus_annual_high.empty:
                indus_annual_high = indus_high[[row[1]]]
            else:
                indus_annual_high = pd.merge(indus_annual_high, indus_high[[row[1]]], left_index=True, right_index=True, how='outer')

    if latest_date != '1990-01-01':
        indus_annual_high = indus_annual_high[indus_annual_high.index > latest_date]
        indus_annual_high = pd.concat([indus_annual_high, xah])

    indus_annual_high = indus_annual_high.dropna(axis=0, how='all').sort_index()
    indus_annual_high.to_csv(os.path.join(ah_path, 'annual_high.csv'))