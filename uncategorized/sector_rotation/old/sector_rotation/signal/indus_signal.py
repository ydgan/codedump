import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import os, pdb
import akshare as ak

def err_call(err):
    print(str(err))

def save_sig(cols, row, spath):
    date = row[0]
    print('saving %s...'%date)
    year = date.split('-')[0]
    month = date.split('-')[1]
    data = list(row[1:])

    ssig = pd.DataFrame({'name':cols, 'signal':data})
    ssig = ssig.sort_values('signal', ascending=False)
    spath = os.path.join(spath, year, month)
    if not os.path.exists(spath):
        os.makedirs(spath)
    ssig.to_csv(os.path.join(spath, '%s.csv'%date))

def get_sig(path, file_name):
    return pd.read_csv(os.path.join(path, file_name+'.csv'), index_col=0)

if __name__ == '__main__':
    trend_following = get_sig('/Users/ydgan/Documents/sector_rotation/signal/trend_following', 'trend_following')
    gas = get_sig('/Users/ydgan/Documents/sector_rotation/signal/gasgas', 'gas')
    indus_reverse = get_sig('/Users/ydgan/Documents/sector_rotation/signal/indus_reverse', 'indus_reverse')
    gold_pit = get_sig('/Users/ydgan/Documents/sector_rotation/signal/gold_pit', 'gold_pit')
    bb = get_sig('/Users/ydgan/Documents/sector_rotation/signal/bull_back', 'bull_back')
    bbb = get_sig('/Users/ydgan/Documents/sector_rotation/signal/bull_back', 'bbull_back')

    sig = pd.concat([trend_following, indus_reverse, gold_pit, bb, bbb])
    sig = sig.groupby(level=0).sum()

    cols = sig.columns.tolist()
    spath = '/Users/ydgan/Documents/sector_rotation/signal/sig'
    p1 = mp.Pool()
    for row in sig.itertuples(name=None):
        p1.apply_async(save_sig, (cols, row, spath,), error_callback=err_call)
    p1.close()
    p1.join()



