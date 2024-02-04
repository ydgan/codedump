import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb

def get_trend():
    trend_path = '/Users/ydgan/Documents/sector_rotation/indus_trend'
    ratio = pd.read_csv(os.path.join(trend_path, 'ratio.csv'), index_col=0)
    change = pd.read_csv(os.path.join(trend_path, 'change.csv'), index_col=0)
    return ratio, change

def get_indus_ratio(flag='20'):
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '%s_days_industry_ratio.csv'%flag), index_col=0)
    return indus_ratio

if __name__ == '__main__':
    ratio, change = get_trend()
    ratio = ratio.apply(
        lambda x:np.where(x>0.5, 1, 0)
    )
    change = change.apply(
        lambda x:np.where(x>0.3, 1, 0)
    )

    indus_ratio = get_indus_ratio()
    indus_ratio = indus_ratio.rolling(10).mean()
    indus_ratio = indus_ratio.apply(
        lambda x:np.where(x<0.25, 1, 0)
    )

    gas = pd.concat([ratio, change, indus_ratio])
    gas = gas.groupby(level=0).sum()
    gas = gas.apply(
        lambda x:np.where(x==3, 1, 0)
    )

    gas_path = '/Users/ydgan/Documents/sector_rotation/signal/gasgas'
    gas.to_csv(os.path.join(gas_path, 'gas.csv'))