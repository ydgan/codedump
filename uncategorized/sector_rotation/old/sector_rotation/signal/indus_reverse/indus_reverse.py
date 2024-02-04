import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np

import os, pdb
import akshare as ak

def get_sharpe(flag='year'):
    sharpe_path = '/Users/ydgan/Documents/sector_rotation/indus_sharpe'
    sharpe = pd.read_csv(os.path.join(sharpe_path, '%s_sharpe.csv'%flag), index_col=0)
    return sharpe

def get_trend():
    trend_path = '/Users/ydgan/Documents/sector_rotation/indus_trend'
    #ratio = pd.read_csv(os.path.join(trend_path, 'ratio.csv'), index_col=0)
    change = pd.read_csv(os.path.join(trend_path, 'change.csv'), index_col=0)
    return change

def get_tmt(flag='indus'):
    tmt_path = '/Users/ydgan/Documents/sector_rotation/indus_tmt'
    tmt = pd.read_csv(os.path.join(tmt_path, '%s_tmt.csv'%flag), index_col=0)
    cols = [x.split('_')[0] for x in tmt.columns]
    tmt.columns = cols
    return tmt

def get_indus_ratio():
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '20_days_industry_ratio.csv'), index_col=0)
    return indus_ratio

if __name__ == '__main__':
    half_year_sharpe = get_sharpe('half_year')
    half_year_sharpe = half_year_sharpe.T.apply(
        lambda x:np.where(x>np.nanpercentile(x, 20), 0, 1)
    ).T

    lf_tmt = get_tmt('lf')
    lf_tmt = lf_tmt.apply(
        lambda x:np.where(x<0.4, 1, 0)
    )
    hf_tmt = get_tmt('hf')
    hf_tmt = hf_tmt.apply(
        lambda x:np.where(x<0.4, 1, 0)
    )
    tmt = pd.concat([lf_tmt, hf_tmt])
    tmt = tmt.groupby(level=0).sum()
    tmt = tmt.apply(
        lambda x:np.where(x==2, 1, 0)
    )

    indus_ratio = get_indus_ratio()
    indus_ratio = indus_ratio.rolling(10).mean()
    indus_ratio = indus_ratio.apply(
        lambda x:np.where(x>0.05, 1, 0)
    )

    change = get_trend()
    change = change.apply(
        lambda x:np.where(x>0.4, 1, 0)
    )

    rev = pd.concat([half_year_sharpe, tmt, indus_ratio, change])
    rev = rev.groupby(level=0).sum()
    rev = rev.apply(
        lambda x:np.where(x==4, 1, 0)
    )

    rev_path = '/Users/ydgan/Documents/sector_rotation/signal/indus_reverse'
    rev.to_csv(os.path.join(rev_path, 'indus_reverse.csv'))