import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb
import akshare as ak

def get_indus_ratio(flag='20'):
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '%s_days_industry_ratio.csv'%flag), index_col=0)
    return indus_ratio

def strange(series):
    if series[-1] < 0.1:
        return 0
    elif sum(series>0.1) < 3:
        return 0
    elif sum(series[series>0.1].diff() > 0) < 2:
        return 0
    else:
        return 1
    
def get_tmt(flag='indus'):
    tmt_path = '/Users/ydgan/Documents/sector_rotation/indus_tmt'
    tmt = pd.read_csv(os.path.join(tmt_path, '%s_tmt.csv'%flag), index_col=0)
    cols = [x.split('_')[0] for x in tmt.columns]
    tmt.columns = cols
    return tmt

def get_trend():
    trend_path = '/Users/ydgan/Documents/sector_rotation/indus_trend'
    ratio = pd.read_csv(os.path.join(trend_path, 'ratio.csv'), index_col=0)
    change = pd.read_csv(os.path.join(trend_path, 'change.csv'), index_col=0)
    return ratio, change

def get_sharpe(flag='year'):
    sharpe_path = '/Users/ydgan/Documents/sector_rotation/indus_sharpe'
    sharpe = pd.read_csv(os.path.join(sharpe_path, '%s_sharpe.csv'%flag), index_col=0)
    return sharpe

if __name__ == '__main__':
    indus_ratio_5 = get_indus_ratio('5')
    indus_ratio_20 = get_indus_ratio()
    strange = indus_ratio_20.rolling(5).apply(
         lambda x:strange(x)
    )

    ratio_trend = indus_ratio_20.rolling(10).mean().pct_change()
    ratio_trend = ratio_trend.apply(
        lambda x:np.where(x>0, 1, 0)
    )

    indus_tmt = get_tmt()
    indus_tmt = indus_tmt.apply(
        lambda x:np.where(x>0.7, 0, 1)
    )

    year_sharpe = get_sharpe()
    half_year_sharpe = get_sharpe('half_year')
    year_sharpe = year_sharpe.T.apply(
        lambda x:np.where(x>=np.nanpercentile(x, 80), 2,
                          np.where(x>=np.nanpercentile(x, 50), 1,
                                   np.where(x>=np.nanpercentile(x, 20), -1, -2)))
    ).T
    half_year_sharpe = half_year_sharpe.T.apply(
        lambda x:np.where(x>=np.nanpercentile(x, 80), 2,
                          np.where(x>=np.nanpercentile(x, 50), 1,
                                   np.where(x>=np.nanpercentile(x, 20), -1, -2)))
    ).T
    idx = half_year_sharpe.index.intersection(year_sharpe.index)
    sharpe_rank = year_sharpe[year_sharpe.index.isin(idx)] + half_year_sharpe[half_year_sharpe.index.isin(idx)]
    sharpe_rank = sharpe_rank.apply(
        lambda x:np.where(x<0, 0, 1)
    )

    ratio, change = get_trend()
    ratio = ratio.apply(
        lambda x:np.where(x>0.2, 1, 0)
    )
    change = change.apply(
        lambda x:np.where(x>0.3, 1, 0)
    )
    trend = pd.concat([ratio, change])
    trend = trend.groupby(level=0).sum()
    trend = trend.apply(
        lambda x:np.where(x>0, 1, 0)
    )

    trend_following = pd.concat([strange, ratio_trend, indus_tmt, sharpe_rank, change])
    trend_following = trend_following.groupby(level=0).sum()
    trend_following = trend_following.apply(
        lambda x:np.where(x==5, 1, 0)
    )

    trend_following_path = '/Users/ydgan/Documents/sector_rotation/signal/trend_following'
    trend_following.to_csv(os.path.join(trend_following_path, 'trend_following.csv'))