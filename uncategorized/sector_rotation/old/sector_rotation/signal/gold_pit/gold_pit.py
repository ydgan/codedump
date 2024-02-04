import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb
import akshare as ak

def get_indus_ratio():
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '20_days_industry_ratio.csv'), index_col=0)
    return indus_ratio

def get_trend():
    trend_path = '/Users/ydgan/Documents/sector_rotation/indus_trend'
    ratio = pd.read_csv(os.path.join(trend_path, 'ratio.csv'), index_col=0)
    change = pd.read_csv(os.path.join(trend_path, 'change.csv'), index_col=0)
    return change

def get_tmt(flag='indus'):
    tmt_path = '/Users/ydgan/Documents/sector_rotation/indus_tmt'
    tmt = pd.read_csv(os.path.join(tmt_path, '%s_tmt.csv'%flag), index_col=0)
    cols = [x.split('_')[0] for x in tmt.columns]
    tmt.columns = cols
    return tmt

if __name__ == '__main__':
    change = get_trend()
    change = change.apply(
        lambda x:np.where(x>0.5, 1, 0)
    )

    indus_ratio = get_indus_ratio()
    indus_ratio = indus_ratio.rolling(10).mean().pct_change()
    indus_ratio = indus_ratio.apply(
        lambda x:np.where(x>0, 1, 0)
    )

    lf_tmt = get_tmt('lf')
    lf_tmt = lf_tmt.apply(
        lambda x:np.where(x<0.1, 1, 0)
    )

    elf_tmt = get_tmt('elf')
    elf_tmt = elf_tmt.apply(
        lambda x:np.where(x<0.1, 1, 0)
    )

    pit = pd.concat([change, indus_ratio, lf_tmt, elf_tmt])
    pit = pit.groupby(level=0).sum()
    pit = pit.apply(
        lambda x:np.where(x==4, 1, 0)
    )

    pit_path = '/Users/ydgan/Documents/sector_rotation/signal/gold_pit'
    pit.to_csv(os.path.join(pit_path, 'gold_pit.csv'))
