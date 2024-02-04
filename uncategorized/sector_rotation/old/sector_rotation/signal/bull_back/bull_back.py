import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb

def get_sharpe(flag='year'):
    sharpe_path = '/Users/ydgan/Documents/sector_rotation/indus_sharpe'
    sharpe = pd.read_csv(os.path.join(sharpe_path, '%s_sharpe.csv'%flag), index_col=0)
    return sharpe

def get_indus_ratio(flag='20'):
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '%s_days_industry_ratio.csv'%flag), index_col=0)
    return indus_ratio

def get_tmt(flag='indus'):
    tmt_path = '/Users/ydgan/Documents/sector_rotation/indus_tmt'
    tmt = pd.read_csv(os.path.join(tmt_path, '%s_tmt.csv'%flag), index_col=0)
    cols = [x.split('_')[0] for x in tmt.columns]
    tmt.columns = cols
    return tmt

def get_yinyang():
    yinyang_path = '/Users/ydgan/Documents/sector_rotation/yin_yang'
    yinyang = pd.read_csv(os.path.join(yinyang_path, 'yinyang.csv'), index_col=0)
    return yinyang

def get_annual_high():
    annual_high_path = '/Users/ydgan/Documents/sector_rotation/annual_high'
    annual_high = pd.read_csv(os.path.join(annual_high_path, 'annual_high.csv'), index_col=0)
    return annual_high

if __name__ == '__main__':
    half_year_sharpe = get_sharpe('half_year')
    half_year_sharpe = half_year_sharpe.T.apply(
        lambda x:np.where(x>=np.nanpercentile(x, 90), 1, 0)
    ).T

    indus_ratio = get_indus_ratio()
    indus_ratio_change = indus_ratio.rolling(10).mean().pct_change()
    indus_ratio = indus_ratio.apply(
        lambda x:np.where(x>0.2, 1, 0)
    )
    indus_ratio_change = indus_ratio_change.apply(
        lambda x:np.where(x>0, 1, 0)
    )
    iratio = pd.concat([indus_ratio, indus_ratio_change])
    iratio = iratio.groupby(level=0).sum()
    iratio = iratio.apply(
        lambda x:np.where(x==2, 1, 0)
    )

    elf_tmt = get_tmt('elf')
    elf_tmt = elf_tmt.apply(
        lambda x:np.where(x>0.5, 1, 0)
    )

    indus_ratio_5 = get_indus_ratio('5')
    indus_ratio_5 = indus_ratio_5.rolling(2).apply(
        lambda x:np.where((x.iloc[0]>0.2 and x.iloc[1]<0.2), 1, 0)
    )

    indus_yinyang = get_yinyang()
    indus_yinyang = indus_yinyang.apply(
        lambda x:np.where(x>-0.02, 1, 0)
    )

    annual_high_5 = get_annual_high()
    annual_high_1 = annual_high_5.T.apply(
        lambda x:np.where((x-x.dropna().mean())>0.2, 1, 0)
    ).T
    annual_high_2 = annual_high_5.T.apply(
        lambda x:np.where((x-x.dropna().mean()-x.dropna().std())>0.1, 1, 0)
    ).T
    annual_high = pd.concat([annual_high_1, annual_high_2])
    annual_high = annual_high.groupby(level=0).sum()
    annual_high = annual_high.apply(
        lambda x:np.where(x>0, 1, 0)
    )

    
    bull_back_tmp = pd.concat([half_year_sharpe, iratio, elf_tmt, indus_ratio_5, indus_yinyang])
    bull_back = bull_back_tmp.groupby(level=0).sum()
    bull_back = bull_back.apply(
        lambda x:np.where(x==5, 1, 0)
    )

    bull_back_tmp = pd.concat([bull_back_tmp, annual_high])
    bbull_back = bull_back_tmp.groupby(level=0).sum()
    bbull_back = bbull_back.apply(
        lambda x:np.where(x==6, 1, 0)
    )

    bb_path = '/Users/ydgan/Documents/sector_rotation/signal/bull_back'
    bull_back.to_csv(os.path.join(bb_path, 'bull_back.csv'))
    bbull_back.to_csv(os.path.join(bb_path, 'bbull_back.csv'))