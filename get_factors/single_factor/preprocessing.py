import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp
from scipy.stats.mstats import winsorize
import statsmodels.api as sm

import os

def err_call(err):
    print(str(err))

def multi_info(stock_path, file, queue):
    market_data = pd.read_csv(os.path.join(stock_path, file), index_col=0)

    queue.put([
        market_data.iloc[:, -6].to_frame().rename(columns={'isST':file[:-4]}),
        market_data.iloc[:, -5].to_frame().rename(columns={'isNEW':file[:-4]}),
        market_data.iloc[:, -4].to_frame().rename(columns={'mcap':file[:-4]}),
        market_data.iloc[:, -3].to_frame().rename(columns={'sw_level_1':file[:-4]})
    ])

def info_concat(queue):
    isST = list()
    isNew = list()
    market_cap = list()
    industry = list()

    while True:
        if not queue.empty():
            res = queue.get(True)

            isST.append(res[0])
            isNew.append(res[1])
            market_cap.append(res[2])
            industry.append(res[3])
        else:
            return pd.concat(isST, axis=1).sort_index(), pd.concat(isNew, axis=1).sort_index(),\
                    pd.concat(market_cap, axis=1).sort_index(), pd.concat(industry, axis=1).sort_index()

def info_collect(stock_path):
    print('Collecting info...')

    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool()
    for file in os.listdir(stock_path):
        p.apply_async(multi_info, (stock_path, file, q,), error_callback=err_call)
    p.close()
    p.join()

    isST, isNew, mcap, industry = info_concat(q)
    return isST, isNew, mcap, industry

def stock_filter(loc, path, isST, isNew):
    factor = pd.read_csv(path, dtype=object)
    factor = pd.merge(factor, isST.loc[loc, :].to_frame().rename(columns={loc:'isST'}),
                               left_on='Code', right_index=True, how='left')
    factor = pd.merge(factor, isNew.loc[loc, :].to_frame().rename(columns={loc:'isNEW'}),
                        left_on='Code', right_index=True, how='left')
    return factor[(factor['isST']!=1) & (factor['isNEW']!=1)]

def factor_neut(loc, factor, mcap, industry):
    factor = pd.merge(factor, mcap.loc[loc, :].to_frame().rename(columns={loc:'mcap'}),
                                left_on='Code', right_index=True, how='left')
    factor = pd.merge(factor, industry.loc[loc, :].to_frame().rename(columns={loc:'industry'}),
                    left_on='Code', right_index=True, how='left')
    
    factor['mcap'] = np.log(factor['mcap'])
    factor[loc] = factor[loc].astype(np.float32)
    factor = factor.replace([-np.inf, np.inf], np.nan)

    factor['mcap'].fillna(factor['mcap'].mean(), inplace=True)
    factor[loc] = np.where(factor[loc] >= np.percentile(factor[loc].dropna(), 95), np.percentile(factor[loc].dropna(), 95),
                           np.where(factor[loc] <= np.percentile(factor[loc].dropna(), 5), np.percentile(factor[loc].dropna(), 5),
                                    factor[loc]))

    neut = pd.get_dummies(factor['industry'], dummy_na=True, dtype=int)
    neut = pd.concat([neut, factor['mcap']], axis=1)
    res_fit = sm.OLS(factor[loc].astype(float), neut.astype(float)).fit()

    factor[loc] = res_fit.resid
    return factor[['Code', loc]]

def preprocess(root, file, isST, isNew, mcap, industry, fac_path):
    for file in files:
        if os.path.exists(os.path.join(fac_path, root.split('\\')[-3], root.split('\\')[-2], root.split('\\')[-1], file)):
            continue

        print('%s - %s'%(root.split('\\')[-3], file[:-4]))
        factor = stock_filter(file[:-4], os.path.join(root, file), isST, isNew)
        
        if not factor.empty:
            factor = factor_neut(file[:-4], factor, mcap, industry)
            factor[file[:-4]] = (factor[file[:-4]] - factor[file[:-4]].mean()) / factor[file[:-4]].std()

            year = root.split('\\')[-2]
            month = root.split('\\')[-1]
            fac = root.split('\\')[-3]
            save_path = os.path.join(fac_path, fac, year, month)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            factor.to_csv(os.path.join(save_path, file), index=False)


if __name__ == '__main__':
    raw_fac_path = r'E:\factors\single_factor\raw'
    fac_path = r'E:\factors\single_factor\preprocessed'
    stock_path = r'E:\data\stock'

    isST, isNew, mcap, industry = info_collect(stock_path)

    for root, dirs, files in os.walk(raw_fac_path):
        preprocess(root, files, isST, isNew, mcap, industry, fac_path)
    



                


