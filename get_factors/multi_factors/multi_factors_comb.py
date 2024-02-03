import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.covariance import LedoitWolf

import os

valid_factors = []

if __name__ == '__main__':

    rebal_freq = 3
    factor_path = r'E:\factors\single_factor\preprocessed'
    
    single_eva_path = os.path.join(r'E:\factors_eva\single_factor_eva', str(rebal_freq))
    multi_path = os.path.join(r'E:\factors\multi_factors', str(rebal_freq))

    start_date = '2018-01-01'
    end_date = '2023-12-31'

    ic_mat = pd.DataFrame()
    multi_weights = dict()
    multi_weights['Equal_Weights'] = pd.DataFrame()
    multi_weights['IC_Weights'] = pd.DataFrame()
    multi_weights['IR_Weights'] = pd.DataFrame()
    multi_weights['Half_Life_IC_Weights'] = pd.DataFrame()
    multi_weights['Half_Life_IR_Weights'] = pd.DataFrame()
    multi_weights['Max_IC_Weights'] = pd.DataFrame()
    multi_weights['Max_IR_Weights'] = pd.DataFrame()
    for factor in valid_factors:
        hist_ic = pd.read_csv(os.path.join(single_eva_path, factor, 'Rep-%s-%s.csv'%(start_date.replace('-', ''),
                                                                                          end_date.replace('-', ''))),
                                                                                            index_col=0)
        hist_ic['IC_IR'] = hist_ic['IC'].rolling(1000, min_periods=1).mean() /\
                                     hist_ic['IC'].rolling(1000, min_periods=1).std()
        hist_ic[['IC', 'IC_IR']] = hist_ic[['IC', 'IC_IR']].shift()

        def half_life_weights(array):
            T = len(array)
            H = T // 2 if T > 1 else 1

            t = np.arange(1, T+1)
            wt = 2**((t-T-1)/H)
            wt /= wt.sum()
            return (array * wt).sum()

        ew = np.sign(hist_ic['IC'].rolling(12, min_periods=1).mean()).to_frame().rename(columns={'IC':factor})
        icw = hist_ic['IC'].rolling(12, min_periods=1).mean().to_frame().rename(columns={'IC':factor})
        irw = hist_ic['IC_IR'].rolling(12, min_periods=1).mean().to_frame().rename(columns={'IC_IR':factor})
        hlicw = hist_ic['IC'].rolling(12, min_periods=1).apply(half_life_weights).to_frame().rename(columns={'IC':factor})
        hlirw = hist_ic['IC_IR'].rolling(12, min_periods=1).apply(half_life_weights).to_frame().rename(columns={'IC_IR':factor})

        multi_weights['Equal_Weights'] = pd.concat([multi_weights['Equal_Weights'], ew], axis=1)
        multi_weights['IC_Weights'] = pd.concat([multi_weights['IC_Weights'], icw], axis=1)
        multi_weights['IR_Weights'] = pd.concat([multi_weights['IR_Weights'], irw], axis=1)
        multi_weights['Half_Life_IC_Weights'] = pd.concat([multi_weights['Half_Life_IC_Weights'], hlicw], axis=1)
        multi_weights['Half_Life_IR_Weights'] = pd.concat([multi_weights['Half_Life_IR_Weights'], hlirw], axis=1)

        ic_mat = pd.concat([ic_mat, hist_ic['IC']], axis=1)
        ic_mat = ic_mat.rename(columns={'IC':factor}).dropna()

    max_ic_weights = pd.DataFrame()
    max_ir_weights = pd.DataFrame()
    for date in ic_mat.index[1:]:
        ic_tmp = ic_mat.loc[:date, :]

        model = LedoitWolf()
        model.fit(ic_tmp)
        ic_cov = model.covariance_
        inv_ic_cov = np.linalg.inv(ic_cov)
        ir_wts = inv_ic_cov * np.mat(ic_tmp.mean()).T
        ir_weights = pd.DataFrame(ir_wts.T, index=[date], columns=ic_mat.columns)
        max_ir_weights = pd.concat([max_ir_weights, ir_weights])

        max_ic = pd.DataFrame()
        for fact in ic_tmp.columns:
            fact_tmp = pd.read_csv(os.path.join(factor_path, fact, date[:4], date[5:7], date+'.csv'),
                                    dtype={'Code':object, date:np.float32})
            fact_tmp = fact_tmp.set_index('Code')
            max_ic = pd.concat([max_ic, fact_tmp], axis=1)
            max_ic = max_ic.rename(columns={date:fact})
        max_ic = max_ic.dropna()

        fact_cov = max_ic.cov()
        inv_fact_cov = np.linalg.inv(fact_cov)
        ic_wts = inv_fact_cov * np.mat(ic_tmp.mean()).T
        ic_weights = pd.DataFrame(ic_wts.T, index=[date], columns=max_ic.columns)
        max_ic_weights = pd.concat([max_ic_weights, ic_weights])

    multi_weights['Max_IC_Weights'] = max_ic_weights
    multi_weights['Max_IR_Weights'] = max_ir_weights

    for wts in multi_weights.keys():
        multi_weights[wts] = multi_weights[wts].dropna()

        if wts == 'Equal_Weights':
            multi_weights[wts] = multi_weights[wts].apply(lambda x:x/abs(x).sum(), axis=1)
        else:
            multi_weights[wts] = multi_weights[wts].apply(lambda x:x/x.sum(), axis=1)


    for root, dirs, files in os.walk(os.path.join(multi_path, 'preprocessed')):

        for file in files:

            date = file[:-4]
            multi_factors = pd.read_csv(os.path.join(root, file), dtype={'Code':object})
            multi_factors = multi_factors.set_index('Code')

            comb = pd.DataFrame()
            for k, v in multi_weights.items():
                try:
                    comb = pd.concat([comb, (multi_factors * v.loc[date]).sum(axis=1).to_frame().rename(columns={0:k})], axis=1)
                except:
                    continue
            
            if comb.empty:
                continue

            save_path = os.path.join(multi_path, 'combined', date[:4], date[5:7])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            comb.to_csv(os.path.join(save_path, date+'.csv'))
