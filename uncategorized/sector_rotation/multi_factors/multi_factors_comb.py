import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.covariance import LedoitWolf

import os, pdb

valid_factors = []

if __name__ == '__main__':

    rebal_freq = 3
    factor_path = r'E:\factors\single_factor\preprocessed'
    
    single_eva_path = os.path.join(r'E:\sector_rotation\single_factor', str(rebal_freq), 'backtesting')
    multi_path = os.path.join(r'E:\sector_rotation\multi_factors', str(rebal_freq))

    start_date = '2018-01-01'
    end_date = '2023-12-31'

    multi_weights = dict()
    for factor in valid_factors:
        hist_ic = pd.read_csv(os.path.join(single_eva_path, '%s-%s-Rep.csv'%(start_date.replace('-', ''),
                                                                                          end_date.replace('-', ''))),
                                                                                            index_col=0)
        multi_weights[factor] = [np.sign(hist_ic.loc[factor]['IC'])]
        #multi_weights[factor] = [1]

    weights = pd.DataFrame(multi_weights)
    weights = weights / weights.shape[1]

    for root, dirs, files in os.walk(os.path.join(multi_path, 'preprocessed')):

        for file in files:

            date = file[:-4]
            multi_factors = pd.read_csv(os.path.join(root, file), dtype={'Code':object})
            multi_factors = multi_factors.set_index('Code')

            comb = (multi_factors * weights.loc[0]).sum(axis=1).to_frame().rename(columns={0:'Equal_Weights'})

            save_path = os.path.join(multi_path, 'combined', date[:4], date[5:7])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            comb.to_csv(os.path.join(save_path, date+'.csv'))
