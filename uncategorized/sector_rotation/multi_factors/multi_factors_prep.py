import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb

valid_factors = []

def symmetry_ortho(factors):
    idx = factors.index
    cols = factors.columns

    D, U = np.linalg.eig(np.dot(factors.T, factors))
    S = np.dot(U, np.diag(D**(-1/2)))

    F_hat = np.dot(factors, S)
    F_hat = np.dot(F_hat, U.T)
    return pd.DataFrame(F_hat, columns=cols, index=idx)


if __name__ == '__main__':
    rebal_freq = 3
    single_factor_path = r'E:\factors\single_factor\preprocessed'
    multi_factors_path = r'E:\sector_rotation\multi_factors'

    start_date = '2018-01-01'
    end_date = '2023-12-31'

    multi_factors = dict()
    for root, dirs, files in os.walk(single_factor_path):
        if not root.split('\\')[-3] in valid_factors:
            continue
        
        for file in files:
            if file[:-4] < start_date or file[:-4] > end_date:
                continue
            
            print('%s - %s'%(root.split('\\')[-3], file[:-4]))

            if not file[:-4] in multi_factors.keys():
                multi_factors[file[:-4]] = pd.DataFrame()

            tmp_factor = pd.read_csv(os.path.join(root, file), dtype={'Code':object, file[:-4]:np.float32})
            tmp_factor = tmp_factor.replace([-np.inf, np.inf], np.nan)
            tmp_factor = tmp_factor.dropna()

            tmp_factor = tmp_factor.rename(columns={file[:-4]:root.split('\\')[-3]})
            tmp_factor.set_index('Code', inplace=True)

            if multi_factors[file[:-4]].empty:
                multi_factors[file[:-4]] = tmp_factor
            else:
                multi_factors[file[:-4]] = pd.merge(multi_factors[file[:-4]], tmp_factor, left_index=True, right_index=True, how='inner')

    for date, factors in multi_factors.items():
        print(date)
        factors_ortho = symmetry_ortho(factors)

        ortho_path = os.path.join(multi_factors_path, str(rebal_freq), 'preprocessed', date[:4], date[5:7])
        if not os.path.exists(ortho_path):
            os.makedirs(ortho_path)
        factors_ortho.to_csv(os.path.join(ortho_path, date+'.csv'))
