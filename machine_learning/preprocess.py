import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import os, pdb


for root, dirs, files in os.walk(r'E:\data\machine_learning\dataset'):

    for file in [x for x in files if x.endswith('.csv')]:

        print(file)
        mdata = pd.read_csv(os.path.join(root, file), index_col=0)
        mdata = mdata.replace(0, np.nan)

        mdata['MidPrice'] = ((mdata['AskPrice1'] + mdata['BidPrice1']) / 2)
        mdata['MidPrice'] = np.log(mdata['MidPrice']).diff()
        
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 'A']:
            #mdata['Wap%s'%i] = ((mdata['BidPrice%s'%i] * mdata['AskVolume%s'%i]) +
                                 #(mdata['AskPrice%s'%i] * mdata['BidVolume%s'%i])) /\
                                    #(mdata['AskVolume%s'%i] + mdata['BidVolume%s'%i])
            #mdata['Wap%s'%i] = np.log(mdata['Wap%s'%i]).diff()

            '''
            mdata['AskPriceSpread%s'%i] = (mdata['AskPrice%s'%i] - mdata['MidPrice']) / mdata['MidPrice']
            mdata['BidPriceSpread%s'%i] = (mdata['BidPrice%s'%i] - mdata['MidPrice']) / mdata['MidPrice']

            mdata['AskVolumeSpread%s'%i] = (mdata['AskVolume%s'%i] - mdata['MidVolume']) / mdata['MidVolume']
            mdata['BidVolumeSpread%s'%i] = (mdata['BidVolume%s'%i] - mdata['MidVolume']) / mdata['MidVolume']

            mdata['AskPriceChange%s'%i] = mdata['AskPrice%s'%i].diff() / mdata['MidPrice']
            mdata['BidPriceChange%s'%i] = mdata['BidPrice%s'%i].diff() / mdata['MidPrice']

            mdata['AskVolumeChange%s'%i] = mdata['AskVolume%s'%i].diff() / mdata['MidVolume']
            mdata['BidVolumeChange%s'%i] = mdata['BidVolume%s'%i].diff() / mdata['MidVolume']
            '''

            mdata['AskPrice%s'%i] = np.log(mdata['AskPrice%s'%i]).diff()
            mdata['BidPrice%s'%i] = np.log(mdata['BidPrice%s'%i]).diff()

        mdata.dropna(inplace=True)
        save_path = os.path.join(r'E:\data\machine_learning\preprocessed', root.split('\\')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mdata.iloc[:, 1:].to_csv(os.path.join(save_path, file))
