import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
from matplotlib import pyplot as plt

import copy
import datetime
import os, pdb


import akshare as ak

etf = {
    '公用事业':'sh561170',
    '家用电器':'sh561120',
    '建筑装饰':'sh516970',
    '建筑材料':'sh516750',
    '电力设备':'sh516160',
    '基础化工':'sh516120',
    '汽车':'sh516110',
    '通信':'sh515880',
    '非银金融':'sh515850',
    '银行':'sh515280',
    '电子':'sh515260',
    '煤炭':'sh515220',
    '钢铁':'sh515210',
    '国防军工':'sh512710',
    '房地产':'sh512200',
    '计算机':'sz159998',
    '石油石化':'sz159930',
    '医药生物':'sz159929',
    '环保':'sz159861',
    '农林牧渔':'sz159825',
    '传媒':'sz159805',
    '社会服务':'sz159766',
    '食品饮料':'sz159736',
    '机械设备':'sz159667',
    '交通运输':'sz159662',
    '有色金属':'sz159652'
}

csi300 = ak.stock_zh_index_daily(symbol="sh000300")
csiscp = ak.stock_zh_index_daily(symbol="sh000905")
c300 = csi300[['date', 'close']]
c500 = csiscp[['date', 'close']]
c300['date'] = c300['date'].astype(str)
c500['date'] = c500['date'].astype(str)
c300['3days'] = c300['close'] / c300['close'].shift(3)
c300['5days'] = c300['close'] / c300['close'].shift(5)
c500['3days'] = c500['close'] / c500['close'].shift(3)
c500['5days'] = c500['close'] / c500['close'].shift(5)

indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                        '20_days_industry_ratio.csv'), index_col=0)
indus_ratio_chg = indus_ratio.rolling(10).mean().pct_change()

start_date = pd.to_datetime('2023-03-16').date()
end_date = pd.to_datetime('2023-08-25').date()

l1 = ak.sw_index_first_info()
indus = l1[['行业名称', '行业代码']]

indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])
indus['etf'] = indus['行业名称'].map(etf)
indus = indus.set_index('行业名称')

sig_path = '/Users/ydgan/Documents/sector_rotation/signal'
md_path = '/Users/ydgan/Documents/backtesting/sector_rotation/marketdata'

position = dict()
buy_list = []
cash = 10000000
date_list = []
asset_value = []
for i in range((end_date - start_date).days + 1):
    cdate = start_date + datetime.timedelta(days=i)
    cdate = cdate.strftime('%Y-%m-%d')

    sig_path_5 = os.path.join(sig_path, 'sig_5', cdate.split('-')[0], cdate.split('-')[1])
    sig_path_20 = os.path.join(sig_path, 'sig_20', cdate.split('-')[0], cdate.split('-')[1])
    if not os.path.exists(os.path.join(sig_path_5, cdate+'.csv')):
        continue
    if not os.path.exists(os.path.join(sig_path_20, cdate+'.csv')):
        continue
    print(cdate)

    if len(position) > 0:
        for hcode in position.keys():
            hindex = pd.read_csv(os.path.join(md_path, hcode+'.csv'), index_col=0)
            hold_index = hindex[hindex['日期'] <= cdate]

            position[hcode]['hold_days'] = [x+1 for x in position[hcode]['hold_days']]
            for i, (hdays, hprice, hshares) in enumerate(zip(position[hcode]['hold_days'],
                                                              position[hcode]['buy_price'],
                                                                position[hcode]['buy_shares'])):
                if hdays == 3:
                    c300_tmp = c300[c300['date'] <= cdate]
                    c500_tmp = c500[c500['date'] <= cdate]
                    c300_3days = c300_tmp['3days'].iloc[-2]
                    c500_3days = c500_tmp['3days'].iloc[-2]

                    hold_3days = hold_index['收盘'].iloc[-2] / hprice
                    if hold_3days < 1 and hold_3days < c300_3days and hold_3days < c500_3days:
                        print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                        cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                        position[hcode]['hold_days'][i] = -1
                        position[hcode]['buy_price'][i] = -1
                        position[hcode]['buy_shares'][i] = -1
                elif hdays == 5:
                    c300_tmp = c300[c300['date'] <= cdate]
                    c500_tmp = c500[c500['date'] <= cdate]
                    c300_5days = c300_tmp['5days'].iloc[-2]
                    c500_5days = c500_tmp['5days'].iloc[-2]

                    hold_5days = hold_index['收盘'].iloc[-2] / hprice
                    if hold_5days > 1 and hold_5days > c300_5days and hold_5days > c500_5days:
                        indus_name = indus[indus['行业代码'] == hcode].index[0]
                        if indus_ratio_chg.loc[cdate, indus_name] < 0:
                            print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                            cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                            position[hcode]['hold_days'][i] = -1
                            position[hcode]['buy_price'][i] = -1
                            position[hcode]['buy_shares'][i] = -1
                    else:
                        print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                        cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                        position[hcode]['hold_days'][i] = -1
                        position[hcode]['buy_price'][i] = -1
                        position[hcode]['buy_shares'][i] = -1
                elif hdays > 5:
                    indus_name = indus[indus['行业代码'] == hcode].index[0]
                    if indus_ratio_chg.loc[cdate, indus_name] < 0:
                        print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                        cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                        position[hcode]['hold_days'][i] = -1
                        position[hcode]['buy_price'][i] = -1
                        position[hcode]['buy_shares'][i] = -1
                    else:
                        c300_tmp = c300[c300['date'] <= cdate]
                        c300_tmp['hold_days'] = c300_tmp['close'] / c300_tmp['close'].shift(hdays)
                        if hold_index['收盘'].iloc[-2] / hprice <= c300_tmp['hold_days'].iloc[-2]:
                            print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                            cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                            position[hcode]['hold_days'][i] = -1
                            position[hcode]['buy_price'][i] = -1
                            position[hcode]['buy_shares'][i] = -1

        posi_tmp = copy.deepcopy(position)
        for hcode in posi_tmp.keys():
            for hd in posi_tmp[hcode]['hold_days']:
                if hd == -1:
                    position[hcode]['hold_days'].remove(-1)
                    position[hcode]['buy_price'].remove(-1)
                    position[hcode]['buy_shares'].remove(-1)

            if len(position[hcode]['hold_days']) == 0:
                del position[hcode]

    if len(buy_list) > 0:
        blist = buy_list.copy()
        for bcode in blist:
            bindex = pd.read_csv(os.path.join(md_path, bcode+'.csv'), index_col=0)
            buy_index = bindex[bindex['日期'] >= cdate]     
            share = sprice // buy_index['开盘'].iloc[0]

            print('Buy %s Price %s'%(bcode, buy_index['开盘'].iloc[0]))
            cash -= buy_index['开盘'].iloc[0] * share * 1.00005
            if not bcode in position.keys():
                position[bcode] = dict()
                position[bcode]['hold_days'] = [0]
                position[bcode]['buy_price'] = [buy_index['开盘'].iloc[0]]
                position[bcode]['buy_shares'] = [share]
            else:
                position[bcode]['hold_days'] += [0]
                position[bcode]['buy_price'] += [buy_index['开盘'].iloc[0]]
                position[bcode]['buy_shares'] += [share]
            
            buy_list.remove(bcode)

    sig_5 = pd.read_csv(os.path.join(sig_path_5, cdate+'.csv'), index_col=0)
    sig_20 = pd.read_csv(os.path.join(sig_path_20, cdate+'.csv'), index_col=0)
    sig = sig_20[sig_20['signal'] >= 103]
    #sig = pd.concat([sig_5[sig_5['signal']>=103], sig_20[sig_20['signal']>=103]])
    #sig = sig.drop_duplicates(subset='name')

    for name in sig['name']:
        code = indus.loc[name,'行业代码']
        buy_list += [code]
    
    if len(buy_list) > 0:
        sprice = cash / len(buy_list)
        if sprice > 500000:
            sprice = 500000

        #if cash < 100000 or sprice < 50000:
            #buy_list = []

    print(cash)
    date_list.append(cdate)
    if len(position) > 0:
        avalue = 0
        for acode, ahold in position.items():
            aindex = pd.read_csv(os.path.join(md_path, acode+'.csv'), index_col=0)
            aindex = aindex[aindex['日期'] >= cdate]
            for xhold in ahold['buy_shares']:
                avalue += xhold * aindex['收盘'].iloc[0]
        asset_value.append((cash + avalue) / 10000000)
    else:    
        asset_value.append(cash / 10000000)

asset = pd.DataFrame({'date':date_list, 'net asset value':asset_value})
c300nav = c300[(c300['date'] >= start_date.strftime('%Y-%m-%d')) & (c300['date'] <= end_date.strftime('%Y-%m-%d'))]
c500nav = c500[(c500['date'] >= start_date.strftime('%Y-%m-%d')) & (c500['date'] <= end_date.strftime('%Y-%m-%d'))]
c300nav['300nav'] = c300nav['close'] / c300nav['close'].iloc[0]
c500nav['500nav'] = c500nav['close'] / c500nav['close'].iloc[0]
asset = pd.merge(asset, c300nav[['date', '300nav']], on='date', how='left')
asset = pd.merge(asset, c500nav[['date', '500nav']], on='date', how='left')
asset.set_index('date', inplace=True)

asset.plot(figsize=(16,8))
plt.savefig('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.png'%(start_date, end_date), bbox_inches='tight')
asset.to_csv('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.csv'%(start_date, end_date))
print('done^^')


'''
if len(position) > 0:
        hposition = position.copy()
        for hcode in hposition.keys():
            hindex = pd.read_csv(os.path.join(md_path, hcode+'.csv'), index_col=0)
            hold_index = hindex[hindex['日期'] < cdate]

            hold_price = hposition[hcode]['buy_price'].copy()
            hold_shares = hposition[hcode]['buy_shares'].copy()
            for hprice, hshares in zip(hold_price, hold_shares):
                if hold_index['收盘'].iloc[-1] < hprice:
                    if not hcode in buy_list:
                        print('Sell %s Price %s'%(hcode, hindex[hindex['日期']==cdate]['开盘'].iloc[0]))
                        cash += hindex[hindex['日期']==cdate]['开盘'].iloc[0] * hshares * 0.99995
                        position[hcode]['buy_price'].remove(hprice)
                        position[hcode]['buy_shares'].remove(hshares)
                elif hold_index['收盘'].iloc[-1] > hprice:
                    position[hcode]['buy_price'] = [hold_index['收盘'].iloc[-1]] * len(position[hcode]['buy_price'])

            if len(position[hcode]['buy_price']) == 0:
                del position[hcode]

    if cash < 500000:
        continue

    if len(buy_list) > 0:
        blist = buy_list.copy()
        for bcode in blist:
            bindex = pd.read_csv(os.path.join(md_path, bcode+'.csv'), index_col=0)
            buy_index = bindex[bindex['日期'] >= cdate]     
            share = sprice // buy_index['开盘'].iloc[0]

            print('Buy %s Price %s'%(bcode, buy_index['开盘'].iloc[0]))
            cash -= buy_index['开盘'].iloc[0] * share * 1.00005
            if not bcode in position.keys():
                position[bcode] = dict()
                position[bcode]['hold_days'] = [1]
                position[bcode]['buy_price'] = [buy_index['开盘'].iloc[0]]
                position[bcode]['buy_shares'] = [share]
            else:
                position[bcode]['hold_days'] += [1]
                if max(position[bcode]['buy_price']) >= buy_index['开盘'].iloc[0]:
                    position[bcode]['buy_price'] = [max(position[bcode]['buy_price'])] * (len(position[bcode]['buy_price']) + 1)
                else:
                    position[bcode]['buy_price'] = [buy_index['开盘'].iloc[0]] * (len(position[bcode]['buy_price']) + 1)
                position[bcode]['buy_shares'] += [share]
            
            buy_list.remove(bcode)




for code,hold in position.items():
    cindex = pd.read_csv(os.path.join(md_path, code+'.csv'), index_col=0)
    cindex = cindex[cindex['日期'] >= cdate]
    for xhold in hold['buy_shares']:
        cash += xhold * cindex['收盘'].iloc[0]
print(cash)
'''