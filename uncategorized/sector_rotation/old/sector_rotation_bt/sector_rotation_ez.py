import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
from matplotlib import pyplot as plt

import copy
import datetime
import os, pdb

import empyrical
import akshare as ak

etf = {
    '公用事业':'sz159611',
    '家用电器':'sz159996',
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
    '机械设备':'sz159886',
    '交通运输':'sh561320',
    '有色金属':'sz159881'
}

def get_index():
    c300 = ak.stock_zh_index_daily(symbol="sh000300")
    c500 = ak.stock_zh_index_daily(symbol="sh000905")
    c300['date'] = c300['date'].astype(str)
    c500['date'] = c500['date'].astype(str)
    return c300, c500

def get_info():
    indus_ratio_path = '/Users/ydgan/Documents/sector_rotation/indus_ratio'
    indus_ratio = pd.read_csv(os.path.join(indus_ratio_path,
                                            '20_days_industry_ratio.csv'), index_col=0)
    indus_ratio_chg = indus_ratio.rolling(10).mean().pct_change()

    l1 = ak.sw_index_first_info()
    indus = l1[['行业名称', '行业代码']]

    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    indus = indus[indus['行业名称'].isin(findus['indus1'].unique())]
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])
    indus['etf'] = indus['行业名称'].map(etf)
    indus = indus.set_index('行业名称')

    return indus_ratio_chg, indus

def get_sig(path, date):
    sig = pd.read_csv(os.path.join(path, date+'.csv'), index_col=0)
    sig = sig[sig['signal'] > 0]
    return sig

def xbuy(path, code, date, sprice, cash, buy_list, position):
    bindex = pd.read_csv(os.path.join(path, code+'.csv'), index_col=0)
    buy_index = bindex[bindex['日期'] <= date]
    if buy_index.empty:
        buy_list.remove(code)
        return cash
    
    share = sprice // buy_index['开盘'].iloc[-1]

    print('Buy %s Price %s'%(code, buy_index['开盘'].iloc[-1]))
    cash -= buy_index['开盘'].iloc[-1] * share * 1.00005
    if not code in position.keys():
        position[code] = dict()
        position[code]['hold_days'] = [0]
        position[code]['buy_price'] = [buy_index['开盘'].iloc[-1]]
        position[code]['buy_shares'] = [share]
    else:
        position[code]['hold_days'] += [0]
        position[code]['buy_price'] += [buy_index['开盘'].iloc[-1]]
        position[code]['buy_shares'] += [share]
    
    buy_list.remove(code)
    return cash

def get_blist(signal, indus, buy_list):
    for name in signal['name']:
        code = indus.loc[name,'行业代码']
        buy_list += [code]
    return buy_list

def hvalue(md_path, cdate, position):
    avalue = 0
    for acode, ahold in position.items():
        aindex = pd.read_csv(os.path.join(md_path, acode+'.csv'), index_col=0)
        aindex = aindex[aindex['日期'] >= cdate]
        for xhold in ahold['buy_shares']:
            avalue += xhold * aindex['收盘'].iloc[0]
    return avalue

def xsell(index, idx, code, shares, cash, position):
    print('Sell %s Price %s'%(code, index['开盘'].iloc[-1]))
    cash += index['开盘'].iloc[-1] * shares * 0.99995
    position[code]['hold_days'][idx] = -1
    position[code]['buy_price'][idx] = -1
    position[code]['buy_shares'][idx] = -1
    return cash

def get_md(indus):
    path = '/Users/ydgan/Documents/backtesting/sector_rotation/marketdata'
    for row in indus.itertuples():
        print(row[0])
        code = row[2]
        etf = ak.fund_etf_hist_em(code[2:], adjust='hfq')
        etf.to_csv(os.path.join(path, row[1]+'.csv'))
    
if __name__ == '__main__':
    start_date = pd.to_datetime('2022-01-01').date()
    end_date = pd.to_datetime('2023-08-25').date()

    c300, c500 = get_index()
    indus_ratio_chg, indus = get_info()
    #get_md(indus)

    sig_path = '/Users/ydgan/Documents/sector_rotation/signal/sig'
    md_path = '/Users/ydgan/Documents/backtesting/sector_rotation/marketdata'

    hold_days = [1, 4, 13]
    cash = 15050000
    position = dict()
    buy_list = []
    date_list = []
    asset_value = []
    for i in range((end_date - start_date).days + 1):
        cdate = start_date + datetime.timedelta(days=i)
        cdate = cdate.strftime('%Y-%m-%d')

        sig_path_tmp = os.path.join(sig_path, cdate.split('-')[0], cdate.split('-')[1])
        if not os.path.exists(os.path.join(sig_path_tmp, cdate+'.csv')):
            continue
        sig = get_sig(sig_path_tmp, cdate)
        print(cdate)

        if position:
            for hcode in position.keys():
                hindex = pd.read_csv(os.path.join(md_path, hcode+'.csv'), index_col=0)
                hold_index = hindex[hindex['日期'] <= cdate]

                position[hcode]['hold_days'] = [x+1 for x in position[hcode]['hold_days']]
                for i, (hdays, hprice, hshares) in enumerate(zip(position[hcode]['hold_days'],
                                                                position[hcode]['buy_price'],
                                                                    position[hcode]['buy_shares'])):
                    t300 = c300[c300['date'] < cdate]
                    t500 = c500[c500['date'] < cdate]
                    tmp300 = (t300['close'] / t300['open'].shift(hdays-1)).iloc[-1]
                    tmp500 = (t500['close'] / t500['open'].shift(hdays-1)).iloc[-1]
                    tmphold = hold_index['收盘'].iloc[-2] / hprice

                    indus_name = indus[indus['行业代码'] == hcode].index[0]
                    idc = indus_ratio_chg[indus_ratio_chg.index < cdate][indus_name].iloc[-1]

                    if hdays == hold_days[0]:
                        if tmphold < 1 and tmphold < tmp300 and tmphold < tmp500:
                            cash = xsell(hold_index, i, hcode, hshares, cash, position)
                    elif hdays == hold_days[1]:
                        if tmphold > 1 and tmphold > tmp300 and tmphold > tmp500:
                            if idc < 0:
                                cash = xsell(hold_index, i, hcode, hshares, cash, position)
                        else:
                            cash = xsell(hold_index, i, hcode, hshares, cash, position)
                    #elif hdays > hold_days[1]:
                        #if idc < 0:
                            #cash = xsell(hold_index, i, hcode, hshares, cash, position)
                        #elif tmphold < tmp300:
                            #cash = xsell(hold_index, i, hcode, hshares, cash, position)
                    elif hdays == hold_days[2]:
                        cash = xsell(hold_index, i, hcode, hshares, cash, position)


            posi_tmp = copy.deepcopy(position)
            for hcode in posi_tmp.keys():
                for hd in posi_tmp[hcode]['hold_days']:
                    if hd == -1:
                        position[hcode]['hold_days'].remove(-1)
                        position[hcode]['buy_price'].remove(-1)
                        position[hcode]['buy_shares'].remove(-1)

                if len(position[hcode]['hold_days']) == 0:
                    del position[hcode]
        
        if buy_list:
            if cash > 50000:
                sprice = (cash - 50000) // len(buy_list)
                if sprice > 1000000:
                    sprice = 1000000

                blist = buy_list.copy()
                for bcode in blist:
                    cash = xbuy(md_path, bcode, cdate, sprice, cash, buy_list, position)
                assert not bool(buy_list)
            else:
                buy_list = []
        buy_list = get_blist(sig, indus, buy_list)

        print(cash)
        date_list.append(cdate)
        asset_value.append((cash + hvalue(md_path, cdate, position)) / 15050000)

    asset = pd.DataFrame({'date':date_list, 'net asset value':asset_value})
    max_drawdown = empyrical.max_drawdown(asset['net asset value'].pct_change())
    sratio = empyrical.sharpe_ratio(asset['net asset value'].pct_change(), risk_free=(pow(1.03, 1/365)-1), annualization=250)
    print('max drawdown %s'%max_drawdown)
    print('sharpe ratio %s'%sratio)

    c300nav = c300[(c300['date'] >= start_date.strftime('%Y-%m-%d')) &
                    (c300['date'] <= end_date.strftime('%Y-%m-%d'))]
    c500nav = c500[(c500['date'] >= start_date.strftime('%Y-%m-%d')) &
                    (c500['date'] <= end_date.strftime('%Y-%m-%d'))]
    c300nav['300nav'] = c300nav['close'] / c300nav['close'].iloc[0]
    c500nav['500nav'] = c500nav['close'] / c500nav['close'].iloc[0]
    asset = pd.merge(asset, c300nav[['date', '300nav']], on='date', how='left')
    asset = pd.merge(asset, c500nav[['date', '500nav']], on='date', how='left')
    asset.set_index('date', inplace=True)

    asset.plot(figsize=(16,8))
    plt.savefig('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.png'%(start_date, end_date), bbox_inches='tight')
    asset.to_csv('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.csv'%(start_date, end_date))
    print('done^^')
