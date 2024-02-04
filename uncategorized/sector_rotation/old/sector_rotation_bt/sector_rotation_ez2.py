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

    etf_map = dict()
    indus_path = '/Users/ydgan/Documents/sector_rotation/sw_industry'
    findus = pd.read_csv(os.path.join(indus_path, 'sw_industry.csv'), index_col=0, dtype=object)
    for ind, etf in zip(findus['indus1'].unique(), findus['etf'].unique()):
        etf_map[ind] = etf

    indus['etf'] = indus['行业名称'].map(etf_map)
    indus = indus.dropna(subset='etf')
    indus['行业代码'] = indus['行业代码'].apply(lambda x:x[:-3])
    indus = indus.set_index('行业名称')
    return indus_ratio_chg, indus

def get_sig(path, date):
    sig = pd.read_csv(os.path.join(path, date+'.csv'), index_col=0)
    sig = sig[sig['signal'] > 0]
    return sig

def get_blist(path, date, signal, indus, buy_list):
    for name in signal['name']:
        code = indus.loc[name,'行业代码']
        bindex = pd.read_csv(os.path.join(path, code+'.csv'), index_col=0)
        buy_index = bindex[bindex['日期'] <= date]
        if buy_index.empty:
            continue

        if not code in buy_list.keys():
            buy_list[code] = dict()
            buy_list[code]['ob_days'] = []
            buy_list[code]['ob_price'] = []
        buy_list[code]['ob_days'] += [0]
        buy_list[code]['ob_price'] += [buy_index['开盘'].iloc[-1]]
    return buy_list

def hvalue(md_path, cdate, position):
    avalue = 0
    for acode, ahold in position.items():
        aindex = pd.read_csv(os.path.join(md_path, acode+'.csv'), index_col=0)
        aindex = aindex[aindex['日期'] >= cdate]
        for xhold in ahold['buy_shares']:
            avalue += xhold * aindex['收盘'].iloc[0]
    return avalue

def get_md(indus):
    path = '/Users/ydgan/Documents/backtesting/sector_rotation/marketdata'
    for row in indus.itertuples():
        print(row[0])
        code = row[2]
        etf = ak.fund_etf_hist_em(code[2:], adjust='hfq')
        etf.to_csv(os.path.join(path, row[1]+'.csv'))
    
def xbuy(path, date, buy_list, buy_listt, c300, c500, hold_days):
    for hcode in buy_list.keys():
        hindex = pd.read_csv(os.path.join(path, hcode+'.csv'), index_col=0)
        hold_index = hindex[hindex['日期'] <= date]

        buy_list[hcode]['ob_days'] = [x+1 for x in buy_list[hcode]['ob_days']]
        for i, (hdays, hprice) in enumerate(zip(buy_list[hcode]['ob_days'],
                                                        buy_list[hcode]['ob_price'])):
            if hdays > hold_days[0]:
                buy_listt.append(hcode)
                buy_list[hcode]['ob_days'][i] = -1
                buy_list[hcode]['ob_price'][i] = -1

    bl_tmp = copy.deepcopy(buy_list)
    for bcode in bl_tmp.keys():
        for bl in bl_tmp[bcode]['ob_days']:
            if bl == -1:
                buy_list[bcode]['ob_days'].remove(-1)
                buy_list[bcode]['ob_price'].remove(-1)
                
        if len(buy_list[bcode]['ob_days']) == 0:
            del buy_list[bcode]
    return buy_list, buy_listt

def xsell(path, date, position, cash, c300, c500, hold_days):
    for hcode in position.keys():
        hindex = pd.read_csv(os.path.join(path, hcode+'.csv'), index_col=0)
        hold_index = hindex[hindex['日期'] <= date]

        position[hcode]['hold_days'] = [x+1 for x in position[hcode]['hold_days']]
        for i, (hdays, hprice, hshares) in enumerate(zip(position[hcode]['hold_days'],
                                                        position[hcode]['buy_price'],
                                                            position[hcode]['buy_shares'])):
            t300 = c300[c300['date'] < date]
            t500 = c500[c500['date'] < date]
            tmp300 = (t300['close'] / t300['open'].shift(hdays-1)).iloc[-1]
            tmp500 = (t500['close'] / t500['open'].shift(hdays-1)).iloc[-1]
            tmphold = hold_index['收盘'].iloc[-2] / hprice

            indus_name = indus[indus['行业代码'] == hcode].index[0]
            idc = indus_ratio_chg[indus_ratio_chg.index < date][indus_name].iloc[-1]
            
            if hdays == hold_days[1]:
                if tmphold < 1 and tmphold < tmp300 and tmphold < tmp500:
                    print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                    cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                    position[hcode]['hold_days'][i] = -1
                    position[hcode]['buy_price'][i] = -1
                    position[hcode]['buy_shares'][i] = -1
            elif hdays == hold_days[2]:
                if tmphold > 1 and tmphold > tmp300 and tmphold > tmp500:
                    if idc < 0:
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
            elif hdays > hold_days[2]:
                if (hold_index['收盘'].iloc[-2] / hold_index['收盘'].iloc[-3]) \
                    < (t300['close'].iloc[-1] / t300['close'].iloc[-2]):
                    print('Sell %s Price %s'%(hcode, hold_index['开盘'].iloc[-1]))
                    cash += hold_index['开盘'].iloc[-1] * hshares * 0.99995
                    position[hcode]['hold_days'][i] = -1
                    position[hcode]['buy_price'][i] = -1
                    position[hcode]['buy_shares'][i] = -1

    posi_tmp = copy.deepcopy(position)
    for thcode in posi_tmp.keys():
        for hd in posi_tmp[thcode]['hold_days']:
            if hd == -1:
                position[thcode]['hold_days'].remove(-1)
                position[thcode]['buy_price'].remove(-1)
                position[thcode]['buy_shares'].remove(-1)

        if len(position[thcode]['hold_days']) == 0:
            del position[thcode]
    return cash, position

if __name__ == '__main__':
    start_date = pd.to_datetime('2022-01-01').date()
    end_date = pd.to_datetime('2022-12-31').date()

    c300, c500 = get_index()
    indus_ratio_chg, indus = get_info()
    #get_md(indus)

    sig_path = '/Users/ydgan/Documents/sector_rotation/signal/sig'
    md_path = '/Users/ydgan/Documents/backtesting/sector_rotation/marketdata'

    hold_days = [0, 10, 20]
    cash = 15050000
    sig = pd.DataFrame()
    position = dict()
    buy_list = dict()
    date_list = []
    asset_value = []
    for i in range((end_date - start_date).days + 1):
        cdate = start_date + datetime.timedelta(days=i)
        cdate = cdate.strftime('%Y-%m-%d')

        sig_path_tmp = os.path.join(sig_path, cdate.split('-')[0], cdate.split('-')[1])
        if not os.path.exists(os.path.join(sig_path_tmp, cdate+'.csv')):
            continue
        print(cdate)
        if position:
            cash, position = xsell(md_path, cdate, position, cash, c300, c500, hold_days)

        if buy_list:
            buy_listt = []
            buy_list, buy_listt = xbuy(md_path, cdate, buy_list, buy_listt, c300, c500, hold_days)
            if buy_listt:
                for bc in buy_listt:
                    if cash > 1000000:
                        sprice = 1000000

                        bindex = pd.read_csv(os.path.join(md_path, bc+'.csv'), index_col=0)
                        buy_index = bindex[bindex['日期'] <= cdate]

                        print('Buy %s Price %s'%(bc, buy_index['开盘'].iloc[-1]))
                        share = sprice // buy_index['开盘'].iloc[-1]
                        cash -= buy_index['开盘'].iloc[-1] * share * 1.00005
                        if not bc in position.keys():
                            position[bc] = dict()
                            position[bc]['hold_days'] = []
                            position[bc]['buy_price'] = []
                            position[bc]['buy_shares'] = []
                        position[bc]['hold_days'] += [0]
                        position[bc]['buy_price'] += [buy_index['开盘'].iloc[-1]]
                        position[bc]['buy_shares'] += [share]
                    else:
                        continue
                buy_listt = []
            
        if not sig.empty:
            buy_list = get_blist(md_path, cdate, sig, indus, buy_list)

        sig = get_sig(sig_path_tmp, cdate)

        print(cash)
        date_list.append(cdate)
        asset_value.append((cash + hvalue(md_path, cdate, position)) / 15050000)

    asset = pd.DataFrame({'date':date_list, 'net asset value':asset_value})
    c300nav = c300[(c300['date'] >= start_date.strftime('%Y-%m-%d')) &
                    (c300['date'] <= end_date.strftime('%Y-%m-%d'))]
    c300nav['300nav'] = c300nav['close'] / c300nav['close'].iloc[0]
    asset = pd.merge(asset, c300nav[['date', '300nav']], on='date', how='left')
    asset.set_index('date', inplace=True)

    max_drawdown = empyrical.max_drawdown(asset['net asset value'].pct_change())
    sratio = empyrical.sharpe_ratio(asset['net asset value'].pct_change(), risk_free=(pow(1.03, 1/365)-1), annualization=250)
    annual_return = empyrical.annual_return(asset['net asset value'].pct_change(), annualization=250)
    annual_excess = pow(asset['net asset value'].iloc[-1] / asset['300nav'].iloc[-1], 365/asset.shape[0]) - 1

    print('annual return %s'%annual_return)
    print('annual excess %s'%annual_excess)
    print('sharpe ratio %s'%sratio)
    print('max drawdown %s'%max_drawdown)

    asset.plot(figsize=(16,8))
    plt.savefig('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.png'%(start_date, end_date), bbox_inches='tight')
    asset.to_csv('/Users/ydgan/Documents/backtesting/sector_rotation/result/%s-%s.csv'%(start_date, end_date))
    print('done^^')
