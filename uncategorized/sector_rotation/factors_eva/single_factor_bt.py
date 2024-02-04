import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.stats as stats

import akshare as ak
import empyrical

import os, pdb
import copy
import datetime


def get_nav(date, cash, position, industry_index):
    hold_values = 0
    for indus_code, share in position.items():
        if not indus_code in industry_index.keys():
            industry_index[indus_code] = pd.read_csv(os.path.join(r'E:\data\sw_industry_index', indus_code+'.csv'), index_col=0)
        sub_index = industry_index[indus_code][industry_index[indus_code].index <= date]

        hold_values += share * sub_index['close'].iloc[-1]
    return (cash + hold_values) / 10000000, hold_values, industry_index

def get_index():
    #i300 = ak.stock_zh_index_daily(symbol='sh000300')
    i500 = ak.stock_zh_index_daily(symbol='sh000905')

    #i300['date'] = i300['date'].astype(str)
    i500['date'] = i500['date'].astype(str)
    return i500

def xbuy(date, group, buy_list, sub_cash, cash, position, industry_index):
    buy_amount = 0
    for indus_code in buy_list:
        try:
            if not indus_code in industry_index.keys():
                industry_index[indus_code] = pd.read_csv(os.path.join(r'E:\data\sw_industry_index', indus_code+'.csv'), index_col=0)
            sub_index = industry_index[indus_code][industry_index[indus_code].index <= date]

            share = sub_cash // sub_index['open'].iloc[-1]
            buy_amount += sub_index['open'].iloc[-1] * share
            cash[group] -= sub_index['open'].iloc[-1] * share * 1.00005
            print('Buy %s shares %s, remaining cash %s'%(share, indus_code, cash[group]))
            position[group][indus_code] = share
        except:
            continue
    return buy_amount, industry_index

def xsell(date, group, sell_list, cash, position, industry_index):
    sell_amount = 0
    for indus_code in sell_list:
        if not indus_code in industry_index.keys():
            industry_index[indus_code] = pd.read_csv(os.path.join(r'E:\data\sw_industry_index', indus_code+'.csv'), index_col=0)
        sub_index = industry_index[indus_code][industry_index[indus_code].index <= date]

        sell_amount += sub_index['open'].iloc[-1] * position[group][indus_code]
        cash[group] += sub_index['open'].iloc[-1] * position[group][indus_code] * 0.99995
        print('Sell %s shares %s, remaining cash %s'%(position[group][indus_code], indus_code, cash[group]))
        del position[group][indus_code]
    return sell_amount, industry_index

def get_traday(freq=5):
    is_traday = ak.tool_trade_date_hist_sina()
    is_traday['trade_date'] = pd.to_datetime(is_traday['trade_date'])
    is_traday['trade_date'] = is_traday['trade_date'].apply(
        lambda x:x.strftime('%Y-%m-%d')
    )

    rebal_date = is_traday.iloc[np.arange(0, is_traday.shape[0], freq)].reset_index(drop=True).rename(columns={'trade_date':'rebalance'})
    retra_date = is_traday.iloc[np.arange(1, is_traday.shape[0], freq)].reset_index(drop=True).rename(columns={'trade_date':'retrade'})
    return is_traday, pd.concat([rebal_date, retra_date], axis=1)

def cal_return(md_path, file, resample_date, queue):
    data = pd.read_csv(os.path.join(md_path, file), index_col=0)
    data = data[data.index.isin(resample_date['retrade'])].sort_index()

    data[file[:-4]] = data['open'].pct_change().shift(-1)
    queue.put(data[file[:-4]])

def ret_concat(queue):
    ret = list()
    while True:
        if not queue.empty():
            rec = queue.get(True)
            ret.append(rec)
        else:
            return pd.concat(ret, axis=1).sort_index().dropna(axis=0, how='all')

def get_return(path, resample_date):
    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool()
    for file in os.listdir(path):
        p.apply_async(cal_return, (path, file, resample_date, q,))
    p.close()
    p.join()

    resample_ret = ret_concat(q)
    resample_ret = pd.merge(resample_date, resample_ret, left_on='retrade', right_index=True, how='right')
    resample_ret.set_index('rebalance', inplace=True)
    resample_ret.drop(columns=['retrade'], inplace=True)
    return resample_ret


if __name__ == '__main__':
    rebal_freq = 3
    stock_path = r'E:\data\stock'
    industry_index_path = r'E:\data\sw_industry_index'
    fac_path = r'E:\sector_rotation\preprocessed'

    start_date = pd.to_datetime('2018-01-01')
    end_date = pd.to_datetime('2023-12-31')

    rep_path = os.path.join(r'E:\sector_rotation\single_factor', str(rebal_freq), 'backtesting')
    if not os.path.exists(rep_path):
        os.makedirs(rep_path)

    if not os.path.exists(os.path.join(rep_path, '%s-%s-Rep.csv'%(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))):
        rep = pd.DataFrame()
    else:
        rep = pd.read_csv(os.path.join(rep_path, '%s-%s-Rep.csv'%(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))))

    sw_index = ak.sw_index_first_info()
    sw_index['行业代码'] = sw_index['行业代码'].str[:-3]

    is_traday, resample_date = get_traday(freq=rebal_freq)
    resample_ret = get_return(industry_index_path, resample_date)
    index_500 = get_index()

    industry_dict = dict()
    industry_index_dict = dict()
    for factor in os.listdir(fac_path):

        print(factor)

        if 'Factor' in rep.columns and factor in rep['Factor'].values:
            continue

        nav = dict()
        cash = dict()
        position = dict()
        turnover = dict()
        ic_list = list()

        group_sig = dict()
        buy_dict = dict()
        sell_dict = dict()

        for i in range((end_date - start_date).days + 1):
            date = start_date + datetime.timedelta(days=i)
            date = date.strftime('%Y-%m-%d')
            if not date in is_traday['trade_date'].values:
                continue
            print(date)

            tov = 0
            tov_flag = 0

            gsig = copy.deepcopy(group_sig)
            for group, signal in gsig.items():
                del group_sig[group]

                if not group in position.keys():
                    position[group] = dict()
                    turnover[group] = []
                buy_dict[group] = [x for x in signal if not x in position[group].keys()]
                sell_dict[group] = [x for x in position[group].keys() if not x in signal]

            sdict = copy.deepcopy(sell_dict)
            for group, sell_list in sdict.items():
                del sell_dict[group]
                if len(sell_list) == 0:
                    continue

                sell_amount, industry_index_dict = xsell(date, group, sell_list, cash, position, industry_index_dict)
                tov += sell_amount
                tov_flag += 1

            bdict = copy.deepcopy(buy_dict)
            for group, buy_list in bdict.items():
                del buy_dict[group]
                if len(buy_list) == 0:
                    continue

                if not group in cash.keys():
                    cash[group] = 10000000
                sub_cash = (cash[group] - 20000) / len(buy_list)
                buy_amount, industry_index_dict = xbuy(date, group, buy_list, sub_cash, cash, position, industry_index_dict)
                tov += buy_amount
                tov_flag += 1

            if date in resample_date['rebalance'].values:
                tmp_fact = pd.read_csv(os.path.join(fac_path, factor, date[:4], date[5:7], date+'.csv'), 
                                    dtype={'Code':object, date:np.float32})
                if tmp_fact.dropna().empty:
                    continue
                
                tmp_fact['Industry'] = np.nan
                for row in tmp_fact.itertuples():
                    if not row[1] in industry_dict.keys():
                        industry_dict[row[1]] = pd.read_csv(os.path.join(stock_path, row[1]+'.zip'), index_col=0)
                        
                    try:
                        tmp_fact.iloc[row[0], -1] = industry_dict[row[1]].loc[date]['sw_level_1']
                    except:
                        continue

                tmp_fact = pd.merge(tmp_fact, sw_index[['行业名称', '行业代码']], left_on='Industry', right_on='行业名称', how='left')
                tmp_fact = tmp_fact.dropna()
                tmp_fact = tmp_fact.sort_values(date, ascending=False)

                tmp_sig = tmp_fact.groupby('行业代码')[date].mean().to_frame().sort_values(date, ascending=False)

                if date in resample_ret.index:
                    tmp_ret = resample_ret.loc[date].to_frame()
                    tmp_ret = pd.concat([tmp_ret, tmp_sig], axis=1).dropna()

                    ic_list.append(stats.spearmanr(tmp_ret)[0])

                group_sig['group1'] = tmp_sig.index[:10].tolist()
                group_sig['group2'] = tmp_sig.index[10:20].tolist()
                group_sig['group3'] = tmp_sig.index[20:].tolist()

            if len(position) > 0:
                if not 'date' in nav.keys():
                    nav['date'] = list()
                nav['date'].append(date)

                for group, pos in position.items():
                    if not group in nav.keys():
                        nav[group] = list()

                    nav_value, hold_value, industry_index_dict = get_nav(date, cash[group], pos, industry_index_dict)
                    nav[group].append(nav_value)

                    if tov_flag > 0:
                        turnover[group].append((tov / tov_flag) / hold_value)

        asset_value = pd.DataFrame(nav)
        if asset_value.empty:
            continue

        asset_value = pd.merge(asset_value, index_500[['date', 'close']], on='date', how='left')
        asset_value.set_index('date', inplace=True)
        asset_value = asset_value / asset_value.iloc[0]
        asset_value.rename(columns={'close':'CSI500'}, inplace=True)

        asset_value.plot(figsize=(20, 10))
        plot_path = os.path.join(rep_path, factor)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plt.savefig(os.path.join(plot_path, '%s-%s.png'%(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))), bbox_inches='tight')

        annual_return_3 = empyrical.annual_return(asset_value['group3'].pct_change())
        annual_return_2 = empyrical.annual_return(asset_value['group2'].pct_change())
        annual_return_1 = empyrical.annual_return(asset_value['group1'].pct_change())

        asset_turnover = pd.DataFrame(turnover)

        group_return = pd.DataFrame({'coe':[1, 2, 3], 
                                        'return':[annual_return_1,annual_return_2,annual_return_3]})
        mono = abs(group_return['coe'].corr(group_return['return']))

        tmp_rep = pd.DataFrame({
            'Factor':[factor],
            'IC':[np.array(ic_list).mean()],
            'IC_IR':[np.array(ic_list).mean() / np.array(ic_list).std()],
            'Group1Return':[annual_return_1],
            'Group1Turnover':[asset_turnover.mean()['group1']],
            'Group3Return':[annual_return_3],
            'Group3Turnover':[asset_turnover.mean()['group3']],
            'Monotonicity':[mono]
        })

        rep = pd.concat([rep, tmp_rep])
        rep.to_csv(os.path.join(rep_path, '%s-%s-Rep.csv'%(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))), index=False)





