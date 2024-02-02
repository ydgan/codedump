import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import multiprocessing as mp

import datetime
import baostock as bs
import akshare as ak
import os


def get_date():
    stock_path = os.path.join(r'E:\data', 'stock')
    if not os.path.exists(stock_path):
        os.makedirs(stock_path)

    if os.path.exists(os.path.join(stock_path, '000001.zip')):
        xdata = pd.read_csv(os.path.join(stock_path, '000001.zip'), dtype=object, index_col=0)
        start_date = xdata.index[-1]
    else:
        start_date = '1990-12-01'
    
    start_date = pd.to_datetime(start_date).date()
    end_date = datetime.date.today()
    return start_date, end_date

def err_call(err):
    print(str(err))

def get_bs_list(date, queue):
    date = date.strftime('%Y-%m-%d')

    bs.login()
    stock_list = bs.query_all_stock(date).get_data()
    if not stock_list.empty:
        queue.put(stock_list['code'].tolist())

def stock_list_concat(queue):
    stock_list = []
    while True:
        if not queue.empty():
            tmp_list = queue.get(True)
            stock_list += tmp_list
            stock_list = list(set(stock_list))
        else:
            return pd.DataFrame({'bs_code':stock_list})

def get_stock_list(start_date, end_date):
    xdate = pd.date_range(start_date, end_date).tolist()
    xdate = xdate[0:-1:120]

    print('Collecting Date List...')

    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool(4)
    for date in xdate:
        p.apply_async(get_bs_list, (date, q), error_callback=err_call)
    p.close()
    p.join()

    stock_list = stock_list_concat(q)
    stock_list['code'] = stock_list['bs_code'].str[3:]
    stock_list = stock_list[stock_list['code'].str[:1]!='8']
    stock_list = stock_list[stock_list['code'].str[:1]!='4']
    stock_list = stock_list[stock_list['code'].str[:3]!='399']
    stock_list = stock_list[~((stock_list['code'].str[:3]=='000') & (stock_list['bs_code'].str[:2]=='sh'))]
    return stock_list

def update_list(stock_list, stock2indus, sw_level_1, index_code, queue):
    comp = ak.index_component_sw(index_code)
    comp.set_index('证券代码', inplace=True)

    for stock_code in comp.index:
        
        if not stock_code in stock_list['code'].values:
            continue

        if not stock_code in stock2indus['stock_code'].values:
            queue.put([
                stock_code,
                comp.loc[stock_code, '计入日期'],
                sw_level_1.loc[index_code, '行业名称']
            ])
        elif comp.loc[stock_code, '计入日期'].strftime('%Y-%m-%d') >\
        stock2indus[stock2indus['stock_code'] == stock_code]['start_date'].max() and \
        sw_level_1.loc[index_code, '行业名称'] != \
            stock2indus[stock2indus['stock_code'] == stock_code]['sw_level_1'].iloc[-1]:
            queue.put([
                stock_code,
                comp.loc[stock_code, '计入日期'],
                sw_level_1.loc[index_code, '行业名称']
            ])
            
def list_concat(queue):
    code_list = list()
    date_list = list()
    indus_list = list()

    while True:
        if not queue.empty():
            res = queue.get(True)
            
            code_list.append(res[0])
            date_list.append(res[1])
            indus_list.append(res[2])
        else:
            return pd.DataFrame({'stock_code':code_list, 'start_date':date_list, 'sw_level_1':indus_list})

def get_stock_indus(stock_list):
    '''
    stock2code = pd.read_csv(r'E:\data\sw_level1_comp\stock2code.csv',
                                dtype=object, encoding='gbk')
    code2indus = pd.read_csv(r'E:\data\sw_level1_comp\code2indus.csv',
                                dtype=object, encoding='gbk')
    stock2code['l1code'] = stock2code['行业代码'].str[:3]
    code2indus['l1code'] = code2indus['行业代码'].str[:3]

    pdb.set_trace()
    stock2code = pd.merge(stock2code, code2indus[['l1code', '一级行业名称']].drop_duplicates(), on='l1code', how='left')
    stock2code = pd.merge(stock2code, code2indus[['行业代码', '二级行业名称', '三级行业名称']], on='行业代码', how='left')
    stock2code['计入日期'] = pd.to_datetime(stock2code['计入日期']).apply(lambda x:x.strftime('%Y-%m-%d'))
    
    stock2code = stock2code.rename(columns={'股票代码':'stock_code',
                                            '计入日期':'start_date',
                                            '一级行业名称':'sw_level_1',
                                            '二级行业名称':'sw_level_2',
                                            '三级行业名称':'sw_level_3'})
    '''

    sw_level_1 = ak.sw_index_first_info()
    sw_level_1['行业代码'] = sw_level_1['行业代码'].str[:-3]
    sw_level_1.set_index('行业代码', inplace=True)
    stock2indus = pd.read_csv(r'E:\data\sw_level1_comp\stock2indus.csv', dtype=object)

    print('Collecting Stock List...')

    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool()
    for index_code in sw_level_1.index:
        p.apply_async(update_list, (stock_list, stock2indus, sw_level_1, index_code, q), error_callback=err_call)
    p.close()
    p.join()

    tmp_list = list_concat(q)
    stock2indus = pd.concat([stock2indus, tmp_list]).sort_values(['stock_code', 'start_date'])
    stock2indus.to_csv(r'E:\data\sw_level1_comp\stock2indus.csv', index=False)
    return stock2indus

def get_md(row, end_date, stock2indus):
    bs.login()
    mdata = bs.query_history_k_data_plus(row[1],
                                         'date,open,high,low,close,volume,amount,turn,isST',
                                         start_date='1990-01-01',
                                         end_date=end_date.strftime('%Y-%m-%d'),
                                         adjustflag='1').get_data()
    
    print('Collecting %s...'%row[2])
    mdata.set_index('date', inplace=True)
    mdata['isNEW'] = 0
    mdata['isNEW'][:250] = 1

    mdata = mdata.replace('', np.nan)
    mdata = mdata.astype(float)
    mdata['turn'] = mdata['turn'] / 100
    mdata['mcap'] = mdata['amount'] / mdata['turn']

    mdata['sw_level_1'] = np.nan
    mdata['sw_level_2'] = np.nan
    mdata['sw_level_3'] = np.nan
    if row[2] in stock2indus['stock_code'].values:
        tmp_s2i = stock2indus[stock2indus['stock_code']==row[2]]
        for r2 in tmp_s2i.itertuples():
            mdata['sw_level_1'][mdata.index >= r2[2]] = r2[3]
            mdata['sw_level_2'][mdata.index >= r2[2]] = r2[4]
            mdata['sw_level_3'][mdata.index >= r2[2]] = r2[5]

    xpath = os.path.join(r'E:\data', 'stock', row[2]+'.zip')
    mdata.to_csv(xpath, compression='zip', encoding='utf_8_sig')

if __name__ == '__main__':
    start_date, end_date = get_date()
    stock_list = get_stock_list(start_date, end_date)
    stock2indus = get_stock_indus(stock_list)

    p1 = mp.Pool()
    for row in stock_list.itertuples(name=None):
        #get_md(row, end_date, stock2indus)
        p1.apply_async(get_md, (row, end_date, stock2indus,), error_callback=err_call)
    p1.close()
    p1.join()
