import os
import akshare as ak

sw_indus_level_1 = ak.sw_index_first_info()
sw_indus_level_1['行业代码'] = sw_indus_level_1['行业代码'].str[:-3]

for code in sw_indus_level_1['行业代码']:
    indus_index = ak.index_hist_sw(code)
    indus_index.rename(columns={
        '日期':'date',
        '收盘':'close',
        '开盘':'open',
        '最高':'high',
        '最低':'low',
        '成交量':'volume',
        '成交额':'amount'
    }, inplace=True)

    indus_index = indus_index[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    indus_index.set_index('date', inplace=True)
    indus_index.to_csv(os.path.join(r'E:\data\sw_industry_index', code+'.csv'))