import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import multiprocessing as mp

import os
import akshare as ak

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, log_loss

from xgboost import XGBClassifier

grid_params = {
    'learning_rate':[0.01, 0.05, 0.1],
    'max_depth':[5, 7, 9],
    'subsample':[0.8, 0.9],
}

def err_call(err):
    print(str(err))

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

def get_return(stock_path, resample_date):
    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Pool()
    for file in os.listdir(stock_path):
        p.apply_async(cal_return, (stock_path, file, resample_date, q,), error_callback=err_call)
    p.close()
    p.join()

    resample_ret = ret_concat(q)
    resample_ret = pd.merge(resample_date, resample_ret, left_on='retrade', right_index=True, how='right')
    resample_ret.set_index('rebalance', inplace=True)
    resample_ret.drop(columns=['retrade'], inplace=True)
    return resample_ret

def get_train_data(ret, date, path):
    sub_ret = ret.loc[:date]
    sub_date = sub_ret.index[-80:-1]

    train_data = pd.DataFrame()
    for sdate in sub_date:
        year = sdate[:4]
        month = sdate[5:7]

        multi_factors = pd.read_csv(os.path.join(path, year, month, sdate+'.csv'), dtype={'Code':object})
        multi_factors = multi_factors.set_index('Code')

        tmp_ret = resample_ret.loc[sdate, :].to_frame().rename(columns={sdate:'Label'}).dropna()
        multi_factors = pd.merge(multi_factors, tmp_ret, left_index=True, right_index=True, how='left')
        multi_factors = multi_factors[(multi_factors['Label'] >= multi_factors['Label'].quantile(0.8)) |\
                                       (multi_factors['Label'] <= multi_factors['Label'].quantile(0.2))]

        multi_factors['Label'] = np.where(multi_factors['Label'] > 0, 1, 0)
        multi_factors.dropna(inplace=True)

        train_data = pd.concat([train_data, multi_factors], axis=0)
        train_data.reset_index(drop=True, inplace=True)

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_preds = pd.read_csv(os.path.join(path, date[:4], date[5:7], date+'.csv'), dtype={'Code':object})
    X_preds = X_preds.set_index('Code')
    return X_train, y_train, X_preds


if __name__ == '__main__':

    rebal_freq = 3
    stock_path = r'E:\data\stock'
    multi_factors_path = os.path.join(r'E:\factors\multi_factors', str(rebal_freq))

    is_traday, resample_date = get_traday(freq=rebal_freq)
    resample_ret = get_return(stock_path, resample_date)

    for i, date in enumerate(resample_ret.index):
        if date < '2019-01-01' or date > '2023-12-31':
            continue

        X_train, y_train, X_pred = get_train_data(resample_ret, date, os.path.join(multi_factors_path, 'preprocessed'))

        init_xgb = XGBClassifier(objective='binary:logistic')
        cv = KFold(n_splits=5, shuffle=True)
        scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

        grid_search = GridSearchCV(init_xgb, grid_params, cv=cv, 
                                   scoring=scoring, verbose=10)
        grid_search.fit(X_train, y_train)

        best_xgb = grid_search.best_estimator_
        best_xgb.fit(X_train, y_train)

        xgb_sig = pd.DataFrame(best_xgb.predict_proba(X_pred)[:, 1], index=X_pred.index, columns=['XGBoost'])
        
        sig_path = os.path.join(multi_factors_path, 'combined', date[:4], date[5:7])
        sig = pd.read_csv(os.path.join(sig_path, date+'.csv'), dtype={'Code':object})
        sig = pd.merge(sig, xgb_sig, left_on='Code', right_index=True)
        sig.to_csv(os.path.join(sig_path, date+'.csv'), index=False)






            