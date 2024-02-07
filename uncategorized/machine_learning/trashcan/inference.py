import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np

import torch
device = torch.device('mps')
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pdb

import xpkgs.get_data as gdt
import xpkgs.get_net as gnt
import xpkgs.backpropagation as bpp

net_para = dict()
net_path = '/Users/ydgan/Documents/pipeline/net'
for root, dirs, files in os.walk(net_path):
    if root.split('/')[-1][:2] == 'sh' or root.split('/')[-1][:2] == 'sz':
        net_para[root.split('/')[-1]] = np.array(dirs, dtype=np.float32).min()

def get_net(config):
    net = gnt.CNNLSTM(
        input_size=int(config['input_size']),
        hidden_size=int(config['hidden_size']),
        num_layers=int(config['num_layers']),
        output_channel=int(config['output_channel']),
        #num_heads=config['num_heads'],
        bidirectional=bool(config['bidirectional']),
        output_size=int(config['output_size']),
        dropout=float(config['dropout']),
        seq_len=int(config['window_size'])
    ).to(device)

    loss_func = gnt.criterion()
    optimizer = gnt.optimizer(net, config['optimizer'], float(config['learning_rate']))
    scheduler = gnt.scheduler(optimizer, int(config['epochs']))
    return net, loss_func, optimizer, scheduler

x_return = dict()
for key,value in net_para.items():
    print('processing %s...'%(key))
    parap = os.path.join(net_path, key, str(value), 'para.csv')
    para = pd.read_csv(parap)
    config = dict()
    for row in para.itertuples():
        config[row[2]] = row[3]
    net, loss_func, optimizer, scheduler = get_net(config)

    etf_data = gdt.get_etf_data(key)
    features = etf_data.iloc[:,1:]
    labels = etf_data.iloc[:,0]
    norm_all_features, _, scaler = gdt.norm(features, features)
    X_all, y_all = gdt.create_dataset(norm_all_features, labels, window=int(config['window_size']))

    all = TensorDataset(X_all, y_all)
    all_loader = DataLoader(all, batch_size=int(config['batch_size']), shuffle=True)

    net.train()
    for epoch in range(int(config['epochs'])):
        for _, (X, y) in enumerate(all_loader):
            loss = bpp.train_batch(X, y, net, loss_func, optimizer)
        scheduler.step()

    forecast = 3
    preds = []
    close = etf_data['close'].values.tolist()
    X_preds = X_all[-1].unsqueeze(0)
    
    net.eval()
    for i in range(forecast):
        with torch.no_grad():
            y_preds = net(X_preds)
            preds.append(y_preds.detach().cpu().numpy()[0][0])

            preds_inv = np.exp(preds[-1])
            close_preds = close[-1] * preds_inv
            close.append(float(close_preds))
            X_tmp = pd.DataFrame({'close':close})
            X_tmp = gdt.get_technical_indicators(X_tmp)
            X_tmp = gdt.get_fourier(X_tmp)

            norm_tmp = scaler.transform(X_tmp)
            X_preds = norm_tmp[-1*int(config['window_size']):]
            X_preds = torch.tensor(X_preds).float().unsqueeze(0).to(device)
    x_return[key] = [sum(preds)]

rank = pd.DataFrame(x_return).T
rank = rank.rename(columns={0:'x_log_return'})
rank = rank.sort_values('x_log_return', ascending=False)
rank.to_csv(os.path.join(net_path, 'rank.csv'))

print('done^^')

        

