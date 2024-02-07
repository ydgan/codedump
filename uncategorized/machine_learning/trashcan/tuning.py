import warnings
warnings.filterwarnings('ignore')

import os, pdb
import wandb
wandb.login()
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
from matplotlib import pyplot as plt

import torch
device = torch.device('mps')
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import xpkgs.get_data as gdt
import xpkgs.get_net as gnt
import xpkgs.backpropagation as bpp

def get_net(config):
    net = gnt.CNNLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_channel=config['output_channel'],
        #num_heads=config['num_heads'],
        bidirectional=config['bidirectional'],
        output_size=config['output_size'],
        dropout=config['dropout'],
        seq_len=config['window_size']
    ).to(device)

    loss_func = gnt.criterion()
    optimizer = gnt.optimizer(net, config['optimizer'], config['learning_rate'])
    scheduler = gnt.scheduler(optimizer, config['epochs'])
    return net, loss_func, optimizer, scheduler

def run(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        
        train_valid = TensorDataset(X_train_valid, y_train_valid)
        train_valid_loader = DataLoader(train_valid, batch_size=cfg['batch_size'], shuffle=True)
        net, loss_func, optimizer, scheduler = get_net(cfg)

        net.train()
        for epoch in range(cfg['epochs']):
            for _, (tvX, tvy) in enumerate(train_valid_loader):
                loss = bpp.train_batch(tvX, tvy, net, loss_func, optimizer)
            scheduler.step()

        net.eval()
        npath = '/Users/ydgan/Documents/pipeline/net'
        with torch.no_grad():
            preds = net(X_test)
            mse = loss_func(y_test, preds).item()
            rmse = np.sqrt(mse)
            wandb.log({'rmse':rmse})
            rmse = round(rmse, 6)

            net_path = os.path.join(npath, cfg['symbol'], str(rmse))
            if not os.path.exists(net_path):
                os.makedirs(net_path)
            torch.save(net, os.path.join(net_path,'net.pt'))

            scsv = pd.DataFrame(list(dict(cfg).items()))
            scsv.to_csv(os.path.join(net_path,'para.csv'))

if __name__ == '__main__':
    etf_list = pd.read_csv('/Users/ydgan/Documents/pipeline/etf_list.csv')
    for symbol in etf_list['code']:
        try:
            sweep_config = {
                'method':'random',
                'name':'etf_sweep_%s'%(symbol),
                'metric':{
                    'name':'rmse',
                    'goal':'minimize'
                },
                'parameters':{
                    'symbol':{'values':[symbol]},
                    'window_size':{'values':[22]},
                    'split_time':{'values':['20230601']},
                    'epochs':{'values':[50]},
                    'input_size':{'values':[14]},
                    'output_size':{'values':[1]},
                    'batch_size':{'values':[32,64,128,256]},
                    'learning_rate':{'values':[1e-3,1e-4]},
                    'optimizer':{'values':['adam','sgd']},
                    'output_channel':{'values':[32,64,128,256]},
                    #'num_heads':{'values':[2,4,8]},
                    'hidden_size':{'values':[32,64,128,256]},
                    'num_layers':{'values':[1,3,5,7]},
                    'bidirectional':{'values':[True,False]},
                    'dropout':{'values':[0.2,0.4,0.6,0.3]}
                }
            }

            etf_data = gdt.get_etf_data(symbol)
            if etf_data.shape[0] < 100:
                continue

            features = etf_data.iloc[:,1:]
            labels = etf_data.iloc[:,0]
            train_valid_features, test_features = gdt.train_test_split(features, sweep_config['parameters']['split_time']['values'][0])
            train_valid_labels, test_labels = gdt.train_test_split(labels, sweep_config['parameters']['split_time']['values'][0])
            norm_train_valid_features, norm_test_features, _ = gdt.norm(train_valid_features, test_features)

            X_train_valid, y_train_valid = gdt.create_dataset(norm_train_valid_features, train_valid_labels, 
                                                            window=sweep_config['parameters']['window_size']['values'][0])
            X_test, y_test = gdt.create_dataset(norm_test_features, test_labels, 
                                                window=sweep_config['parameters']['window_size']['values'][0])

            sweep_id = wandb.sweep(sweep_config, project='0808tuning')
            wandb.agent(sweep_id, run, count=10)
        except:
            continue