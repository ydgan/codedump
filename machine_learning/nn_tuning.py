import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import pdb

import torch
device = torch.device('cuda:0')
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from sklearn.preprocessing import StandardScaler


class CNNLSTM(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name,param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 1)
    
    def __init__(self, input_size=None, output_channel=None, 
                 hidden_size=None, num_layers=None, dropout=None, 
                    output_size=None, bidirectional=None, seq_len=None):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=output_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.lstm = nn.LSTM(input_size=output_channel, hidden_size=hidden_size, 
                            bidirectional=bidirectional, num_layers=num_layers, batch_first=True)
        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc1 = nn.Linear(hidden_size*seq_len, hidden_size)
        #self.fc1 = nn.Linear(hidden_size*num_layers, hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, output_size)

        self.apply(self.init_weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)

        outputs, (hidden, cell) = self.lstm(x)
        out = outputs.reshape(len(x), -1)
        #out = hidden.permute(1, 0, 2).reshape(len(x), -1)

        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

def get_train_test(feature_path, train_test):
    print('Collecting train-test data...')

    lookback_window = 30
    train_feature, train_label = [], []
    test_feature, test_label = [], []
    #weights = {'-1':0, '0':0, '1':0}
    for root, _, files in os.walk(feature_path):

        if not root.split('\\')[-1] in train_test:
            continue

        for file in [x for x in files if x.endswith('.csv')]:
            #print(file)
            mdata = pd.read_csv(os.path.join(root, file), index_col=0, dtype=np.float32)
            mdata.reset_index(drop=True, inplace=True)

            mdata = (mdata - mdata.mean()) / mdata.std()

            #valid_index =[int(x) for x in mdata[mdata['Label'] != 0.0].index]
            #tmp_index = np.arange(10, mdata.shape[0], 10).tolist()

            #tmp_index += valid_index
            #tmp_index = list(set(tmp_index))
            #tmp_index = [x for x in tmp_index if x >= lookback_window]

            #for key in weights.keys():
                #weights[key] += mdata[mdata.index.isin(tmp_index)][key].sum()

            if root.split('\\')[-1] == train_test[-1]:
                test_feature += [mdata.iloc[i-lookback_window:i, :-1] for i in range(lookback_window, mdata.shape[0]+1)]
                test_label += [mdata.iloc[i-1, -1] for i in range(lookback_window, mdata.shape[0]+1)]
            else:
                train_feature += [mdata.iloc[i-lookback_window:i, :-1] for i in range(lookback_window, mdata.shape[0]+1)]
                train_label += [mdata.iloc[i-1, -1] for i in range(lookback_window, mdata.shape[0]+1)]

    #wts = [x for _, x in weights.items()]
    #wts = torch.tensor([max(wts) / x for x in wts], dtype=torch.float32)

    return torch.from_numpy(np.array(train_feature)).float().to(device),\
             torch.from_numpy(np.array(train_label)).float().to(device), \
                torch.from_numpy(np.array(test_feature)).float().to(device),\
                    torch.from_numpy(np.array(test_label)).float().to(device)


if __name__ == '__main__':
    feature_path = r'E:\data\machine_learning\preprocessed'

    test_loss = []
    train_test = ['1', '2', '3', '4', '5']
    for t in range(6, 11):
        train_test.append(str(t))
        X_train, y_train, X_test, y_test = get_train_test(feature_path, train_test)

        train = TensorDataset(X_train, y_train)
        test = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train, batch_size=256, shuffle=True)
        test_loader = DataLoader(test, batch_size=256, shuffle=False)

        net = CNNLSTM(
            input_size=42,
            hidden_size=256,
            num_layers=1,
            output_channel=128,
            output_size=1,
            bidirectional=False,
            dropout=0,
            seq_len=30
        ).to(device)

        learning_rate = 1e-3
        epochs = 20

        loss_func = nn.MSELoss().to(device)  
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        for epoch in range(epochs):

            net.train()
            train_loss = []
            for _, (Xtr, ytr) in enumerate(train_loader):
                yhat = net(Xtr)
                loss = loss_func(yhat, ytr)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                print('Epoch %s, Train Loss %s'%(epoch+1, np.mean(train_loss)))

                net.eval()
                tloss = []
                with torch.no_grad():
                    for _, (Xts, yts) in enumerate(test_loader):
                        ypreds = net(Xts)
                        tloss.append(loss_func(ypreds, yts).item())
                    print('Epoch %s, Test Loss %s'%(epoch+1, np.mean(tloss)))
        test_loss.append(np.mean(tloss))

    print(np.mean(test_loss))
                


                
