import numpy as np

import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

import pdb

class AttentionLayer(nn.Module):
    def __init__(self, num_heads=None, query_dim=None, 
                 key_dim=None, value_dim=None,
                 hidden_size=None, dropout=0.2):
        super(AttentionLayer, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        
        self.hidden_size = hidden_size

        self.fcquery = nn.Linear(query_dim, hidden_size)
        self.fckey = nn.Linear(key_dim, hidden_size)
        self.fcvalue = nn.Linear(value_dim, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = np.sqrt(hidden_size // num_heads)
        
    def heads_trans(self, x):
        return x.view(x.shape[0], -1, self.num_heads, 
                      self.hidden_size // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        querys = self.fcquery(query)
        keys = self.fckey(key)
        values = self.fcvalue(value)

        mquerys = self.heads_trans(querys)
        mkeys = self.heads_trans(keys)
        mvalues = self.heads_trans(values)
        
        attn_weights = torch.matmul(mquerys, mkeys.permute(0, 1, 3, 2))
        attn_weights = attn_weights / self.scale
        attn_weights = self.dropout(F.softmax(attn_weights, dim=-1))

        context = torch.matmul(attn_weights, mvalues)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.shape[0], -1, self.hidden_size)
        return context

class BiLSTMATT(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name,param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def __init__(self, input_size=None, hidden_size=None,
                    num_layers=None, dropout=0.2, output_size=None,
                      num_heads=None, bidirectional=None,
                      seq_len=None):
        super(BiLSTMATT, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            bidirectional=bidirectional, dropout=dropout, 
                            num_layers=num_layers, batch_first=True)
        if bidirectional:
            hidden_size = hidden_size * 2
            
        self.attention = AttentionLayer(num_heads=num_heads, query_dim=hidden_size, 
                                        key_dim=hidden_size, value_dim=hidden_size, 
                                        hidden_size=hidden_size, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_size*seq_len, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.apply(self.init_weight)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        context = self.attention(outputs, outputs, outputs)
        context = context.reshape(len(x), -1)
        context = self.dropout(self.relu(self.fc1(context)))
        out = self.fc2(context)
        return out

class CNNLSTM(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name,param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def __init__(self, input_size=None, output_channel=None, 
                 hidden_size=None, num_layers=None, dropout=None, 
                    output_size=None, bidirectional=None,
                    seq_len=None):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=output_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.lstm = nn.LSTM(input_size=seq_len*output_channel, hidden_size=hidden_size, 
                            bidirectional=bidirectional, dropout=dropout, 
                            num_layers=num_layers, batch_first=True)
        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc1 = nn.Linear(hidden_size*num_layers, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, output_size)

        self.apply(self.init_weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), 1, -1)

        outputs, (hidden, cell) = self.lstm(x)
        out = hidden.permute(1, 0, 2).reshape(len(x), -1)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

class SimpleLSTM(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name,param in m.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def __init__(self, input_size=None, hidden_size=None,
                    num_layers=None, dropout=0.2, 
                    output_size=None, bidirectional=None):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            bidirectional=bidirectional, dropout=dropout, 
                            num_layers=num_layers, batch_first=True)
        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc1 = nn.Linear(hidden_size*num_layers, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, output_size)

        self.apply(self.init_weight)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        out = hidden.permute(1, 0, 2).reshape(len(x), -1)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, 
                 kernel_size, stride,
                 dilation, padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(input_size, output_size, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(output_size, output_size, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(input_size, output_size, 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, channel_size, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_layers = len(channel_size)
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else channel_size[i-1]
            out_channels = channel_size[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                     stride=1, dilation=dilation_size, 
                                     padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, channel_size, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, channel_size, 
                                   kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(channel_size[-1], output_size)

    def forward(self, x):
        output = self.tcn(x)
        return self.fc(output[:,:,-1])