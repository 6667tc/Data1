import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import math
from itertools import chain
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

#读取数据
df = pd.read_csv("D:\\县域日前功率预测案例分析\\案例采用数据\\功率数据\\STGCN采用功率数据和气象.csv",encoding='ISO-8859-1')
print(df.head(10))
print(len(df))
#数据参数
epochs=20
seq_len=96
input_size=5
output_size=96
batch_size=1
step_size=96

#模型优化器类型
lr=0.0001
optimizer='adam'
weight_decay=1e-4
gamma=0.25
#模型保存路径
path="D:\\县域日前功率预测案例分析\\预测结果\\模型SUM预测_晴天\\TCN\\model.pkl"


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

#param x:true
#param y:pred
def MSE(x, y):
    return np.mean((y - x) ** 2)
def RMSE(x, y):
    return np.sqrt(MSE(x, y))
def MAE(x, y):
    return np.mean(np.abs(y - x))

dataset=df
train = dataset[:96*240]
val = dataset[96*200:96*260]
test = dataset[96*200:96*260]
m_power = np.max(dataset[dataset.columns[1]])
n_power = np.min(dataset[dataset.columns[1]])
def process(data, batch_size, step_size, shuffle):
        load = data[data.columns[1]]
        data = data.values.tolist()
        #load=load/m
        load = (load - n_power) / (m_power - n_power )
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - seq_len-output_size, step_size):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(1, 6):
                    if c == 1:
                        x.append(load[j])
                    else:
                        m=np.max(dataset[dataset.columns[c]])
                        n=np.min(dataset[dataset.columns[c]])
                        x.append((data[j+96][c]-n)/(m-n))
                train_seq.append(x)
            for j in range(i+seq_len, i + seq_len+output_size):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

Dtr = process(train, batch_size, step_size=step_size, shuffle=True)
Val = process(val, batch_size, step_size=step_size, shuffle=True)
Dte = process(test, batch_size, step_size=output_size, shuffle=False)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()

        # padding = (ks - 1)*dilation
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, padding=2, dilation=1),
            Chomp1d(2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=6, dilation=3),
            Chomp1d(6),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=12, dilation=6),
            Chomp1d(12),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, padding=18, dilation=9),
            Chomp1d(18),
            nn.ReLU())

        self.linear = nn.Sequential(
            nn.Linear(960, 96))

        self.dowansample = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=1)
        # self.gap = GAP1d()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size,input_size,seq_len)
        # print(x.shape)
        out = self.conv1(x)  # (batch_size,out_channels,seq_len)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.conv3(out)  # (batch_size,out_channels,seq_len)
        out = self.conv4(out)

        res = self.dowansample(x)

        out = self.relu(res + out)
        out = self.flatten(out)
        # print(out.shape)

        out = self.linear(out)
        # print(out.shape)
        return out


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq, label = seq.to(device), label.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(Dtr, Val, path):
    model = TCN().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('training...')
    global epochs
    epochs = epochs
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    final_val_loss = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    for epoch in range(epochs):
        train_loss = []
        for batch_idx, (seq, target) in enumerate(Dtr, 0):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        # validation
        val_loss = get_val_loss(model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        # print('epoch {:03d} train_loss {:.8f} '.format(epoch, np.mean(train_loss)))
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        final_val_loss.append(val_loss)
        model.train()

    state = {'model': best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)

    return np.mean(train_loss)


def test(Dte, path, m, n):
    print('loading model...')
    model = TCN().to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(Dte, 0):
        seq = seq.to(device)
        target = target.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)

    y = (m - n) * y + n
    pred = (m - n) * pred + n
    # y=y*m
    # pred=pred*m
    y=y[96*10:96*11]
    pred=pred[96*10:96*11]
    print('MSE', MSE(y, pred))
    print('MAE', MAE(y, pred))
    print('RMSE', RMSE(y, pred))

    # 未经过修正绘图
    plt.plot(y[:], c='green', label='true')
    plt.plot(pred[:], c='red', label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()




train(Dtr,Val,path)

test(Dte,path,m_power,n_power)
