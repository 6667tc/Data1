import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from itertools import chain
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("D:\\PycharmProjects\\Pytorch\\data.csv",encoding='ISO-8859-1')
print(df.head(10))
print(len(df))
#训练次数
epochs=200
#预测需求决定以下参数
input_len=96#输入时间序列长度
input_size=5#输出时间序列维度
output_len=96#输出时间序列长度
output_size=1#输出时间序列维度
step_size=96#划分数据集步长，一般等于输出长度
#模型参数，可调整
d_model=64#模型维度
n_heads=8#多头注意力头数
en_num_layer=1#编码层数
de_num_layer=1#解码层数
#以下参数一般不变
lr=0.001
batch_size=1
optimizer='adam'
weight_decay=1e-4
gamma=0.25
#模型保存路径
path = 'result_1.pkl'

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

#定义误差函数
#param x:true
#param y:pred
def MSE(x, y):
    return np.mean((y - x) ** 2)
def RMSE(x, y):
    return np.sqrt(MSE(x, y))
def MAE(x, y):
    return np.mean(np.abs(y - x))
#数据数据处理
dataset=df
train = dataset[0:96*210]#训练集划分，根据自己的需求改
val = dataset[96*210:]##验证集划分，根据自己的需求改
test = dataset[96*210:]##测试集划分，根据自己的需求改
m_power = np.max(dataset[dataset.columns[1]])#数字8的意思是取数据集中的第9列为标签（预测变量），这一步求标签最大值，为了归一化
n_power = np.min(dataset[dataset.columns[1]])#数字8的意思是取数据集中的第9列为标签（预测变量），这一步求标签最小值，为了归一化
def process(data, batch_size, step_size, shuffle):#数据划分函数
        load = data[data.columns[1]]#取标签数据（预测变量）
        data = data.values.tolist()
        load = (load - n_power) / (m_power - n_power )#归一化（0，1）区间内
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - input_len-output_len, step_size):
            train_seq1 = []#创建输入训练数组.气象数据
            train_seq2 = []  # 创建输入训练数组.功率数据
            train_label = []#创建输出标签数组
            for j in range(i, i + input_len):#向输入训练数组中加数据
                train_seq2.append(load[j])
            for l in range(i+input_len , i+input_len+output_len):#向输入训练数组中加数据
                x=[]
                for c in range(2, 7):#（1，9）的意思是数据集中的第2到8列数据也为输入变量
                    m=np.max(dataset[dataset.columns[c]])
                    n=np.min(dataset[dataset.columns[c]])
                    x.append((data[l][c]-n)/(m-n))
                train_seq1.append(x)
            for k in range(i+input_len , i+input_len+output_len):#向输出标签数组中加数据
                train_label.append(load[k])

            train_seq1 = torch.FloatTensor(train_seq1)
            train_seq2 = torch.FloatTensor(train_seq2)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq1, train_seq2, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

Dtr = process(train, batch_size, step_size=96, shuffle=True)
Val = process(val, batch_size, step_size=96, shuffle=True)
Dte = process(test, batch_size, step_size=step_size, shuffle=False)

#定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x

#定义编码类
class Embedding(nn.Module):
    def __init__(self, size, d_model):
        super(Embedding, self).__init__()
        self.size = size
        self.d_model = d_model
        self.emb = nn.Linear(self.size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)

    def forward(self, x):
        y = self.emb(x)
        z = self.pos_emb(y)
        return y+z

#定义不带mask操作对多头注意力机制
class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, input_size, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.input_size = input_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v):
        # 编码器
        # batch_size * seq_len * d_model
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # batch_size * n_heads * seq_len * head_dim
        q = self.split_heads(q, self.n_heads)
        k = self.split_heads(k, self.n_heads)
        v = self.split_heads(v, self.n_heads)

        # batch_size * n_heads * seq_len * seq_len
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # batch_size * n_heads * seq_len * head_dim
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # batch_size * seq_len * d_model
        z = torch.matmul(attention, v)
        z = self.combine_heads(z)
        z = self.output_linear(z)
        z = self.dropout(z)
        return z

    def split_heads(self, x, n_heads):
        batch_size, seq_len, d_model = x.size()
        head_dim = d_model // n_heads
        x = x.view(batch_size, seq_len, n_heads, head_dim)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size, n_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, n_heads * head_dim)
        return x

#定义带mask操作对多头注意力机制
class Mask_MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, output_size, output_len, dropout=0.1):
        super(Mask_MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.output_size = output_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.output_len = output_len
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, y):
        # batch_size * seq_len * output_size --- batch_size * seq_len * d_model
        # 编码器
        # batch_size * seq_len * d_model
        q = self.q_linear(y)
        k = self.k_linear(y)
        v = self.v_linear(y)

        # batch_size * n_heads * seq_len * head_dim
        q = self.split_heads(q, self.n_heads)
        k = self.split_heads(k, self.n_heads)
        v = self.split_heads(v, self.n_heads)

        # batch_size * n_heads * seq_len * seq_len
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        mask = torch.tril(torch.ones(self.output_len, self.output_len)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # batch_size * n_heads * seq_len * head_dim
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # batch_size * seq_len * d_model
        z = torch.matmul(attention, v)
        z = self.combine_heads(z)
        z = self.output_linear(z)
        z = self.dropout(z)
        return z

    def split_heads(self, x, n_heads):
        batch_size, seq_len, d_model = x.size()
        head_dim = d_model // n_heads
        x = x.view(batch_size, seq_len, n_heads, head_dim)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size, n_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, n_heads * head_dim)
        return x

#定义模型
class transformer(nn.Module):
    def __init__(self, n_heads, d_model, input_size, output_size, output_len, input_len,dropout=0.1):
        super(transformer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        self.output_len = output_len
        self.input_len=input_len

        self.embedding1 = Embedding(size=self.input_size, d_model=self.d_model)
        self.embedding2 = Embedding(size=self.output_size, d_model=self.d_model)
        self.multiheadattention = MultiheadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                     input_size=self.input_size)
        self.pad = nn.Linear(self.output_len,self.input_len)
        self.normlayer =torch.nn.LayerNorm(self.d_model)
        self.feedforward1 = nn.Linear(self.d_model, 4*self.input_size)
        self.feedforward2 = nn.Linear(4*self.input_size, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.mask_multiheadattention = Mask_MultiheadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                               output_size=self.output_size,
                                                               output_len=self.output_len)
        self.feedforward3 = nn.Linear(self.d_model, 4*self.input_size)
        self.feedforward4 = nn.Linear(4*self.input_size, self.d_model)
        self.linear=nn.Linear(self.d_model,self.d_model)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.d_model * self.input_len, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.output_len)
        self.fc3 = nn.Linear(self.output_len, self.output_len)
        self.fc4 = nn.Linear(self.d_model,self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(self.d_model, self.d_model, 1, batch_first=True)

    def forward(self, x, y):
        x = self.embedding1(x)
        y = y.reshape(y.size(0),self.output_len, self.output_size)
        y = self.embedding2(y)
        ed_out = self.multiheadattention(x, x, x)
        ed_out = self.normlayer((x + ed_out).view(-1, self.d_model)).view(1, self.input_len, self.d_model)
        fd1 = self.feedforward1(ed_out)
        fd2 = self.feedforward2(fd1)
        ed_out = self.normlayer((ed_out + fd2).view(-1, self.d_model)).view(1, self.input_len, self.d_model)
        ed_out = self.dropout(ed_out)

        mask_out = self.mask_multiheadattention(y)
        mask_out = self.normlayer((y + mask_out).view(-1, self.d_model)).view(1, self.output_len, self.d_model)

        if ed_out.shape[1] == mask_out.shape[1]:
            pass
        else:
            mask_out = mask_out.transpose(1, 2)
            mask_out = self.pad(mask_out)
            mask_out = mask_out.transpose(1, 2)

        de_out = self.multiheadattention(ed_out, ed_out, mask_out)
        ed_out = self.normlayer((de_out + mask_out).view(-1, self.d_model)).view(1, self.input_len, self.d_model)
        fd3 = self.feedforward3(ed_out)
        fd4 = self.feedforward4(fd3)
        de_out = self.normlayer((fd4 + de_out).view(-1, self.d_model)).view(1, self.input_len, self.d_model)
        de_out = self.linear(de_out)
        # de_out = F.softmax(de_out)
        de_out = self.flatten(de_out)
        de_out = self.fc1(de_out)
        de_out = self.fc2(de_out)
        return de_out


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq1, seq2,label) in Val:
        with torch.no_grad():
            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            label = label.to(device)
            y_pred = model(seq1,seq2)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

def train(Dtr,Val,path):
    model = transformer(n_heads=n_heads,d_model=d_model,input_size=input_size,input_len=input_len,
                        output_size=output_size,output_len=output_len).to(device)
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
        for batch_idx, (seq1,seq2,target) in enumerate(Dtr, 0):
            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred = model(seq1,seq2)
            loss = loss_function(y_pred, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        #validation
        val_loss = get_val_loss(model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        final_val_loss.append(val_loss)
        model.train()

    state = {'model': best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)

    return np.mean(train_loss)

def test(Dte,path,m, n):
    print('loading model...')
    model = transformer(n_heads=n_heads,d_model=d_model,input_size=input_size,input_len=input_len,
                        output_size=output_size,output_len=output_len).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    pred = []
    y = []
    for batch_idx, (seq1,seq2,target) in enumerate(Dte, 0):
        seq1 = seq1.to(device)
        seq2 = seq2.to(device)
        target = target.to(device)
        with torch.no_grad():
            y_pred = model(seq1,seq2)
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)

    y = (m - n) * y + n
    pred = (m - n) * pred + n
    #y=y*m
    #pred=pred*m
    y=y[0:]
    pred=pred[0:]
    print('MSE', MSE(y, pred))
    print('MAE', MAE(y, pred))
    print('RMSE', RMSE(y, pred))

    #未经过修正绘图
    plt.plot(y[:], c='green', label='true')
    plt.plot(pred[:], c='red', label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()



train(Dtr,Val,path)
test(Dte,path,m_power,n_power)