import os
import pickle
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch_geometric
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch_geometric_temporal.nn import STConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
from torch.optim.lr_scheduler import StepLR
from itertools import chain
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs=50
input_size=14
seq_len=24
output_size=24
batch_size=32
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

#计算相关性
def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)
#计算误差
def get_mae(y, pred):
    return mean_absolute_error(y, pred)
def get_mse(y, pred):
    return mean_squared_error(y, pred)
def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def create_graph(num_nodes, data):
    features = torch.randn((num_nodes, 256))
    edge_index = [[], []]
    # 计算相关系数
    # data (x, num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x, y = data[:, i], data[:, j]
            corr = calc_corr(x, y)
            if corr >= 0.4:
                edge_index[0].append(i)
                edge_index[1].append(j)

    edge_index = torch.LongTensor(edge_index)
    # graph = Data(x=features, edge_index=edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    # print(graph)
    return edge_index

def nn_seq_gnn(num_nodes, seq_len, B, pred_step_size):
    '''
    num_nodes = 节点个数
    seq_len= 输入长度
    B = batch_size
    pred_step_size= 输出长度
    '''
    input_data = pd.read_csv("C:\\Users\\admin\\Desktop\\县域日前预测\数据\\NWP_辐照度9-11_24.csv",encoding='ISO-8859-1')
    input_data.drop([input_data.columns[0]], axis=1, inplace=True)
    output_data = pd.read_csv("C:\\Users\\admin\\Desktop\\县域日前预测\数据\\功率数据.csv",encoding='ISO-8859-1')
    output_data.drop([output_data.columns[0]], axis=1, inplace=True)
    # 数据划分
    train_input = input_data[:24*80]
    val_input = input_data[24*80:24*90]
    test_input = input_data[24*80:24*90]
    # normalization
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    train_input = scaler_input.fit_transform(train_input.values)
    val_input = scaler_input.transform(val_input.values)
    test_input = scaler_input.transform(test_input.values)

    # 数据划分
    train_output = output_data[:24*80]
    val_output = output_data[24*80:24*90]
    test_output = output_data[24*80:24*90]
    # normalization

    train_output = scaler_output.fit_transform(train_output.values)
    val_output = scaler_output.transform(val_output.values)
    test_output = scaler_output.transform(test_output.values)
    #创建图
    edge_index = create_graph(num_nodes, input_data[0:96*80].values)

    def process(dataset_input,dataset_output,batch_size, step_size, shuffle):
        '''

        :param dataset: 数据集
        :param batch_size: 一次训练样本数
        :param step_size: 时间步长，滑动窗口长度
        :param shuffle: 是否打乱数据集
        :return:
        '''
        dataset_input = dataset_input.tolist()
        dataset_output = dataset_output.tolist()
        seq = []
        for i in tqdm(range(0, len(dataset_input) - seq_len - pred_step_size, step_size)):
            train_seq = []
            for j in range(i,i+seq_len):
                x = []
                for c in range(len(dataset_input[0])):  # 取所有变量
                    x.append(dataset_input[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset_output[0])):
                y = []
                for k in range(i, i + seq_len):
                    y.append(dataset_output[k][j])
                train_labels.append(y)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            train_labels = torch.FloatTensor(train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train_input,train_output, B, step_size=1, shuffle=True)
    Val = process(val_input,val_output, B, step_size=1, shuffle=True)
    Dte = process(test_input,test_output,B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler_input,scaler_output, edge_index


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_head):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=num_head, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=num_head, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        # 24 128 / 2 118
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


class STGCN_MLP(nn.Module):
    def __init__(self, input_size,seq_len,output_size):
        super(STGCN_MLP, self).__init__()
        self.input_size=input_size
        self.output_size =output_size
        self.seq_len =seq_len
        self.out_feats = 128
        self.gat = GAT(in_feats=24,h_feats=64,out_feats=24,num_head=1)
        self.fcs = nn.ModuleList()
        for k in range(input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(24, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ))

    def forward(self, x, edge_index):
        # x(batch_size, seq_len, input_size)
        x=x.transpose(1,2)
        xs =[[] for i in range(batch_size)]
        for l in range(batch_size):
            _x = x[l,:,:]
            out_gat = self.gat(_x,edge_index)
            xs[l].append(out_gat)

        out_gat = [torch.stack(x, dim=0) for x in xs]
        out_gat = torch.stack(out_gat, dim=0)#(batch_size,1,input_dize,seq_len)
        out_gat=out_gat.reshape(batch_size,input_size,seq_len)#(batch_size,input_size,seq_len)

        preds = [[] for i in range(input_size)]
        for k in range(input_size):
            preds[k].append(self.fcs[k](out_gat[:,k,:]))

        pred = [torch.stack(x, dim=1) for x in preds]
        pred = torch.stack(pred, dim=1)
        pred = pred.reshape(batch_size, input_size, seq_len)
        pred = pred.transpose(0, 1)
        #(input_size,batch_size,seq_len)
        return pred

def get_val_loss(model, Val, edge_index):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, labels) in Val:
        seq = seq.to(device)
        labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
        preds = model(seq, edge_index)
        total_loss = 0
        for k in range(input_size):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)


def train(Dtr, Val, edge_index):
    edge_index = edge_index.to(device)
    model = STGCN_MLP(input_size=input_size,seq_len=seq_len,output_size=output_size).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=24, gamma=0.25)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for (seq, labels) in Dtr:
            seq = seq.to(device)
            labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
            preds = model(seq, edge_index)  # (n_outputs, batch_size, pred_step_size)
            # print(labels.shape)
            # print(preds.shape)
            total_loss = 0
            for k in range(input_size):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

            total_loss = total_loss / preds.shape[0]
            # total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

        scheduler.step()
        # validation
        val_loss = get_val_loss(model, Val, edge_index)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            state = {'model': best_model.state_dict()}
            torch.save(state, 'models_stgcn.pkl')

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, 'models_stgcn.pkl')


def test(Dte,scaler,edge_index):
    print('loading models...')
    edge_index = edge_index.to(device)
    model = STGCN_MLP(input_size=input_size,seq_len=seq_len,output_size=output_size).to(device)
    model.load_state_dict(torch.load('models_stgcn.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(input_size)]
    preds = [[] for i in range(input_size)]
    for (seq, targets) in tqdm(Dte):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            _pred = model(seq, edge_index)
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    # ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    mses, rmses, maes, = [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个序列:')
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        print('--------------------------------')
        plot(y, pred, ind + 1, label='STGCN')

    df = {"mse": mses, "rmse": rmses,
          "mae": maes}
    df = pd.DataFrame(df)
    df.to_csv('data_result.csv')
    plt.show()


def plot(y, pred, ind, label):
    # plot
    plt.plot(y[:500], color='blue', label='pred value')

    plt.plot(pred[:500], color='red', label='true value')
    plt.title('第' + str(ind) + '变量预测示意图')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()
Dtr, Val, Dte, scaler_input,scaler_output, edge_index=nn_seq_gnn(input_size, seq_len, batch_size, output_size)

train(Dtr,Val,edge_index)
test(Dte,scaler_output,edge_index)
