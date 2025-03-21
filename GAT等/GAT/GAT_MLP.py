
import random
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch_geometric
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os
from itertools import chain
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs=100#训练次数
seq_len=24
input_size=14
output_size = 24
hid_feats=64
gat_head=2
step_size=24#一般等于输出序列长度
batch_size=32
lr=0.0001
optimizer='adam'
weight_decay=1e-4
gamma=0.25
path='result.pkl'#必须是pkl文件


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)
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


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
#创建图
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
    graph = Data(x=features, edge_index=edge_index)
    graph.edge_index = to_undirected(graph.edge_index, num_nodes=num_nodes)
    return graph


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

    graph = create_graph(num_nodes, input_data[0:96*80].values)

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
        graphs = []
        for i in tqdm(range(0, len(dataset_input) - seq_len - pred_step_size, step_size)):
            train_seq = []
            for j in range(len(dataset_input[0])):
                x = []
                for c in range(i, i + seq_len):  # 取所有变量
                    x.append(dataset_input[c][j])
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
            # print(train_seq.shape, train_labels.shape)  # 24 13, 13 1
            # 此处可利用train_seq创建动态的邻接矩阵
            temp = Data(x=train_seq, edge_index=graph.edge_index, y=train_labels)
            # print(temp)
            graphs.append(temp)

        loader = torch_geometric.loader.DataLoader(graphs, batch_size=batch_size,
                                                   shuffle=shuffle, drop_last=False)

        return loader

    Dtr = process(train_input,train_output, B, step_size=1, shuffle=True)
    Val = process(val_input,val_output, B, step_size=1, shuffle=True)
    Dte = process(test_input,test_output,B, step_size=pred_step_size, shuffle=False)

    return graph, Dtr, Val, Dte, scaler_input, scaler_output




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

class GNN_MLP(nn.Module):
    def __init__(self, seq_len,input_size,output_size,hid_feats,gat_head):
        super(GNN_MLP, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.gat_head = gat_head
        self.hid_feats = hid_feats
        self.gat = GAT(in_feats=self.seq_len, h_feats=32, out_feats=self.hid_feats, num_head=self.gat_head)
        self.fc = nn.Sequential(
            nn.Linear(self.hid_feats, 8),
            nn.ReLU(),
            nn.Linear(8, self.output_size)
        )
        self.fcs = nn.ModuleList()
        for k in range(self.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(self.hid_feats, 8),
                nn.ReLU(),
                nn.Linear(8, self.output_size),
                nn.Sigmoid()
            ))

    def forward(self, data):
        # Data(x=[13, 24], edge_index=[2, 32], y=[13, 1])
        # DataBatch(x=[6656, 24], edge_index=[2, 16384], y=[6656, 1], batch=[6656], ptr=[513])
        # output(13, 512, 1) y(512, 13, 1)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = torch.max(batch).item() + 1
        x = self.gat(x, edge_index)   # 6656 128 = 512 * (13, 128)   # y = 6656 1 = 512 * (13 1)
        batch_list = batch.cpu().numpy()
        # print(batch_list)
        # split
        xs = [[] for k in range(batch_size)]
        ys = [[] for k in range(batch_size)]
        for k in range(x.shape[0]):
            xs[batch_list[k]].append(x[k, :])
            ys[batch_list[k]].append(data.y[k, :])

        xs = [torch.stack(x, dim=0) for x in xs]
        ys = [torch.stack(x, dim=0) for x in ys]
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        # print(x.shape, y.shape)  # 512 13 128 / 512 13 1
        # output(13, 512, 1) y(512, 13, 1)
        preds = []
        x = x.permute(1, 0, 2)  # 13 512 128
        for idx, fc in enumerate(self.fcs):
            preds.append(fc(x[idx, :, :]))

        pred = torch.stack(preds, dim=0)

        return pred, y

def get_val_loss2(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for graph in Val:
        graph = graph.to(device)
        preds, labels = model(graph)
        total_loss = 0
        for k in range(input_size):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)


def train2(Dtr, Val, path):
    model = GNN_MLP(seq_len=seq_len,input_size=input_size,output_size=output_size,hid_feats=hid_feats,gat_head=gat_head).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for graph in Dtr:
            graph = graph.to(device)
            preds, labels = model(graph)
            total_loss = 0
            for k in range(input_size):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

            total_loss = total_loss / preds.shape[0]
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

        scheduler.step()
        # validation
        val_loss = get_val_loss2(model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            state = {'model': best_model.state_dict()}
            torch.save(state, path)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, path)




def plot(y, pred, ind):
    # plot
    plt.plot(y, color='blue', label='true value')
    plt.plot(pred, color='red', label='pred value')
    plt.title('第' + str(ind) + '变量的预测示意图')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()

def test2(Dte,scaler,path):
    print('loading models...')
    model = GNN_MLP(seq_len=seq_len,input_size=input_size,output_size=output_size,hid_feats=hid_feats,gat_head=gat_head).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(input_size)]
    preds = [[] for i in range(input_size)]
    for graph in tqdm(Dte):
        graph = graph.to(device)
        _pred, targets = model(graph)
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        for i in range(_pred.shape[0]):
            pred = _pred[i]
            pred = list(chain.from_iterable(pred.data.tolist()))
            preds[i].extend(pred)

    # ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    mses, rmses, maes,  = [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个变量:')
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        print('--------------------------------')
        plot(y, pred, ind + 1)

    df = {"mse": mses, "rmse": rmses,
          "mae": maes}
    df = pd.DataFrame(df)
    df.to_csv('error.csv')

graph, Dtr, Val, Dte, scaler_input,scaler_output  = nn_seq_gnn(input_size,seq_len,batch_size,output_size)
train2(Dtr,Val,path)
test2(Dte,scaler_output,path)
