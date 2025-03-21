# -*- coding:utf-8 -*-

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch_geometric
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric import loader
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric_temporal import DynamicGraphTemporalSignalBatch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_corr(a, b):
    s1 = Series(a)
    s2 = Series(b)
    return s1.corr(s2)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


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


def adj2coo(adj):
    # adj numpy
    edge_index_temp = sp.coo_matrix(adj)
    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    return edge_index


def nn_seq(num_nodes, seq_len, B, pred_step_size):
    data = pd.read_csv('data/data.csv')
    data = data[:5000]

    data.drop([data.columns[0]], axis=1, inplace=True)
    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # normalization
    scaler = MinMaxScaler()
    train = scaler.fit_transform(data[:int(len(data) * 0.8)].values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    edge_index = create_graph(num_nodes, data[:int(len(data) * 0.8)].values)

    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()
        seq = []
        edge_indices = []
        features = []
        targets = []
        edge_weights = []
        batches = []
        ind = 0
        for i in tqdm(range(0, len(dataset) - seq_len - pred_step_size, step_size)):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # 前24个时刻的所有变量
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            # print(train_seq.shape)   # 24 13
            train_labels = torch.FloatTensor(train_labels)
            # edge_index = create_graph(num_nodes, train_seq.numpy())
            # graph = Data(x=train_seq.T, edge_index=edge_index, y=train_labels)
            seq.append((train_seq, train_labels))
            # seq.append(graph)
            # # new
            # edge_indices.append(edge_index.numpy())
            # edge_weights.append(np.array([1 for _ in range(edge_index.shape[1])]))
            # features.append(train_seq.T.numpy())
            # targets.append(train_labels.numpy())
            # batches.append(ind)
            # if len(batches) % batch_size == 0:
            #     ind += 1

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler, edge_index


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset
