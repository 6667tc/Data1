# -*- coding:utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=4, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=4, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        # 24 128 / 2 118
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
        else:
            x = F.elu(self.conv1(x, edge_index, edge_weight))
            x = self.conv2(x, edge_index, edge_weight)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
        else:
            # print(edge_index.shape, edge_weight.shape)
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

        return x


class GNN_MLP(nn.Module):
    def __init__(self, args):
        super(GNN_MLP, self).__init__()
        self.args = args
        self.out_feats = 128
        self.gat = GAT(in_feats=args.seq_len, h_feats=100, out_feats=self.out_feats)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, args.output_size)
        )
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
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


class GNN_LSTM(nn.Module):
    def __init__(self, args):
        super(GNN_LSTM, self).__init__()
        self.args = args
        self.out_feats = 128
        self.gat = GAT(in_feats=args.seq_len, h_feats=100, out_feats=self.out_feats)
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=128,
                            num_layers=args.num_layers, batch_first=True, dropout=0.5)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
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
        x = x.permute(0, 2, 1)   # 512 128 13
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        preds = []
        for fc in self.fcs:
            preds.append(fc(x))

        pred = torch.stack(preds, dim=0)

        return pred, y


class LSTM_GNN(nn.Module):
    def __init__(self, args):
        super(LSTM_GNN, self).__init__()
        self.args = args
        self.out_feats = 128
        self.sage = GraphSAGE(in_feats=args.hidden_size, h_feats=100, out_feats=self.out_feats)
        self.lstm = nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True, dropout=0.5)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
            ))

    def create_edge_index(self, adj):
        adj = adj.cpu()
        ones = torch.ones_like(adj)
        zeros = torch.zeros_like(adj)
        edge_index = torch.where(adj > 0, ones, zeros)
        #
        edge_index_temp = sp.coo_matrix(edge_index.numpy())
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(indices)
        # edge_weight
        edge_weight = []
        t = edge_index.numpy().tolist()
        for x, y in zip(t[0], t[1]):
            edge_weight.append(adj[x, y])
        edge_weight = torch.FloatTensor(edge_weight)
        edge_weight = edge_weight.unsqueeze(1)
        # print(edge_index)
        # print(edge_weight)
        return edge_index.to(device), edge_weight.to(device)

    def forward(self, x):
        # x (b, s, i)
        x, _ = self.lstm(x)  # b, s, h
        # 对s * h间执行图卷积
        s = torch.randn((x.shape[0], x.shape[1], 128)).to(device)
        for k in range(x.shape[0]):
            feat = x[k, :, :]  # s, h
            # creat edge_index
            adj = torch.matmul(feat, feat.T)  # s * s
            adj = F.softmax(adj, dim=1)
            edge_index, edge_weight = self.create_edge_index(adj)
            feat = self.sage(feat, edge_index, edge_weight)
            s[k, :, :] = feat

        # s(b, s, 64)
        s = s[:, -1, :]
        preds = []
        for fc in self.fcs:
            preds.append(fc(s))

        pred = torch.stack(preds, dim=0)

        return pred
