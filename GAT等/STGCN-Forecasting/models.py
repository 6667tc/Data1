# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal.nn import STConv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STGCN(nn.Module):
    def __init__(self, num_nodes, size, K):
        super(STGCN, self).__init__()
        self.conv1 = STConv(num_nodes=num_nodes, in_channels=1, hidden_channels=16,
                            out_channels=32, kernel_size=size, K=K)
        self.conv2 = STConv(num_nodes=num_nodes, in_channels=32, hidden_channels=16,
                            out_channels=32, kernel_size=size, K=K)

    def forward(self, x, edge_index):
        # x(batch_size, seq_len, num_nodes, input_size)
        x, edge_index = x.to(device), edge_index.to(device)
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


class STGCN_MLP(nn.Module):
    def __init__(self, args):
        super(STGCN_MLP, self).__init__()
        self.args = args
        self.out_feats = 128
        self.stgcn = STGCN(num_nodes=args.input_size, size=3, K=1)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(16 * 32, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
            ))

    def forward(self, x, edge_index):
        # x(batch_size, seq_len, input_size)
        x = x.unsqueeze(3)
        x = self.stgcn(x, edge_index)
        preds = []
        for k in range(x.shape[2]):
            preds.append(self.fcs[k](torch.flatten(x[:, :, k, :], start_dim=1)))

        pred = torch.stack(preds, dim=0)

        return pred
