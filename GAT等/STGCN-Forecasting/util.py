# -*- coding:utf-8 -*-
import copy
import os
import sys

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from get_data import setup_seed

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

from models import device, STGCN_MLP
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

setup_seed(123)


@torch.no_grad()
def get_val_loss(args, model, Val, edge_index):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, labels) in Val:
        seq = seq.to(device)
        labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
        preds = model(seq, edge_index)
        total_loss = 0
        for k in range(args.input_size):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)


def train(args, Dtr, Val, edge_index):
    edge_index = edge_index.to(device)
    model = STGCN_MLP(args).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, labels) in Dtr:
            seq = seq.to(device)
            labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
            preds = model(seq, edge_index)  # (n_outputs, batch_size, pred_step_size)
            # print(labels.shape)
            # print(preds.shape)
            total_loss = 0
            for k in range(args.input_size):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

            total_loss = total_loss / preds.shape[0]
            # total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val, edge_index)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            state = {'model': best_model.state_dict()}
            torch.save(state, 'models/stgcn.pkl')

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, 'models/stgcn.pkl')


def test(args, Dte, scaler, edge_index):
    print('loading models...')
    edge_index = edge_index.to(device)
    model = STGCN_MLP(args).to(device)
    model.load_state_dict(torch.load('models/stgcn.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for (seq, targets) in tqdm(Dte):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(args.input_size):
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
    mses, rmses, maes, mapes = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个序列:')
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        print('mape:', get_mape(y, pred))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        mapes.append(get_mape(y, pred))
        print('--------------------------------')
        plot(y, pred, ind + 1, label='STGCN')

    df = {"mse": mses, "rmse": rmses,
          "mae": maes, "mape": mapes}
    df = pd.DataFrame(df)
    df.to_csv('data/result.csv')
    plt.show()


def plot(y, pred, ind, label):
    # plot
    plt.plot(y[:500], color='blue', label='pred value')

    plt.plot(pred[:500], color='red', label='true value')
    plt.title('第' + str(ind) + '变量预测示意图')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))
