# -*- coding:utf-8 -*-
import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from get_data import setup_seed
from model_train import device
from models import LSTM_GNN, GNN_MLP, GNN_LSTM

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

setup_seed(123)


def test(args, Dte, scaler, path):
    print('loading models...')
    model = LSTM_GNN(args).to(device)
    model.load_state_dict(torch.load(path + '/models/lstm-gnn.pkl')['model'])
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
            _pred = model(seq)
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
        plot(y, pred, ind + 1, label='LSTM-GNN')

    df = {"mse": mses, "rmse": rmses,
          "mae": maes, "mape": mapes}
    df = pd.DataFrame(df)
    df.to_csv(path + '/data/result/lstm-gnn-result.csv')


@torch.no_grad()
def test2(args, Dte, scaler, model_type, path):
    print('loading models...')
    if model_type == 'gnn':
        model = GNN_MLP(args).to(device)
    else:
        model = GNN_LSTM(args).to(device)
    model.load_state_dict(torch.load(path + '/models/' + model_type + '.pkl')['model'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.input_size)]
    preds = [[] for i in range(args.input_size)]
    for graph in tqdm(Dte):
        graph = graph.to(device)
        _pred, targets = model(graph)
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(args.input_size):
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
    mses, rmses, maes, mapes = [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('第', str(ind), '个变量:')
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        print('mape:', get_mape(y, pred))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        mapes.append(get_mape(y, pred))
        print('--------------------------------')
        plot(y, pred, ind + 1, label=model_type)

    df = {"mse": mses, "rmse": rmses,
          "mae": maes, "mape": mapes}
    df = pd.DataFrame(df)
    df.to_csv(path + '/data/result/' + model_type + '-result.csv')


def plot(y, pred, ind, label):
    # plot
    plt.plot(y, color='blue', label='true value')

    plt.plot(pred, color='red', label='pred value')
    plt.title('第' + str(ind) + '变量的预测示意图')
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
