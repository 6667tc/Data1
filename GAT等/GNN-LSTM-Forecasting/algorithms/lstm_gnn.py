# -*- coding:utf-8 -*-
"""
@Time：2023/01/10 20:08
@Author：KI
@File：lstm_gnn.py
@Motto：Hungry And Humble
"""
import os
import sys

from get_data import nn_seq, setup_seed

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import lstm_gnn_args_parser
from model_test import test
from model_train import train

args = lstm_gnn_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
setup_seed(123)


def main():
    graph, Dtr, Val, Dte, scaler = nn_seq(args.input_size, args.seq_len,
                                          args.batch_size, args.output_size)
    print(len(Dtr), len(Val), len(Dte))
    print(graph)
    # 若需重新训练请取消注释
    # train(args, Dtr, Val, path=path)
    test(args, Dte, scaler, path=path)


if __name__ == '__main__':
    main()
