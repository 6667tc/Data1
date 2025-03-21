# -*- coding:utf-8 -*-
"""
@Time：2023/01/10 14:43
@Author：KI
@File：gnn.py
@Motto：Hungry And Humble
"""
import os
import sys

from get_data import nn_seq_gnn, setup_seed

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import gnn_args_parser
from model_test import test2
from model_train import train2


args = gnn_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
setup_seed(123)


def main():
    graph, Dtr, Val, Dte, scaler = nn_seq_gnn(args.input_size, args.seq_len,
                                              args.batch_size, args.output_size)
    print(len(Dtr), len(Val), len(Dte))
    print(graph)
    # 若需重新训练请取消注释
    # train2(args, Dtr, Val, model_type='gnn', path=path)
    test2(args, Dte, scaler, model_type='gnn', path=path)


if __name__ == '__main__':
    main()
