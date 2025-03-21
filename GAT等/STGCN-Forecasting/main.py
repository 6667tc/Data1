# -*- coding:utf-8 -*-

from args import args_parser
from get_data import nn_seq, setup_seed
from util import test

setup_seed(42)


def main():
    args = args_parser()
    Dtr, Val, Dte, scaler, edge_index = nn_seq(args.input_size, args.seq_len,
                                               args.batch_size, args.output_size)
    print(len(Dtr), len(Val), len(Dte))
    # 如需训练请取消注释
    # train(args, Dtr, Val, edge_index)
    test(args, Dte, scaler, edge_index)


if __name__ == '__main__':
    main()
