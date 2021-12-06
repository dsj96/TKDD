'''
Descripttion: 
version: 
Date: 2021-06-14 10:36:26
LastEditTime: 2021-07-14 16:45:52
'''
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool , default=False,
                        help='if use CUDA training.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')

    parser.add_argument('--output_dim', type=int, default=128,
                        help='Number of output units.')

    parser.add_argument('--topk', type=int, default=5,
                        help='find the topk most similary item.')

    parser.add_argument('--negk', type=int, default=5,
                        help='find the topk most not similary item.')

    parser.add_argument('--percent', type=float, default=0.2,
                        help='Number of percent to test.')

    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--num_walks', type=int, default=20,
                        help='.')

    parser.add_argument('--walk_length', type=float, default=10,
                        help='.')

    parser.add_argument('--isweighted', type=bool, default=True,
                        help='walk is weighted.')

    parser.add_argument('--model', type=str, default="GCN",
                        choices=["SGC", "GCN"], help='model to use.')

    parser.add_argument('--k_knn', type=int, default=5,
                        choices=[2, 3, 4, 5, 6, 7, 8, 9], help='select the k for knn.')

    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')

    parser.add_argument('--num_slice', type=int, default=12, # 12 denotes 12 slice in a day, max to 120
                        help='take the num_slice G into consideration.') #

    parser.add_argument('--num_head', type=int, default=3,
                        help='The number head of attention.')
    # normalization
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        help='Adj normalize way AugNormAdj or FAMENormAdj.')
    # is weighted adj
    parser.add_argument('--isweighted_adj', type=bool, default=False,
                        help='Adj is weighted sum or just sum.')

    parser.add_argument('--n_trials', type=int, default=100,
                        help='Set the optuna n_trials.') # n_trials

    parser.add_argument('--matual_split', type=bool, default=False,
                        help='') # n_trials


    args, _ = parser.parse_known_args()
    # args.cuda = not args.cuda and torch.cuda.is_available()
    return args
