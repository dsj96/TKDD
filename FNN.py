import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import math
import numpy as np
import argparse

from metrics import RMSE, MAPE_y, MAPE_y_head
from utils import *


class FNNModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout):
        super(FNNModel, self).__init__()
        self.n_input  = n_input
        self.n_output = n_output
        self.dropout  = dropout
        self.layer1 = nn.Linear(n_input, n_hidden, bias=True)

        self.layer2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer3 = nn.Linear(n_hidden, n_output, bias=True)

    def forward(self, x):
        out1 = self.layer1(x)
        out1 = F.sigmoid(out1)
        out1 = F.dropout(out1, self.dropout, training=True)
        out2 = self.layer2(out1)
        out2 = F.sigmoid(out2)
        out2 = F.dropout(out2, self.dropout, training=True)
        out3 = self.layer3(out2)

        return out3


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool , default=False,
                        help='if use CUDA training.')

    parser.add_argument('--dataset', type=str , default='jinan', # hangzhou or jinan
                        help='select the dataset.')

    parser.add_argument('--seed', type=int, default=0, help='Random seed.') # jinan:2 y=0.55 y_head=0.497 hangzhou:0

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--n_hidden', type=int, default=128,
                        help='Number of hidden units.')

    parser.add_argument('--n_output', type=int, default=1,
                        help='Number of output units.')

    parser.add_argument('--percent', type=float, default=0.2,
                        help='Number of percent to test.')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--num_slice', type=int, default=12,
                        help='take the num_slice G into consideration.')

    parser.add_argument('--matual_split', type=bool, default=False,
                        help='')

    args, _ = parser.parse_known_args()

    return args


def matual_split_data_hangzhou(normed_ways_segment_volume_dict):
    test_ways_list = [0,24,59,201,289]
    train_ways_segment_volume_dict = {}
    test_ways_segment_volume_dict  = {}
    train_ways_set = set(normed_ways_segment_volume_dict.keys()) - set(test_ways_list)
    for test_way in test_ways_list:
        test_ways_segment_volume_dict[test_way] = normed_ways_segment_volume_dict[test_way]
    for train_way in train_ways_set:
        train_ways_segment_volume_dict[train_way] = normed_ways_segment_volume_dict[train_way]
    return train_ways_segment_volume_dict, test_ways_segment_volume_dict


def evalueat_model(model, args, features, test_ways_segment_volume_dict, cur_slice):
    model.eval()
    pre_list, true_list = [], []
    result = model(features)
    for cur_way, volume_list in test_ways_segment_volume_dict.items():
        pre_list.append(result[cur_way])
        true_list.append(volume_list[cur_slice])

    mape_y      = MAPE_y(pre_list, true_list)
    mape_y_head = MAPE_y_head(pre_list, true_list)
    rmse        = RMSE(pre_list, true_list)
    model.train()
    return mape_y, mape_y_head, rmse



def train_model(args, features, ways_segment_volume_dict):

    if args.matual_split:
        train_ways_segment_volume_dict, test_ways_segment_volume_dict = matual_split_data_hangzhou(ways_segment_volume_dict)
    else:
        data_feature, data_target = preprocess_split_data(ways_segment_volume_dict)
        train_volume_arr, test_volume_arr, train_leida_id_arr, test_leida_id_arr = \
                train_test_split(data_feature, data_target, test_size=args.percent, random_state=args.seed)
        train_ways_segment_volume_dict = combine_ways_segment_volume_dict(train_leida_id_arr, train_volume_arr)
        test_ways_segment_volume_dict  = combine_ways_segment_volume_dict(test_leida_id_arr, test_volume_arr)

    mape_y_list, mape_y_head_list, rmse_list = [], [], []


    model = FNNModel(n_input=features.shape[1], n_hidden=args.n_hidden, n_output=args.n_output, dropout = args.dropout)
    optimizer = optim.Adam(model.parameters() , lr=args.lr, weight_decay=args.weight_decay)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    for cur_slice in range(args.num_slice):
        for i in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            result = model(features)
            train_loss = 0.
            for cur_way, volume_list in train_ways_segment_volume_dict.items():
                train_loss = train_loss + (result[cur_way] - volume_list[cur_slice])**2

            print("train_loss:", train_loss)
            train_loss.backward()
            optimizer.step()



        cur_mape_y, cur_mape_y_head, cur_rmse = evalueat_model(model, args, features, test_ways_segment_volume_dict, cur_slice)
        mape_y_list.append(cur_mape_y)
        mape_y_head_list.append(cur_mape_y_head)
        rmse_list.append(cur_rmse)


        model = FNNModel(n_input=features.shape[1], n_hidden=args.n_hidden, n_output=args.n_output, dropout=args.dropout)
        optimizer = optim.Adam(model.parameters() , lr=args.lr, weight_decay=args.weight_decay)

    return mape_y_list, mape_y_head_list, rmse_list


if __name__ == '__main__':
    '''0. preprocess data'''
    args = get_args()
    features = read_pkl('FNN_data/{}/features.pkl'.format(args.dataset))
    ways_segment_volume_dict = read_pkl('FNN_data/{}/ways_segment_volume_dict.pkl'.format(args.dataset))

    '''1.train\evaluate'''
    mape_y_list, mape_y_head_list, rmse_list = train_model(args, features, ways_segment_volume_dict)

    print( 'mape_y_list:{}, mape_y_head_list:{}, rmse_list:{}'.format(np.mean(mape_y_list), np.mean(mape_y_head_list), np.mean(rmse_list)))
    print('over')
