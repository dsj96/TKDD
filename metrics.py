from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import time

from utils import inverse_norm
# from plot_fig.histogram import plot_histogram


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def MAPE_y_head(pre_volume, true_volume):
    MAPE_SCORE = []
    eps = 1e-5
    for i in range(len(pre_volume)):
        cur_mape = abs(pre_volume[i] - true_volume[i])/(pre_volume[i]+eps)
        MAPE_SCORE.append(cur_mape.item())
    return np.mean(MAPE_SCORE)

def MAPE_y(pre_volume, true_volume):
    MAPE_SCORE = []
    eps = 1e-5
    for i in range(len(pre_volume)):
        cur_mape = abs(pre_volume[i] - true_volume[i])/(true_volume[i]+eps)
        MAPE_SCORE.append(cur_mape.item())
    return np.mean(MAPE_SCORE)

def RMSE(pre_volume, true_volume):
    RMSE_SCORE = []
    for i in range(len(pre_volume)):
        cur_rmse = (pre_volume[i] - true_volume[i])**2
        RMSE_SCORE.append(cur_rmse.item())
    return (np.sum(RMSE_SCORE)/len(pre_volume))**0.5


def show_info(epoch, leida, time_slice_volume, unnormed_ways_segment_volume_dict):
    # if len(time_slice_volume) == 12:
    #     file_path = "hangzhou"
    # else:
    file_path = "jinan"
    with open("{}\log\{}_epoch_{}_roadid_{}_log.txt".format(file_path,time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())),epoch,leida), "w", encoding='utf-8') as f:
        # print("cur_raod id:{}".format(leida))
        f.write("cur_raod id:{}\n".format(leida))
        for i in range(len(time_slice_volume)):
            # print("第{}个时间片上\t预测流量:{:.2f}\t真实流量:{}".format(i, time_slice_volume[i], unnormed_ways_segment_volume_dict[leida][i]))
            f.write("第{}个时间片上\t预测流量:{:.2f}\t真实流量:{}\n".format(i, time_slice_volume[i], unnormed_ways_segment_volume_dict[leida][i]))

def calculate_index(epoch, pre_leida_time_slice_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, topk):
    for leida, time_slice_volume in pre_leida_time_slice_volume_dict.items():
        for idx,volume in enumerate(time_slice_volume):
            pre_leida_time_slice_volume_dict[leida][idx] = volume*volume_sqrt_var[idx] + volume_mean[idx]

    leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info   = {}, {}, {}
    for leida, time_slice_volume in pre_leida_time_slice_volume_dict.items(): # key=leida_id value=[] len =12
        for idx,item in enumerate(time_slice_volume):
            if item < 0:
                time_slice_volume[idx] =  -time_slice_volume[idx]
        leida_pre_MAPE_info_y[leida]      = MAPE_y(time_slice_volume, unnormed_ways_segment_volume_dict[leida])
        leida_pre_MAPE_info_y_head[leida] = MAPE_y_head(time_slice_volume, unnormed_ways_segment_volume_dict[leida])
        leida_pre_RMSE_info[leida]        = RMSE(time_slice_volume, unnormed_ways_segment_volume_dict[leida])
        show_info(epoch, leida, time_slice_volume, unnormed_ways_segment_volume_dict)
    # plot_histogram(pre_leida_time_slice_volume_dict, unnormed_ways_segment_volume_dict)
    return leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info


def evaluate_metric(epoch, output_embedding, train_ways_segment_volume_dict, train_ways_segment_vec_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, topk, volume_sqrt_var, volume_mean):
    true_volume = []
    MAP_SCORE = []
    test_ways_segment_vec_dict = {}
    test_ways_segment_list = list(test_ways_segment_volume_dict.keys())
    for i, item in enumerate(test_ways_segment_list):
        test_ways_segment_vec_dict[item] = output_embedding[:, item]

    pre_leida_time_slice_volume_dict = { }
    for k1,v1 in test_ways_segment_vec_dict.items():
        score_dict = {}
        for k2, v2 in train_ways_segment_vec_dict.items():
            if(k1 != k2):
                curr_score = torch.cosine_similarity(v1, v2, dim=-1)
                score_dict[k2] = curr_score # size=12
        sorted_score_dict_max_list = []
        for j in range(v1.shape[0]):
            sorted_score_dict_max_list.append([item[0] for item in sorted(score_dict.items(), key=lambda item:item[1][j], reverse = True)[:topk]])
        pre_volume     = []
        for time_slice, top_list in enumerate(sorted_score_dict_max_list):
            sum_volume_max = .0
            sum_sim_score  = .0
            for top_item in top_list:
                sum_volume_max = sum_volume_max + train_ways_segment_volume_dict[top_item][time_slice]*score_dict[top_item][time_slice]
                sum_sim_score = sum_sim_score + score_dict[top_item][time_slice]
            cur_pre_volume = sum_volume_max/sum_sim_score
            pre_volume.append(cur_pre_volume)
        pre_leida_time_slice_volume_dict[k1] = pre_volume
    leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info = calculate_index(epoch, pre_leida_time_slice_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, topk)
    return leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info
