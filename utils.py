import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import random
import networkx as nx
import numpy as np
import time
from time import perf_counter
import math
import scipy.sparse as sp

from normalization import *

def read_pkl(fname):
    with open(fname, 'rb') as fo:
        pkl_data = pickle.load(fo, encoding='bytes')
    return pkl_data

def write_pkl(pkl_data, fname):
    fo = open(fname, 'wb')
    pickle.dump(pkl_data, fo)
    print("pkl_file write over!")


def write_dict(dict_data, fname):
    with open(fname, "w", encoding="utf-8") as f:
        for k,v in dict_data:
            f.write(str(k)+"\t"+str(v) + "\n")
        print("dict write over!")

def read_way_segment_all_info(fname):
    '''
    return:
    ways_segment_leida_dict         key=way_seg_id  value=leida_id              str
    ways_segment_road_length_dict   key=way_seg_id  value=road length           int
    ways_segment_grade_dict         key=way_seg_id  value=segment_grade         str
    ways_segment_speed_dict         key=way_seg_id  value=max limit speed       int
    ways_segment_lanes_dict         key=way_seg_id  value=num_of_lanes          int

    ways_segment_start_end_id_dict  key=way_seg_id  value=[start_id, end_id]    str
    ways_segment_start_end_name_dict    key=way_seg_id  value=[start_name, end_name]   int
    ways_segment_start_end_lans_dict    key=way_seg_id  value=[num_of_lanes,num_of_lanes]   int

    '''
    ways_segment_leida_dict = dict()
    ways_segment_road_length_dict = dict()
    ways_segment_grade_dict = dict()
    ways_segment_speed_dict = dict()
    ways_segment_lanes_dict = dict()
    ways_segment_all_name_dict = dict()

    ways_segment_start_end_id_dict = dict()
    ways_segment_start_end_name_dict = dict()
    ways_segment_start_end_lans_dict = dict()


    data = pd.read_csv(fname, encoding="gbk",keep_default_na=False)

    slice_data = data.iloc[:,:]

    for i in range(len(slice_data)):
        segment_id = slice_data.iloc[i][0]
        all_name = slice_data.iloc[i][1]
        leida_id = slice_data.iloc[i][16]
        road_length = slice_data.iloc[i][2]
        grade = str(slice_data.iloc[i][3])
        speed = slice_data.iloc[i][4]
        num_lanes  = slice_data.iloc[i][5]
        start_id = slice_data.iloc[i][6]
        end_id = slice_data.iloc[i][11]
        start_name = slice_data.iloc[i][7]
        end_name = slice_data.iloc[i][12]
        num_start_lans = slice_data.iloc[i][8]
        num_end_lans = slice_data.iloc[i][13]


        if leida_id != "无设备":
            ways_segment_leida_dict[segment_id] = leida_id

        ways_segment_road_length_dict[segment_id] = road_length
        ways_segment_grade_dict[segment_id] = grade
        ways_segment_speed_dict[segment_id] = speed
        ways_segment_lanes_dict[segment_id] = num_lanes
        ways_segment_all_name_dict[segment_id] = all_name

        ways_segment_start_end_id_dict[segment_id] = [start_id, end_id]
        ways_segment_start_end_name_dict[segment_id] = [start_name, end_name]
        ways_segment_start_end_lans_dict[segment_id] = [int(num_start_lans), int(num_end_lans)]

    return ways_segment_leida_dict, ways_segment_road_length_dict, ways_segment_grade_dict, \
           ways_segment_speed_dict, ways_segment_lanes_dict, ways_segment_start_end_id_dict, \
           ways_segment_start_end_name_dict, ways_segment_start_end_lans_dict, \
           ways_segment_all_name_dict

def read_leida_volume_dict(fname):

    leida_lukou_dict = dict()
    leida_volume_dict = dict()
    leida_direction_dict = dict()
    data = pd.read_csv(fname, encoding="gbk")

    slice_data = data.iloc[:,[0,2,5,3]]
    for i in range(len(slice_data)):
        leida_id = slice_data.iloc[i][0]
        lukou_id = slice_data.iloc[i][1]
        volume = slice_data.iloc[i][2]
        direction = slice_data.iloc[i][3]
        leida_lukou_dict[leida_id] = lukou_id

        if leida_id not in leida_direction_dict.keys():
            leida_direction_dict[leida_id] = direction
        else:
            if leida_direction_dict[leida_id] != direction:
                leida_direction_dict[leida_id] = leida_direction_dict[leida_id] + direction

        if leida_id in leida_volume_dict.keys():
            leida_volume_dict[leida_id] = leida_volume_dict[leida_id] + volume
        else:
            leida_volume_dict[leida_id] = volume
    return leida_volume_dict, leida_lukou_dict, leida_direction_dict


def gen_graph_direct_edge(ways_segment_start_end_id_dict, ways_segment_grade_dict, ways_segment_lanes_dict, ways_segment_all_name_dict):
    '''
    parm:
    ways_segment_start_end_id_dict  key=way_seg_id  value=[start_id, end_id]    [str, str]
    ways_segment_grade_dict         key=way_seg_id  value=segment_grade         str
    ways_segment_lanes_dict         key=way_seg_id  value=num_of_lanes          int
    ways_segment_all_name_dict      key=way_seg_id  value=all_name              str
    return:
    edge_list = [(way_seg_id1, way_seg_id2), (way_seg_id2, way_seg_id3)......]

    '''
    edge_list = []
    for segments_id, start_end_list in ways_segment_start_end_id_dict.items():
        for k,v in ways_segment_start_end_id_dict.items():
            first_segment_all_name = ways_segment_all_name_dict[segments_id].split(":")[0]
            second_segment_all_name = ways_segment_all_name_dict[k].split(":")[0]
            if start_end_list[1] == v[0] and v[1] != start_end_list[0] and first_segment_all_name==second_segment_all_name and first_segment_all_name!="无名道路":
                edge_list.append((segments_id, k))
    return edge_list

def gen_graph_direct_edge_jinan(ways_segment_start_end_id_dict):
    edge_list = []
    for segments_id, start_end_list in ways_segment_start_end_id_dict.items():
        for k,v in ways_segment_start_end_id_dict.items():
            if start_end_list[1] == v[0] and v[1] != start_end_list[0]:
                edge_list.append((segments_id, k))
    return edge_list


def get_G_from_edges(edge_list, ways_segment_start_end_id_dict):
    edge_dict = {}
    temp_G =nx.Graph()
    isolated_way_segments = []


    for start_end_truple in edge_list:
        temp_G.add_edge(start_end_truple[0], start_end_truple[1])
        temp_G[start_end_truple[0]][start_end_truple[1]]["weight"] = 0.0


    for segment_id in ways_segment_start_end_id_dict.keys():
        if segment_id not in temp_G.nodes:
            isolated_way_segments.append(segment_id)


    for elem in isolated_way_segments:
        temp_G.add_node(elem)

    print('number of nodes G: ', temp_G.number_of_nodes())
    print('number of edges G: ', temp_G.number_of_edges())
    return temp_G, isolated_way_segments

def get_G_from_edges_jinan(edge_list, ways_segment_start_end_id_dict):
    edge_dict = {}
    temp_G =nx.Graph()
    isolated_way_segments = []

    for start_end_truple in edge_list:
        temp_G.add_edge(start_end_truple[0], start_end_truple[1])
        temp_G[start_end_truple[0]][start_end_truple[1]]["weight"] = 1.

    for segment_id in ways_segment_start_end_id_dict.keys():
        if segment_id not in temp_G.nodes:
            isolated_way_segments.append(segment_id)

    for elem in isolated_way_segments:
        temp_G.add_node(elem)

    print('number of nodes G: ', temp_G.number_of_nodes())
    print('number of edges G: ', temp_G.number_of_edges())
    return temp_G, isolated_way_segments

def update_G_with_attr(G, ways_segment_grade_dict, ways_segment_lanes_dict, ways_segment_all_name_dict,\
                    ways_segment_road_length_dict, ways_segment_speed_dict, ways_segment_start_end_id_dict, \
                    ways_segment_start_end_lans_dict):
    '''
    ways_segment_leida_dict         key=way_seg_id  value=leida_id              str
    ways_segment_road_length_dict   key=way_seg_id  value=road length           int
    ways_segment_grade_dict         key=way_seg_id  value=segment_grade         str
    ways_segment_speed_dict         key=way_seg_id  value=max limit speed       int
    ways_segment_lanes_dict         key=way_seg_id  value=num_of_lanes          int

    ways_segment_start_end_id_dict  key=way_seg_id  value=[start_id, end_id]    str
    ways_segment_start_end_name_dict    key=way_seg_id  value=[start_name, end_name]   int
    ways_segment_start_end_lans_dict    key=way_seg_id  value=[num_of_lanes,num_of_lanes]   int
    '''
    for n,nbrs in G.adjacency():
        G.nodes[n]['segment_grade'] = ways_segment_grade_dict[n]
        G.nodes[n]['num_of_lanes'] = ways_segment_lanes_dict[n]
        G.nodes[n]['all_name'] = ways_segment_all_name_dict[n].split(":")[0]
        G.nodes[n]['road_length'] = ways_segment_road_length_dict[n]
        G.nodes[n]['speed'] = ways_segment_speed_dict[n]
        G.nodes[n]['start_end_id'] = ways_segment_start_end_id_dict[n]
        G.nodes[n]['start_end_lans'] = ways_segment_start_end_lans_dict[n]
    return G

def update_G_with_attr_jinan(G, G_edge_list_attr):
    for n,nbrs in G.adjacency():
        try:
            G.nodes[n]['features'] = G_edge_list_attr[n]
            G.nodes[n]['num_of_lanes'] = G_edge_list_attr[n][3]
        except:
            pass
    return G

def update_G_with_lanes(G):
    for n,nbrs in G.adjacency():
        num_of_lanes_1 = G._node[n]["num_of_lanes"]
        for nbr,attr in nbrs.items():
            num_of_lanes_2 = G._node[nbr]["num_of_lanes"]
            G[n][nbr]["weight"] = calc_weight_by_num_of_lanes(num_of_lanes_1,num_of_lanes_2)
    return G

def calc_weight_by_num_of_lanes(num_of_lanes_1, num_of_lanes_2):
    max_num = max(num_of_lanes_1, num_of_lanes_2)
    min_num = min(num_of_lanes_1, num_of_lanes_2)
    w = liner_fun((max_num-(max_num-min_num))/max_num) # [-6,6]
    return sigmoid_fun(w)

def sigmoid_fun(x):
    return 1/(1+np.exp(-x))

def tanh_fun(x):
    return np.tanh(x)

def liner_fun(x):
    return 15*x-9

def indexing(item_dict, ifpad=False):
	# POI Dictionary
    if ifpad:
        item2id = {"<PAD>":0}
        id2item = ["<PAD>"]

    else:
        item2id = {}
        id2item = []

    for k,v in item_dict.items():

        if k not in item2id:
            # poi2id = {"<PAD>":0, '0':1, '1':2,.......}
            item2id[k] = len(item2id)
            # id2poi = ["<PAD>",'0', '1', .......]
            id2item.append(k)

    return item2id, id2item

def change_dict_key(orign_dict, item2id):
    new_dict = {}
    for k,v in orign_dict.items():
        new_dict[item2id[k]] = v
    return new_dict

def change_dict_value(orign_dict, item2id):
    new_dict = {}
    for k,v in orign_dict.items():
        new_dict[k] = item2id[v]
    return new_dict


def find_second_order_nbrs(node, G):
    second_order_nbrs = []
    first_order_nbrs = list(G[node].keys())
    for elem in first_order_nbrs:
        second_order_nbrs = second_order_nbrs + list(G[elem].keys())
    second_order_nbrs_set = set(second_order_nbrs) - set([node])
    return list(second_order_nbrs_set)

def create_second_order_edge_list(G, isolated_way_segments):
    second_order_edge_list = []
    for n,nbrs in G.adjacency():

        if (G._node[n]["segment_grade"] in ["44000", "45000", "52000"]) and (n not in isolated_way_segments):
            current_second_order_node_list = []
            second_order_nbrs_list = find_second_order_nbrs(n, G)
            for elem in second_order_nbrs_list:
                second_order_edge_list.append((n, elem))
    return second_order_edge_list

def find_third_order_nbrs(node, G):
    second_order_nbrs_list = find_second_order_nbrs(node, G)
    third_order_nbrs_list = []
    for elem in second_order_nbrs_list:
        third_order_nbrs_list = find_second_order_nbrs(elem, G)
    return third_order_nbrs_list

def create_third_order_edge_list(G, isolated_way_segments):
    third_order_edge_list = []
    for n,nbrs in G.adjacency(): #  迭代器
        if (G._node[n]["segment_grade"] in ["44000", "45000"]) and (n not in isolated_way_segments):
            current_third_order_node_list = []
            third_order_nbrs_list = find_third_order_nbrs(n, G)

            for elem in third_order_nbrs_list:
                third_order_edge_list.append((n, elem))
    return third_order_edge_list

def create_fourth_order_edge_list(G, isolated_way_segments):
    fourth_order_edge_list = []
    for n,nbrs in G.adjacency():
        if (G._node[n]["segment_grade"] in [44000]) and (n not in isolated_way_segments):
            current_fourth_order_node_list = []
            third_order_nbrs_list = find_third_order_nbrs(n, G)

            for elem in third_order_nbrs_list:
                fourth_order_edge_list.append((n, elem))
    return fourth_order_edge_list

def create_fourth_order_edge_list(G, isolated_way_segments):
    fourth_order_edge_list = []
    for n,nbrs in G.adjacency():
        if (G._node[n]["segment_grade"] in ["44000"]) and (n not in isolated_way_segments):
            current_fourth_order_node_list = []
            third_order_nbrs_list = find_third_order_nbrs(n, G)

            for elem in third_order_nbrs_list:
                fourth_order_edge_list.append((n, elem))
    return fourth_order_edge_list

def calc_weight_between_two_node(G, node1, node2):
    num_of_lanes_1 = G._node[node1]["num_of_lanes"]
    num_of_lanes_2 = G._node[node2]["num_of_lanes"]

    return calc_weight_by_num_of_lanes(num_of_lanes_1,num_of_lanes_2)

def update_G_with_order(G, all_order_edge_list):
    update_num = 0

    print("all_order_edge_list:",len(all_order_edge_list))
    for idx,item in enumerate(all_order_edge_list):
        if item[0] == item[1]:
            del all_order_edge_list[idx]
    print("all_order_edge_list after delet self-edge:",len(all_order_edge_list))

    for edge in all_order_edge_list:
        if edge[1] not in G[edge[0]].keys():
            update_num = update_num + 1
            weight = calc_weight_between_two_node(G, edge[0], edge[1])
            G.add_edge( edge[0], edge[1])
            G[edge[0]][edge[1]]["weight"] = weight
    print("UPDATE NUM:", update_num)
    print('after order update, number of nodes G: ', G.number_of_nodes())
    print('after order update, number of edges G: ', G.number_of_edges())
    return G

def write_G_to_file(G, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for n,nbrs in G.adjacency():
            for nbr,attr in nbrs.items():
                data = attr['weight']
                f.write(str(n)+'\t'+str(nbr)+'\t'+str(data)+'\n')
                # print(n,'\t',nbr,'\t',data,'\n')
    print('Write G finished!!')

def judeg_road_grade(grade):
    if grade == "44000":
        return 0
    elif grade == "45000":
        return 1
    elif grade == "52000":
        return 2
    else:
        print("道路等级不匹配")
        return -1


def feature_process(features, G):
    '''
    name:
    Date: 2021-03-13 11:58:40
    msg: 遍历图中每个节点的属性6种，维度8，并填补features 一个二维的numpy矩阵，np.float32
        "segment_grade" "num_of_lanes" "road_length" "speed" "start_end_lans"
        # 44000(城市主干道) 4阶；45000(城市次干道) 3阶；52000(县道) 2阶；
        如果是城市主干道 [0]=1  城市次干道 [1]=1    县道 [2]=1
    param {*}
    return {*}
    '''
    for n,nbrs in G.adjacency():
        road_grade_index = judeg_road_grade(G._node[n]["segment_grade"]) # str
        features[n][road_grade_index] = 1.

        num_of_lanes = G._node[n]["num_of_lanes"]
        features[n][3] = num_of_lanes

        road_length = G._node[n]["road_length"]
        features[n][4] = road_length

        speed = G._node[n]["speed"]
        features[n][5] = speed

        num_start_lans, num_end_lans = G._node[n]["start_end_lans"]
        features[n][6] = num_start_lans
        features[n][7] = num_end_lans

    features = nomalize_features(features)

    features = torch.FloatTensor(np.array(features))
    return features

def feature_process_jinan(features, G_edge_list_attr):
    for i in range(len(features)):
        features[i] = np.array(G_edge_list_attr[i])
    features = nomalize_features_jinan(features)
    features = torch.FloatTensor(np.array(features))
    return features

def nomalize_features(features):
    start_clom = 3
    dim_1, dim_2 = features.shape
    eps_ = 1e-5
    features_var = np.var(features, axis=0, keepdims=True) # (1,8) # 按照列
    print("features_var:", features_var)
    features_mean = np.mean(features, axis=0, keepdims=True) # (1,8) # 按照列
    print("features_mean:", features_mean)
    for i in range(start_clom, dim_2):
        features[:, i] = (features[:, i] - features_mean[0][i])/np.sqrt(features_var[0][i] + eps_)
    return features

def nomalize_features_jinan(features):
    start_clom = 3
    dim_1, dim_2 = features.shape
    eps_ = 1e-5
    features_var = np.var(features, axis=0, keepdims=True)
    print("features_var:", features_var)
    features_mean = np.mean(features, axis=0, keepdims=True)
    print("features_mean:", features_mean)

    features[:, 3] = (features[:, 3] - features_mean[0][3])/np.sqrt(features_var[0][3] + eps_)
    return features

def change_dict_key_to_str(orign_dict):
    new_dict = {}
    for k, v in orign_dict.items():
        new_dict[str(k)] = v
    return new_dict

def update_G_with_volume(G, ways_segment_leida_dict, matched_leida_id_set, \
                                hangzhou_5min_slice_volume_dict, leida_direction_dict, leida_lukou_dict):
    matched_way_segments_list = []
    for matched_leida_id in matched_leida_id_set:
        for k,v in ways_segment_leida_dict.items():
            if str(matched_leida_id) == v:
                left_slice_volume,right_slice_volume,stright_slice_volume,sum_slice_volume = find_hangzhou_volume_slice_by_leida(matched_leida_id, hangzhou_5min_slice_volume_dict)
                matched_way_segments_list.append(k)
                G.nodes[k]['leida_id'] = str(matched_leida_id)
                G.nodes[k]['left_volume'] = left_slice_volume
                G.nodes[k]['right_volume'] = right_slice_volume
                G.nodes[k]['stright_volume'] = stright_slice_volume
                G.nodes[k]['sum_volume'] = sum_slice_volume
                G.nodes[k]['leida_direction'] = leida_direction_dict[str(matched_leida_id)]
                G.nodes[k]['lukou'] = leida_lukou_dict[str(matched_leida_id)]
    return G, matched_way_segments_list

def find_hangzhou_volume_slice_by_leida(matched_leida_id, hangzhou_5min_slice_volume_dict):
    sum_slice_volume = []
    left_slice_volume = []
    right_slice_volume = []
    stright_slice_volume = []

    for hour_item, min_slice_volume in hangzhou_5min_slice_volume_dict.items():
        for min_item, leida_volume_info in min_slice_volume.items():
            left_slice_volume.append(leida_volume_info[matched_leida_id]["left_volume"])
            right_slice_volume.append(leida_volume_info[matched_leida_id]["right_volume"])
            stright_slice_volume.append(leida_volume_info[matched_leida_id]["stright_volume"])
            sum_slice_volume.append(leida_volume_info[matched_leida_id]["sum_volume"])

    return left_slice_volume,right_slice_volume,stright_slice_volume,sum_slice_volume


def find_jinan_volume_slice_by_leida(matched_road_id_list, new_jinan_5min_slice_volume_dict):
    ways_segment_volume_dict = {}
    for road_id in matched_road_id_list:
        slice_volume = []
        for day_item, hour_slice_volume in  new_jinan_5min_slice_volume_dict.items():
            for hour_item, min_slice_volume in hour_slice_volume.items():
                for min_item, road_volume_info in min_slice_volume.items():
                    slice_volume.append(road_volume_info[road_id])
        ways_segment_volume_dict[road_id] = slice_volume
    return ways_segment_volume_dict

def preprocess_adj(adj, normalization="AugNormAdj"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # adj
    adj_normalizer = fetch_normalization(normalization) # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    adj = adj_normalizer(adj) # A'
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def get_ways_segment_volume(G, matched_ways_segment_list):
    ways_segment_volume = {}
    for matched_ways_segment in matched_ways_segment_list:
        left_volume = G.nodes[matched_ways_segment]['left_volume']
        right_volume = G.nodes[matched_ways_segment]['right_volume']
        stright_volume = G.nodes[matched_ways_segment]['stright_volume']
        sum_volume = G.nodes[matched_ways_segment]['sum_volume']
        ways_segment_volume[matched_ways_segment] = sum_volume
    return ways_segment_volume

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features) # X=S*X
    precompute_time = perf_counter()-t
    return features, precompute_time

def weight_sgc_precompute(adj, degree):
    weight_adj_list = [0 for i in range(degree)]
    features = torch.eye(adj.shape[0])
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features) # X=S*X
        weight_adj_list[i] = features
    precompute_time = perf_counter()-t
    return weight_adj_list, precompute_time


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def update_score_by_diag(calc_score):
    neg = np.float(-np.inf)
    dim = calc_score.shape[0]
    for i in range(dim):
        calc_score[i][i] = neg
    return calc_score

def calc_sim_score(vector1, vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

def show_pre_info(leida_pre_MAPE_info, leida_pre_RMSE_info, ways_segment_leida_dict):
    for way_id, MAPE in leida_pre_MAPE_info.items():
        cur_leida = ways_segment_leida_dict[way_id]
        cur_MAPE = leida_pre_MAPE_info[way_id]
        cur_RMSE = leida_pre_RMSE_info[way_id]

        print("雷达:"+str(cur_leida)+",MAPE_y:"+str(cur_MAPE)+",RMSE:"+str(cur_RMSE))

def calc_avg_dict_value(example_dict):
    num_value = len(example_dict)
    sum_value = 0.
    for k,v in example_dict.items():
        sum_value = sum_value + v
    return sum_value/num_value

def norm_volume(ways_segment_volume_dict):
    new_ways_segment_volume_dict = {}
    for k,v in ways_segment_volume_dict.items():
        new_ways_segment_volume_dict[k] = []
        for item in v:
            new_ways_segment_volume_dict[k].append(item)

    eps_ = 1e-5
    mean_volume_list = []
    var_volume_list = []

    for k, v in ways_segment_volume_dict.items():
        slice_len = len(v)
        break

    for i in range(slice_len):
        cur_slice_volume_list = []
        for leida,slice_volume_list in ways_segment_volume_dict.items():
            cur_slice_volume_list.append(slice_volume_list[i])
        mean_volume_list.append(np.mean(cur_slice_volume_list))
        var_volume_list.append(math.sqrt(  np.var(cur_slice_volume_list)+ eps_  ))

    for i in range(slice_len):
        for leida,slice_volume_list in ways_segment_volume_dict.items():
            ways_segment_volume_dict[leida][i] = (ways_segment_volume_dict[leida][i]-mean_volume_list[i])/var_volume_list[i]

    return ways_segment_volume_dict, mean_volume_list, var_volume_list, new_ways_segment_volume_dict

def norm_volume_jinan(edge_volume_dict, limit_volume):
    filtered_edge_volume_dict = {}
    for k,v in edge_volume_dict.items():
        if v >= limit_volume:
            filtered_edge_volume_dict[k] = v
    eps_ = 1e-5
    volume_list = []
    for k,v in filtered_edge_volume_dict.items():
        volume_list.append(v)
    volume_var = np.var(np.array(volume_list))
    volume_mean = np.mean(np.array(volume_list))
    for k,v in filtered_edge_volume_dict.items():
        filtered_edge_volume_dict[k] = (v-volume_mean)/(math.sqrt(volume_var)+eps_)

    return filtered_edge_volume_dict, math.sqrt(volume_var), volume_mean

def preprocess_split_data(normed_ways_segment_volume_dict):
    ways_segment_list = []
    ways_slice_volume_list  = []
    normed_ways_segment_volume_dict = sorted(normed_ways_segment_volume_dict.items(), key=lambda item:item[0], reverse = False)

    for ways_volume_truple in normed_ways_segment_volume_dict:
        ways_segment_list.append(ways_volume_truple[0])
        ways_slice_volume_list.append(ways_volume_truple[1])
    return ways_slice_volume_list, ways_segment_list




def combine_ways_segment_volume_dict(test_leida_id_arr, test_volume_arr):
    test_ways_segment_volume_dict = {}
    for i in range(len(test_leida_id_arr)):
        test_ways_segment_volume_dict[test_leida_id_arr[i]] = test_volume_arr[i]
    return test_ways_segment_volume_dict

def inverse_norm(num_arr, volume_sqrt_var, volume_mean):
    num_arr = num_arr*volume_sqrt_var+ volume_mean
    return num_arr


def preprocess_walks_list(walks_list):
    new_list = []
    for walk in tqdm(walks_list):
        new_list = new_list + walk
        # for node in walk:
        #     new_list.append
    return new_list

def find_positive_samples(G):
    adj_weight_dict = {}
    for n,nbrs in G.adjacency():
        adj_weight_dict[n] = dict(G[n])
    return adj_weight_dict


def change_jinan_5min_slice_volume_dict_edge2id(ways_segment2id, jinan_5min_slice_volume_dict):
    day_list = [i for i in range(1,11)]
    hour_list = [i for i in range(8,9)]
    min_list = [i for i in range(0,60,5)]
    new_jinan_5min_slice_volume_dict = {}
    cur_set = find_all_slice_have_volume_cams_set(jinan_5min_slice_volume_dict)
    intersect_ways = set(ways_segment2id.keys()) & cur_set
    for day_item in day_list:
        new_jinan_5min_slice_volume_dict[day_item] = {}
        for hour_item in hour_list:
            new_jinan_5min_slice_volume_dict[day_item][hour_item] = {}
            for min_item in min_list:
                new_jinan_5min_slice_volume_dict[day_item][hour_item][min_item] = {}
                for ways_segment in intersect_ways:
                    new_jinan_5min_slice_volume_dict[day_item][hour_item][min_item][ways_segment2id[ways_segment]] = jinan_5min_slice_volume_dict[day_item][hour_item][min_item][ways_segment]
    matched_road_id = []
    for item in cur_set:
        matched_road_id.append(ways_segment2id[item])
    return new_jinan_5min_slice_volume_dict, matched_road_id

def change_jinan_5min_slice_volume_dict_edge2id_10min(ways_segment2id, jinan_5min_slice_volume_dict):
    day_list = [i for i in range(1,32)]
    hour_list = [i for i in range(7,9)]
    min_list = [i for i in range(0,60,10)]
    new_jinan_5min_slice_volume_dict = {}
    cur_set = find_all_slice_have_volume_cams_set_10min(jinan_5min_slice_volume_dict)
    intersect_ways = set(ways_segment2id.keys()) & cur_set # len(intersect_ways)=37
    for day_item in day_list:
        new_jinan_5min_slice_volume_dict[day_item] = {}
        for hour_item in hour_list:
            new_jinan_5min_slice_volume_dict[day_item][hour_item] = {}
            for min_item in min_list:
                new_jinan_5min_slice_volume_dict[day_item][hour_item][min_item] = {}
                for ways_segment in intersect_ways:
                    new_jinan_5min_slice_volume_dict[day_item][hour_item][min_item][ways_segment2id[ways_segment]] = jinan_5min_slice_volume_dict[day_item][hour_item][min_item][ways_segment]
    matched_road_id = []
    for item in cur_set:
        matched_road_id.append(ways_segment2id[item])
    return new_jinan_5min_slice_volume_dict, matched_road_id


def find_all_slice_have_volume_cams_set(jinan_5min_slice_volume_dict):
    day_list = [i for i in range(1,11)]
    hour_list = [i for i in range(8,9)]     #TODO: 12slice
    min_list = [i for i in range(0,60,5)]

    cur_set = set(jinan_5min_slice_volume_dict[1][8][0].keys()) # TODO: 12slice
    for day_item in day_list:
        for hour_item in hour_list:
            for min_item in min_list:
                cur_set = set(jinan_5min_slice_volume_dict[day_item][hour_item][min_item].keys()) & cur_set
    return cur_set

def find_all_slice_have_volume_cams_set_10min(jinan_5min_slice_volume_dict):

    day_list = [i for i in range(1,32)]
    hour_list = [i for i in range(7,9)]
    min_list = [i for i in range(0,60,10)]

    cur_set = set(jinan_5min_slice_volume_dict[1][7][0].keys())
    cur_len = len(cur_set)
    cur_day, cur_hour, cur_min = day_list[0], hour_list[0], min_list[0]
    for day_item in day_list:
        for hour_item in hour_list:
            for min_item in min_list:
                if cur_len >= len(set(jinan_5min_slice_volume_dict[day_item][hour_item][min_item].keys())):
                    cur_day, cur_hour, cur_min = day_item, hour_item, min_item
                    cur_len = len(set(jinan_5min_slice_volume_dict[day_item][hour_item][min_item].keys()))
                cur_set = set(jinan_5min_slice_volume_dict[day_item][hour_item][min_item].keys()) & cur_set
    return cur_set

def test_dict_v_sum(test_dict):
    num= 0
    for k,v in test_dict.items():
        num = num + v
    return num


def gen_affinity_graph_direct_edge(edge_list, num_slice, id2ways_segment):
    new_edge_list = []
    isolated_way_segments = []
    num_node = len(id2ways_segment)
    edge_node_set = set()
    for edge_tuple in edge_list:
        for node in edge_tuple:
            edge_node_set.add(node)

    for i in range(num_node):
        if i not in edge_node_set:
            isolated_way_segments.append(i)

    for i in range(num_slice):
        for edge_tuple in edge_list:
            new_edge_list.append(  (  edge_tuple[0]+(i*num_node),edge_tuple[1]+(i*num_node)  ) )

    for i in range(num_slice):
        for edge_tuple in edge_list:
            new_edge_list.append(  (  edge_tuple[0]+(i*num_node),edge_tuple[1]+(i*num_node)  ) )

    return new_edge_list, isolated_way_segments

def mask_list(volume_mean_list, value):
    for idx in range(len(volume_mean_list)):
        volume_mean_list[idx] = value
    return volume_mean_list

def change_tuple_elem(G_edge_list, ways_segment2id):
    new_G_edge_list = []
    for elem in G_edge_list:
        new_G_edge_list.append((ways_segment2id[elem[0]], ways_segment2id[elem[1]]))
    return new_G_edge_list

def check_model_param(model):

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
