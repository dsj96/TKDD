import random
from sklearn.model_selection import train_test_split
import itertools

import matplotlib.pyplot as plt
import optuna
import time
'''

self.optimizer = optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
'''

from utils import *
from args import *
from CTVI_model import *
from metrics import *
from walk import RWGraph
from extract_city_volume_info import *
from extract_features_graph import *
from attention import Multi_Head_SelfAttention

def matched_cams_plot(way_segments_cams_dict, matched_road_id_list):
    matched_cams = []
    for road_id in matched_road_id_list:
        matched_cams.append(way_segments_cams_dict[road_id])
    with open('matched_cams.txt','w', encoding='utf-8') as f:
        for item in matched_cams:
            f.write(str(item)+'\n')

def matual_split_data(normed_ways_segment_volume_dict):
    test_ways_list = [465, 4, 331, 121, 69, 65, 50, 139]
    train_ways_segment_volume_dict = {}
    test_ways_segment_volume_dict  = {}
    train_ways_set = set(normed_ways_segment_volume_dict.keys()) - set(test_ways_list)
    for test_way in test_ways_list:
        test_ways_segment_volume_dict[test_way] = normed_ways_segment_volume_dict[test_way]
    for train_way in train_ways_set:
        train_ways_segment_volume_dict[train_way] = normed_ways_segment_volume_dict[train_way]
    return train_ways_segment_volume_dict, test_ways_segment_volume_dict

def objective(trial):
    '''0.optuna hyper-parameters'''
    config = {
        "epochs" : trial.suggest_int('epochs',  10, 10),
        "hy_RW"  : trial.suggest_uniform('hy_RW' , 0.1, 10),
        "hy_volume_current" : trial.suggest_uniform('hy_volume_current' , 0.1, 10),
    }

    args = get_args()

    t = perf_counter()
    '''1.read data'''
    all_info = extract_jinan_volume_info()
    jinan_5min_slice_volume_dict, \
    cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict,\
    G_edge_list, G_edge_list_attr = all_info



    '''2.indexing'''
    ways_segment2id, id2ways_segment = indexing(G_edge_list_attr,ifpad=False)
    way_segments_cams_dict = change_dict_key(way_segments_cams_dict, ways_segment2id)
    cams_way_segments_dict = change_dict_value(cams_way_segments_dict, ways_segment2id)
    G_edge_list = change_tuple_elem(G_edge_list, ways_segment2id)
    G_edge_list_attr = change_dict_key(G_edge_list_attr, ways_segment2id)
    new_jinan_5min_slice_volume_dict, matched_road_id_list = change_jinan_5min_slice_volume_dict_edge2id(ways_segment2id, jinan_5min_slice_volume_dict)


    '''3.construct graph'''
    G_0, isolated_way_segments = get_G_from_edges_jinan(G_edge_list, G_edge_list_attr)
    G_1 = update_G_with_attr_jinan(G_0, G_edge_list_attr) # edge_volume_dict
    G_2 = update_G_with_lanes(G_1)




    ways_segment_volume_dict = find_jinan_volume_slice_by_leida(matched_road_id_list, new_jinan_5min_slice_volume_dict)


    ''''process features adj '''
    features = np.zeros((len(G_edge_list_attr),7),dtype=np.float32)
    sadj = nx.adjacency_matrix(G_2)
    if args.cuda:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cuda') # FloatTensor
        sadj = preprocess_adj(sadj, normalization=args.normalization).to('cuda')
        fadj = load_feature_graph('jinan', features, args.k_knn).to('cuda')
    else:
        features = feature_process_jinan(features, G_edge_list_attr).to(device='cpu')
        sadj = preprocess_adj(sadj, normalization=args.normalization).to('cpu')
        fadj = load_feature_graph('jinan', features, args.k_knn).to('cpu')


    '''normallize'''
    normed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, unnormed_ways_segment_volume_dict = norm_volume(ways_segment_volume_dict)
    volume_sqrt_var_list = mask_list(volume_sqrt_var_list, 1.)
    volume_mean_list = mask_list(volume_mean_list, 0.)


    '''delet abnormal sensors & splite data'''
    unnormed_ways_segment_volume_dict.pop(109) # 85
    unnormed_ways_segment_volume_dict.pop(71)
    # unnormed_ways_segment_volume_dict.pop(85)
    if args.matual_split:
        train_ways_segment_volume_dict, test_ways_segment_volume_dict = matual_split_data(unnormed_ways_segment_volume_dict)
    else:
        data_feature, data_target = preprocess_split_data(unnormed_ways_segment_volume_dict)
        train_volume_arr, test_volume_arr, train_leida_id_arr, test_leida_id_arr = \
                train_test_split(data_feature, data_target, test_size=args.percent, random_state=args.seed)

        train_volume_arr_, valid_volume_arr, train_leida_id_arr_, valid_leida_id_arr = \
                train_test_split(train_volume_arr, train_leida_id_arr, test_size=args.percent, random_state=args.seed)

        train_ways_segment_volume_dict = combine_ways_segment_volume_dict(train_leida_id_arr, train_volume_arr)
        valid_ways_segment_volume_dict = combine_ways_segment_volume_dict(valid_leida_id_arr, valid_volume_arr)
        test_ways_segment_volume_dict  = combine_ways_segment_volume_dict(test_leida_id_arr, test_volume_arr)


    set_seed(args.seed, args.cuda)


    '''train & evaluate model'''
    model = JINAN_model(num_head=args.num_head ,num_slice=args.num_slice, nfeat=features.shape[1], nhid=args.hidden, nclass=args.output_dim, dropout=args.dropout, degree=args.degree)

    leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info, mean_mape = train_regression(model, features, train_ways_segment_volume_dict, valid_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, G_2, sadj, fadj, args.weight_decay, args.lr, args.dropout, config)
    show_pre_info(leida_pre_MAPE_info_y, leida_pre_RMSE_info, way_segments_cams_dict)
    print("="*40)
    test_regression(model, features, train_ways_segment_volume_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var_list, volume_mean_list, sadj, fadj, config)
    print("="*40)
    print("over!")
    return mean_mape


def objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, topk, negk):
    pre_volume  = []
    true_volume = []
    loss_term = 0.
    for k1,v1 in train_ways_segment_vec_dict.items():
        num_slice = v1.shape[0]
        for i in range(num_slice):
            curr_score_dict, recent_score_dict, period_score_dict = {}, {}, {}
            for k2, v2 in train_ways_segment_vec_dict.items():
                if(k1 != k2):
                    curr_score   = torch.cosine_similarity(v1[i], v2[i], dim=-1)
                    curr_score_dict[k2]   = curr_score
            cur_sum_volume_max, cur_sum_sim_score        =  0, .0
            cur_sorted_score_dict_max    = sorted(curr_score_dict.items(),   key=lambda item:item[1], reverse = True)[:topk]
            for truple in cur_sorted_score_dict_max:
                cur_sum_volume_max = cur_sum_volume_max + train_ways_segment_volume_dict[truple[0]][i]*truple[1]
                cur_sum_sim_score = cur_sum_sim_score + truple[1]
            cur_pre_volume = cur_sum_volume_max / cur_sum_sim_score
            loss_term = abs((train_ways_segment_volume_dict[k1][i] - cur_pre_volume))**3 + loss_term
    return loss_term






def train_regression(model, train_features, train_ways_segment_volume_dict,
                     test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, G, sadj, fadj,
                     weight_decay, lr, dropout, config ):
    epochs = config['epochs']
    hy_RW = config['hy_RW']
    hy_volume_current = config['hy_volume_current']
    hy_volume_recent = config['hy_volume_recent']
    hy_volume_daily = config['hy_volume_daily']
    hy_volume_weekly = config['hy_volume_weekly']



    '''objective_rw'''
    walker = RWGraph(G)
    walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted)

    walks_list = [col for row in walks_list for col in row]
    vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]

    word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    adj_weight_dict = find_positive_samples(G)
    vocab_list = [item[0] for item in vocab_list]

    criterion = nn.MSELoss()
    train_ways_segment_list = list(train_ways_segment_volume_dict.keys())

    all_epoch_mape_y,all_epoch_mape_y_head, all_epoch_RMSE = [], [], []
    params_list = []
    for i in range(args.num_slice):
        params_list.append({"params":model.model_list[i].parameters()})

    for i in range(args.num_head):
        params_list.append({"params":model.attention.at_block_list[i].parameters()})


    optimizer = optim.Adam( params_list, lr=lr, weight_decay=weight_decay)
    t = perf_counter()

    per_list = []
    for epoch in range(epochs):
        train_ways_segment_vec_dict = {}
        model.train()
        optimizer.zero_grad()

        output = model(train_features, sadj, fadj)

        for i, item in enumerate(train_ways_segment_list):
            train_ways_segment_vec_dict[item] = output[:, item, :]
            
        loss_train_rw = objective_rw(train_ways_segment_vec_dict, args.negk, adj_weight_dict, output, vocab_list, word_freqs)
        loss_train_volume_current = objective_volume_current(train_ways_segment_volume_dict, train_ways_segment_vec_dict, args.topk, args.negk)

        loss = hy_volume_current*loss_train_volume_current + hy_RW*loss_train_rw

        loss.backward()
        optimizer.step()

        if (epoch) % 2 == 0:
            print('Validating...')
            with open('jinan/valid_log.txt', 'a', encoding='utf-8') as f:
                with torch.no_grad():
                    train_ways_segment_vec_dict = {}
                    model.eval()
                    output = model(train_features, sadj, fadj)
                    for i, item in enumerate(train_ways_segment_list):
                        train_ways_segment_vec_dict[item] = output[:, item]

                    leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info = evaluate_metric(epoch, output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, args.topk, volume_sqrt_var, volume_mean)
                    print("epoch:{}\tloss:{}".format(epoch, loss))
                    print("MAPE_y: {}".format(leida_pre_MAPE_info_y))
                    print("MAPE_y_head: {}".format(leida_pre_MAPE_info_y_head))
                    print("RMSE: {}".format(leida_pre_RMSE_info))

                    print("mean MAPE_y: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
                    print("mean MAPE_y_head: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
                    print("mean RMSE: {}".format(calc_avg_dict_value(leida_pre_RMSE_info)))

                    per_list.append(calc_avg_dict_value(leida_pre_MAPE_info_y))

                    f.write("epoch: {}\n".format(epoch))
                    f.write("mean MAPE_y: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
                    f.write("mean MAPE_y_head: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
                    f.write("mean RMSE: {}\n".format(calc_avg_dict_value(leida_pre_RMSE_info)))
                    all_epoch_mape_y.append(calc_avg_dict_value(leida_pre_MAPE_info_y))
                    all_epoch_mape_y_head.append(calc_avg_dict_value(leida_pre_MAPE_info_y_head))
                    all_epoch_RMSE.append(calc_avg_dict_value(leida_pre_RMSE_info))
    train_time = perf_counter()-t
    print('In this trial, avg_mape_y:{},  avg_mape_y_head:{},   avg_RMSE：{}'.format(np.mean(all_epoch_mape_y), np.mean(all_epoch_mape_y_head), np.mean(all_epoch_RMSE)))
    return leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head, leida_pre_RMSE_info, min(per_list)


def test_regression(model, train_features, train_ways_segment_volume_dict,
                     test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, volume_sqrt_var, volume_mean, sadj, fadj, config ):
    epochs = config['epochs']

    train_ways_segment_list = list(train_ways_segment_volume_dict.keys())
    all_epoch_mape_y,all_epoch_mape_y_head, all_epoch_RMSE = [], [], []
    t = perf_counter()
    per_list = []
    train_ways_segment_vec_dict = {}

    model.eval()
    output = model(train_features, sadj, fadj)

    for i, item in enumerate(train_ways_segment_list):
        train_ways_segment_vec_dict[item] = output[:, item, :]


    print('Testing...')
    with open('jinan/test_log.txt', 'a', encoding='utf-8') as f:
        with torch.no_grad():
            leida_pre_MAPE_info_y, leida_pre_MAPE_info_y_head,leida_pre_RMSE_info = evaluate_metric(epochs, output, train_ways_segment_volume_dict, train_ways_segment_vec_dict, test_ways_segment_volume_dict, unnormed_ways_segment_volume_dict, args.topk, volume_sqrt_var, volume_mean)

            print("MAPE_y: {}".format(leida_pre_MAPE_info_y))
            print("MAPE_y_head: {}".format(leida_pre_MAPE_info_y_head))
            print("RMSE: {}".format(leida_pre_RMSE_info))

            print("mean MAPE_y: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
            print("mean MAPE_y_head: {}".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
            print("mean RMSE: {}".format(calc_avg_dict_value(leida_pre_RMSE_info)))

            per_list.append(calc_avg_dict_value(leida_pre_MAPE_info_y))

            f.write("mean MAPE_y: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y)))
            f.write("mean MAPE_y_head: {}\n".format(calc_avg_dict_value(leida_pre_MAPE_info_y_head)))
            f.write("mean RMSE: {}\n".format(calc_avg_dict_value(leida_pre_RMSE_info)))
            all_epoch_mape_y.append(calc_avg_dict_value(leida_pre_MAPE_info_y))
            all_epoch_mape_y_head.append(calc_avg_dict_value(leida_pre_MAPE_info_y_head))
            all_epoch_RMSE.append(calc_avg_dict_value(leida_pre_RMSE_info))
    train_time = perf_counter()-t


if __name__=="__main__":
    args = get_args()

    '''0. optuna train'''
    study_name = 'jinan_study'
    args = get_args()
    study = optuna.create_study(study_name=study_name, storage='sqlite:///jinan/log/jinan_study.db', load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)
    print('over!')

    # '''1. read db file, find the best parameters'''
    # study = optuna.create_study(study_name='jinan_study', storage='sqlite:///jinan/log/jinan_study_100_seed2.db', load_if_exists=True)
    # df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    # print(df)
    # df.to_csv('jinan\log\{}_jinan_study.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))))
    # print('over!')


    # '''2. visualization'''
    # import plotly as py
    # study = optuna.create_study(study_name='jinan_study', storage='sqlite:///jinan/log/jinan_study_100_sed2.db', load_if_exists=True)
    # fig = optuna.visualization.plot_parallel_coordinate(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent', 'hy_volume_daily', 'hy_volume_weekly'])
    # py.offline.plot(fig,auto_open=True) # filename="iris1.html"
    # print('over!')

