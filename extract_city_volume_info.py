from utils import *
from random import choice

def read_snap_shot_volume_one_hour(fname):

    leida_time_volume = {}
    data = pd.read_csv(fname, encoding="gbk",keep_default_na=False)
    slice_data = data.iloc[:,0]
    for i in range(len(slice_data)):
        time_snap  = int(data.iloc[i][0].split(":")[0])
        leida_id   = data.iloc[i][1]
        cur_volume = data.iloc[i][2]
        if leida_id not in leida_time_volume.keys():
            leida_time_volume[leida_id] = {}
            if time_snap not in leida_time_volume[leida_id].keys():
                leida_time_volume[leida_id][time_snap] = cur_volume
            else:
                leida_time_volume[leida_id][time_snap] = leida_time_volume[leida_id][time_snap] + cur_volume
        else:
            if time_snap not in leida_time_volume[leida_id].keys():
                leida_time_volume[leida_id][time_snap] = cur_volume
            else:
                leida_time_volume[leida_id][time_snap] = leida_time_volume[leida_id][time_snap] + cur_volume
    return leida_time_volume

def read_snap_shot_volume_five_min(fname):

    leida_time_volume = {}
    data = pd.read_csv(fname, encoding="gbk",keep_default_na=False)
    slice_data = data.iloc[:,0]
    for i in range(len(slice_data)):
        hour_tag  = int(data.iloc[i][0].split(":")[0])
        min_tag   = int(data.iloc[i][0].split(":")[1])
        leida_id   = data.iloc[i][1]
        sum_volume = data.iloc[i][2]
        left_volume = data.iloc[i][3]
        stright_volume = data.iloc[i][4]
        right_volume = data.iloc[i][5]

        if leida_id not in leida_time_volume.keys():
            leida_time_volume[leida_id] = {}
            if hour_tag not in leida_time_volume[leida_id].keys():
                leida_time_volume[leida_id][hour_tag] = {}
                if min_tag not in leida_time_volume[leida_id][hour_tag].keys():
                    leida_time_volume[leida_id][hour_tag][min_tag] = {}
                    leida_time_volume[leida_id][hour_tag][min_tag]["left_volume"] = int(left_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["stright_volume"] = int(stright_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["right_volume"] = int(right_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["sum_volume"] = int(sum_volume)
                else:
                    print("error0!")

            else:
                if min_tag not in leida_time_volume[leida_id][hour_tag].keys():
                    leida_time_volume[leida_id][hour_tag][min_tag] = {}
                    leida_time_volume[leida_id][hour_tag][min_tag]["left_volume"] = int(left_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["stright_volume"] = int(stright_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["right_volume"] = int(right_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["sum_volume"] = int(sum_volume)
                else:
                    print("error0!")

        else:
            if hour_tag not in leida_time_volume[leida_id].keys():
                leida_time_volume[leida_id][hour_tag] = {}
                if min_tag not in leida_time_volume[leida_id][hour_tag].keys():
                    leida_time_volume[leida_id][hour_tag][min_tag] = {}
                    leida_time_volume[leida_id][hour_tag][min_tag]["left_volume"] = int(left_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["stright_volume"] = int(stright_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["right_volume"] = int(right_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["sum_volume"] = int(sum_volume)
                else:
                    print("error0!")

            else:
                if min_tag not in leida_time_volume[leida_id][hour_tag].keys():
                    leida_time_volume[leida_id][hour_tag][min_tag] = {}
                    leida_time_volume[leida_id][hour_tag][min_tag]["left_volume"] = int(left_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["stright_volume"] = int(stright_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["right_volume"] = int(right_volume)
                    leida_time_volume[leida_id][hour_tag][min_tag]["sum_volume"] = int(sum_volume)
                else:
                    print("error0!")
    return leida_time_volume

def test_read_snap_shot_volume_five_min(leida_time_volume):
    '''test read_snap_shot_volume_five_min fun'''
    all_leida_volume = 0
    leida_hour_volume = {}
    for k1,v1 in leida_time_volume.items():
        for k2,v2 in v1.items():
            cur_leida_volume = 0
            for k3,v3 in v2.items():
                cur_leida_volume = v3["sum_volume"] + cur_leida_volume
                all_leida_volume = all_leida_volume + v3["sum_volume"]
    return all_leida_volume

def preprocess_leida_time_volume_dict(leida_time_volume):
    temp1_leida_time_volume_dict, temp2_leida_time_volume_dict = {}, {}
    for k,v in leida_time_volume.items():
        if len(v) == 24:
            temp1_leida_time_volume_dict[k] = v

    bad_leida_set = set()
    for k1,v1 in temp1_leida_time_volume_dict.items():
        for k2,v2 in v1.items():
            if v2 == 0:
                bad_leida_set.add(k1)
    for k1,v1 in temp1_leida_time_volume_dict.items():
        if k1 not in bad_leida_set:
            temp2_leida_time_volume_dict[k1] = v1
    return temp2_leida_time_volume_dict

def intersect_dict(leida_time_volume_one_hour, leida_time_volume_five_min):

    complete_leida = set(leida_time_volume_one_hour.keys())
    new_dict = dict()
    for k,v in leida_time_volume_five_min.items():
        if k in complete_leida:
            new_dict[k] = v
    return new_dict

def intersect_46_leida_set_dict( leida_set, leida_time_volume_five_min, start_time, end_time):

    start_hour = eval(start_time.split('_')[0])
    start_min  = eval(start_time.split('_')[1])
    end_hour   = eval(end_time.split('_')[0])
    end_min    = eval(end_time.split('_')[1])

    start_time = start_hour * 60 + start_min
    end_time   = end_hour * 60 + end_min
    num_slice = int((end_time-start_time)/5) + 1

    new_leida_time_slice_volume_dict = {}
    for leida, hour_slice_volume in leida_time_volume_five_min.items():
        if str(leida) in leida_set:
            new_leida_time_slice_volume_dict[leida] = {}
            for hour, min_slice_volume in hour_slice_volume.items():
                for minute, volume_info in min_slice_volume.items():

                    if hour*60+minute >= start_time and hour*60+minute <= end_time and volume_info['sum_volume']!=0:
                        if hour not in new_leida_time_slice_volume_dict[leida].keys():
                            new_leida_time_slice_volume_dict[leida][hour] = {}
                            new_leida_time_slice_volume_dict[leida][hour][minute] = volume_info
                        else:
                            new_leida_time_slice_volume_dict[leida][hour][minute] = volume_info

    len_not_satisfy = []
    for leida, hour_slice_volume in new_leida_time_slice_volume_dict.items():
        cur_len = 0
        for hour, min_slice_volume in hour_slice_volume.items():
            cur_len = cur_len + len(min_slice_volume)
        if cur_len != num_slice:
            len_not_satisfy.append(leida)
    for leida in len_not_satisfy:
        new_leida_time_slice_volume_dict.pop(leida)

    return new_leida_time_slice_volume_dict



def read_txt_file(fname, seperater=" "):

    data_dict = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip(" \n").split(seperater)
            if len(line) == 4:
                pass
            else:
                data_dict[line[0]] = int(line[1])
    return data_dict

def read_all_volume_file_jinan(file_path):

    day_tag = [i for i in range(1,11)]
    hour_tag = [i for i in range(8,9)]
    min_tag = [i for i in range(0,60,5)]
    month = "8"
    jinan_5min_slice_volume_dict = {} # {day:  { hour:{  min:{edge:volume}  }  }
    for day in day_tag:
        for hour in hour_tag:
            for minute in min_tag:
                fname = str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(minute+4)+".volume"
                edge_volume_dict = read_txt_file(file_path+"/"+fname, seperater=",")
                if day not in jinan_5min_slice_volume_dict.keys():
                    jinan_5min_slice_volume_dict[day] = {}
                    if hour not in jinan_5min_slice_volume_dict[day]:
                        jinan_5min_slice_volume_dict[day][hour] = {}

                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                    # hour
                    else:
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                # day
                else:
                    if hour not in jinan_5min_slice_volume_dict[day]:
                        jinan_5min_slice_volume_dict[day][hour] = {}
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                    # hour
                    else:
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
    return jinan_5min_slice_volume_dict

def read_all_volume_file_jinan_10min(file_path):

    day_tag = [i for i in range(1,32)]
    hour_tag = [i for i in range(7,9)]
    min_tag = [i for i in range(0,60,10)]
    month = "8"
    jinan_5min_slice_volume_dict = {} # {day:  { hour:{  min:{edge:volume}  }  }
    for day in day_tag:
        for hour in hour_tag:
            for minute in min_tag:
                fname = str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + "_" + str(minute+9)+".volume"
                edge_volume_dict = read_txt_file(file_path+"/"+fname, seperater=",")
                if day not in jinan_5min_slice_volume_dict.keys():
                    jinan_5min_slice_volume_dict[day] = {}
                    if hour not in jinan_5min_slice_volume_dict[day]:
                        jinan_5min_slice_volume_dict[day][hour] = {}

                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                    # hour
                    else:
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                # day
                else:
                    if hour not in jinan_5min_slice_volume_dict[day]:
                        jinan_5min_slice_volume_dict[day][hour] = {}
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
                    # hour
                    else:
                        jinan_5min_slice_volume_dict[day][hour][minute] = edge_volume_dict
    return jinan_5min_slice_volume_dict

def change_hangzhou_5min_volume_dict(leida_time_volume_five_min):
    all_leida_set = set(leida_time_volume_five_min.keys())
    a_sample = choice(list(all_leida_set))
    hangzhou_5min_slice_volume_dict = dict()
    hour_list = [i for i in range(24)]
    min_list  = [i for i in range(0,60,5)]
    for hour_item in hour_list:
        hangzhou_5min_slice_volume_dict[hour_item] = {}
        for min_item in min_list:
            hangzhou_5min_slice_volume_dict[hour_item][min_item] = {}
            for leida in all_leida_set:
                hangzhou_5min_slice_volume_dict[hour_item][min_item][leida] = {}

    for k1,v1 in hangzhou_5min_slice_volume_dict.items():
        for k2,v2 in v1.items():
            for leida in all_leida_set:
                try:
                    hangzhou_5min_slice_volume_dict[k1][k2][leida]["left_volume"]    = leida_time_volume_five_min[leida][k1][k2]["left_volume"]
                    hangzhou_5min_slice_volume_dict[k1][k2][leida]["right_volume"]   = leida_time_volume_five_min[leida][k1][k2]["right_volume"]
                    hangzhou_5min_slice_volume_dict[k1][k2][leida]["stright_volume"] = leida_time_volume_five_min[leida][k1][k2]["stright_volume"]
                    hangzhou_5min_slice_volume_dict[k1][k2][leida]["sum_volume"] = leida_time_volume_five_min[leida][k1][k2]["sum_volume"]
                except:
                    continue
    for hour_item in hour_list:
        for min_item in min_list:
            if(len(hangzhou_5min_slice_volume_dict[hour_item][min_item][a_sample])==0):
                hangzhou_5min_slice_volume_dict[hour_item].pop(min_item)
    for hour_item in hour_list:
        if(len(hangzhou_5min_slice_volume_dict[hour_item])==0):
            hangzhou_5min_slice_volume_dict.pop(hour_item)

    return hangzhou_5min_slice_volume_dict

def extract_hangzhou_volume_info():
    leida_time_volume_one_hour = read_snap_shot_volume_one_hour('hangzhou/雷达过车0113_5分钟_1hour.csv')
    leida_time_volume_five_min = read_snap_shot_volume_five_min('hangzhou/雷达过车0113_5分钟_1hour.csv')
    leida_time_volume_five_min = intersect_46_leida_set_dict(read_pkl("hangzhou/46leida_set.pkl"), leida_time_volume_five_min, '7_55', '8_50') # 雷达188 只有直行有流量，故删去
    hangzhou_5min_slice_volume_dict = change_hangzhou_5min_volume_dict(leida_time_volume_five_min)
    return hangzhou_5min_slice_volume_dict, leida_time_volume_five_min, set(leida_time_volume_five_min.keys())

def according_attr_return_list(num_of_lanes, road_grade, liminted_speed):
    attr_list = [0 for i in range(7)]
    if road_grade == '主干':
        attr_list[0] = 1
    elif road_grade == '次干':
        attr_list[1] = 1
    elif road_grade == '支路':
        attr_list[2] = 1
    attr_list[3] = int(num_of_lanes)
    if liminted_speed == '40':
        attr_list[4] = 1
    elif liminted_speed == '50':
        attr_list[5] = 1
    elif liminted_speed == '60':
        attr_list[6] = 1
    attr_list[3] = int(num_of_lanes)
    return attr_list

def read_attr_file_jinan(fname, seperater):

    flag = True
    cams_way_segments_dict = {}
    way_segments_cams_dict = {}
    cams_attr_dict = {}
    way_segments_attr_dict = {}

    all_way_segments_list = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if flag:
                flag = False
                continue
            line = line.strip(" \n").split(seperater)
            cams_way_segments_dict[line[0]] = line[1]
            way_segments_cams_dict[line[1]] = line[0]
            cams_attr_dict[line[0]]         = according_attr_return_list(line[2], line[3], line[4])
            way_segments_attr_dict[line[1]] = according_attr_return_list(line[2], line[3], line[4])
            all_way_segments_list.append(line[1])

    way_freq = Counter(all_way_segments_list).most_common()
    return cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict, way_segments_attr_dict

def read_roadnet_file_jinan(fname, seperater):

    edge_list = []
    G_edge_list = []
    G_edge_list_attr = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip(" \n").split(seperater)
            start_node   = line[0].split('_')[0]
            end_node     = line[0].split('_')[1]
            num_of_lanes = line[1]
            road_grade   = line[2]
            liminted_speed = line[3]

            cur_attr_list = according_attr_return_list(num_of_lanes, road_grade, liminted_speed)
            edge_list.append((start_node, end_node))
            edge_list.append((end_node, start_node))
            G_edge_list_attr[start_node+'_'+end_node] = cur_attr_list
            G_edge_list_attr[end_node+'_'+start_node] = cur_attr_list
    for i,edge_tuple_1 in enumerate(edge_list):
        for j,edge_tuple_2 in enumerate(edge_list):

            if edge_tuple_1[1] == edge_tuple_2[0] and edge_tuple_1[0] != edge_tuple_2[1]:
                s = edge_tuple_1[0] + '_' + edge_tuple_1[1]
                e = edge_tuple_2[0] + '_' + edge_tuple_2[1]
                G_edge_list.append((s,e))

    return edge_list, G_edge_list, G_edge_list_attr

def extract_jinan_volume_info():
    jinan_5min_slice_volume_dict = read_all_volume_file_jinan("jinan/5min_slice_volume")
    cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict, way_segments_attr_dict =\
                    read_attr_file_jinan('jinan/cams_attr.txt', ',')
    edge_list, G_edge_list, G_edge_list_attr = read_roadnet_file_jinan('jinan/roadnet.txt',',')

    return jinan_5min_slice_volume_dict, \
           cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict,\
           G_edge_list, G_edge_list_attr


def extract_jinan_volume_info_10min():
    jinan_5min_slice_volume_dict = read_all_volume_file_jinan_10min("jinan/10min_slice_volume")
    cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict, way_segments_attr_dict =\
                    read_attr_file_jinan('jinan/cams_attr.txt', ',')
    edge_list, G_edge_list, G_edge_list_attr = read_roadnet_file_jinan('jinan/roadnet.txt',',')

    return jinan_5min_slice_volume_dict, \
           cams_way_segments_dict, way_segments_cams_dict, cams_attr_dict,\
           G_edge_list, G_edge_list_attr


def select_max_cams_interval(jinan_5min_slice_volume_dict):
    all_cams_list = []
    for day_item,hour_list in jinan_5min_slice_volume_dict.items():
        for hour_item, min_list in hour_list.items():
            for min_item, volume_list in min_list.items():
                cur_set = set(volume_list.keys())
                all_cams_list.append(cur_set)
    start_idx_list = [24*i for i in range(0,31)]



    slice_set_dict = {}
    check_len = 12
    for k in range(24-check_len+1):
        for i in start_idx_list:            # 0
            for j in range(check_len):             # 0
                cur_set = all_cams_list[i + k + j]
                if k not in slice_set_dict.keys():
                    slice_set_dict[k] = []
                    slice_set_dict[k].append(cur_set)
                else:
                    slice_set_dict[k].append(cur_set)


    cams_num_list = []
    cur_idx = -1
    for item,list_set in slice_set_dict.items():
        init_set = list_set[0]
        for cur_set in list_set:
            init_set = init_set & cur_set
        cams_num_list.append(len(init_set))
    return slice_set_dict