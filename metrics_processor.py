from metrics_io import *
import os
import numpy as np
import json
from matplotlib import pyplot as plt

FILES_PATH = "./Metrological/Intensity/"
CONTACT_FILES_PATH = "./Metrological/Intensity/"
SIGNAL_FILE = "signal_int.csv"
CONTACT_SIGNAL_FILE = "Contactless.txt"
PIECE_LENGTH = 64
ALGO_NAME = "net"

def prepare_dir_list(root_dir):
    dir_list = []
    for i,j,y in os.walk(FILES_PATH):
        if not i == FILES_PATH:
            files_list = os.listdir(i)
            if ("phase.csv" in files_list and "flag.csv" in files_list):
                dir_list.append(i)
    for dir in dir_list:
        print(dir)
    input()
    return dir_list



def full_metrics_processor():
    dir_list = prepare_dir_list(FILES_PATH)
    full_summ_frag = 0
    true_sum_flag = 0
    good_signal_counter = 0
    dir_map = []
    full_hr = []
    for dir in dir_list:
        try:
            metrics = read_metrics(dir)
        except:
            continue
        flag = metrics[-1]
        hr = metrics[1]
        for i in range(flag.shape[0]):
            for j in range(flag.shape[1]):
                for k in range(flag.shape[2]):
                    if flag[i][j][k] == 1:
                        if   [k] >0 and hr[i][j][k] < 200:

                            full_hr.append(hr[i][j][k])
        flag = flag.astype(int)
        print(dir)
        tp = np.sum(flag == 1)/ np.sum(flag != 0)
        print("TRUE PERCENTAGE ", np.sum(flag == 1)/ np.sum(flag != 0))
        print("FRAGMENT SIZE ", flag.shape[0])
        full_summ_frag += flag.shape[0]
        cnt = 0
        for elem in flag:
            if np.sum(elem == 1) != 0:
                cnt += 1
        if cnt>0:
            good_signal_counter += 1
        print("TRUE COUNTER ", cnt, "FROM ", flag.shape[0])
        true_sum_flag += cnt
        print("TRUE PERCENTAGE BY FRAMES ", cnt / flag.shape[0])
        print()
        tp = cnt / flag.shape[0]
        dir_map.append({dir: {"net": tp}})

    plt.hist(full_hr, bins = 50)
    plt.show()
    write_json('distances_1d_net.json', dir_map)
    print("TOTAL TRUE ", true_sum_flag)
    print("TOTAL FRAGMENTS ", full_summ_frag)
    print("SIGNAL COUNTER ", len(dir_list))
    print("GOOD SIGNAL COUNTER ", good_signal_counter)
    print(true_sum_flag/full_summ_frag)

    with open ("intensity_net_1.json", 'w') as f:
        json.dump(dir_map, f)

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
    return

full_metrics_processor()
