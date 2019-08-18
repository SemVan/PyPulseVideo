from metrics_io import *
import os
import numpy as np


FILES_PATH = "./Segmented/Signals/"
CONTACT_FILES_PATH = "./Signals/"
SIGNAL_FILE = "signal.csv"
CONTACT_SIGNAL_FILE = "Contact.txt"
PIECE_LENGTH = 255


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
    for dir in dir_list:
        metrics = read_metrics(dir)
        flag = metrics[-1]
        flag = flag.astype(int)
        print(dir)
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
    print("TOTAL TRUE ", true_sum_flag)
    print("TOTAL FRAGMENTS ", full_summ_frag)
    print("SIGNAL COUNTER ", len(dir_list))
    print("GOOD SIGNAL COUNTER ", good_signal_counter)
    print(true_sum_flag/full_summ_frag)




full_metrics_processor()
