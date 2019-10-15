from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os
import csv
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

LOG_PATH = "logger.txt"
COLOR_FILE = "color.txt"
GEOM_FILE = "geom.txt"
COLGEOM_FILE = "colgeom.txt"
CONTACT_SIGNAL_FILE = "Contact.txt"
PIECE_LENGTH = 255
KEY_LIST = ["color", "geom", "colgeom"]
FILE_NAME_MAP = {"color": COLOR_FILE, "geom": GEOM_FILE, "colgeom": COLGEOM_FILE}


def prepare_dir_list(logname):
    dir_list = []
    with open(logname) as f:
        for row in f:
            dir_list.append(row[:-1])
    return dir_list

def all_1d_signals_processor(use_model=False):
    dir_list = prepare_dir_list(LOG_PATH)
    print("Directory listing done")
    file_map = {}
    result_signal_map = {}
    for dir in dir_list:
        signal_map = {}
        result_map = {}

        try:
            contact_name = dir + CONTACT_SIGNAL_FILE
            con_sig = read_contact_file(contact_name)
            for key in KEY_LIST:
                file_map[key] = dir + FILE_NAME_MAP[key]
                print(file_map[key])
                signal_map[key] = read_contactless_file(file_map[key])
                result_map[key] = one_1d_vpg_processor(signal_map[key], con_sig)
            print(result_map)
            result_signal_map[dir] = result_map
        except:
            print("blea")
            continue
    measurement_statistics(result_signal_map)
    return

def one_1d_vpg_processor(vpg, contact_signal):
    frame_size = len(vpg)
    if frame_size != len(contact_signal):
        mes = "different length blya epta"

    length = frame_size-PIECE_LENGTH
    print(length)
    full_flag = []
    for i in range(length):
        print(i, " from ", length)
        vpg_piece = vpg[i:i+PIECE_LENGTH]
        contact_piece = contact_signal[i:i+PIECE_LENGTH]
        hr, snr, flag = one_segment_procedure(vpg_piece, contact_piece)
        print(flag)
        full_flag.append(flag)
    return full_flag


def measurement_statistics(result_map):
    algo_metric = {}
    for dir in result_map:
        for key in result_map[dir]:
            if not key in algo_metric:
                algo_metric[key] = []
            a = np.asarray(result_map[dir][key])
            a[a < 0] = 0
            algo_metric[key].append(np.count_nonzero(a)/ len(a))
    input(algo_metric)
    write_1d_metric(algo_metric)
    return

def stat_sign(algo_metric):
    for key in algo_metric:
        print (key, np.mean(algo_metric[key]))
        for key2 in algo_metric:
            t, p = ttest_ind(algo_metric[key], algo_metric[key2])
            print(key2, t, p)
        print()
        # plt.plot(range(len(algo_metric[key])), algo_metric[key])
        # plt.set_label(key)
    # plt.show()

def write_1d_metric(resmap):
    with open ("1d_stat.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for dir in resmap:
            row = [dir]
            for elem in resmap[dir]:
                row.append(elem)
            writer.writerow(row)

def read_metrics():
    data = {}
    with open("1d_stat.csv") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data[row[0]] = [float(x) for x in row[1:]]
    return data

def read_contactless_file(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(" ")
            data[0].append(rowList[0])
            if len(rowList)==2:
                data[1].append(float(rowList[1]))
            else:
                if (float(rowList[0])+float(rowList[2])>0):
                    data[1].append(float(rowList[1])/(float(rowList[0])+float(rowList[2])))
                else:
                    data[1].append(float(rowList[1]))
    return get_y_reverse_signal(np.asarray(data[1]))


all_1d_signals_processor()
d = read_metrics()
stat_sign(d)
