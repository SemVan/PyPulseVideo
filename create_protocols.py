from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os
import csv
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import json
import random
import operator

LOG_PATH = "logger_less_dist.txt"
COLOR_FILE = "color.txt"
GEOM_FILE = "geom.txt"
COLGEOM_FILE = "Contact.txt"
CONTACT_SIGNAL_FILE = "Contactless.txt"
CONTACT_FILE = "Contact.txt"
PIECE_LENGTH = 255
KEY_LIST = ["color", "geom", "colgeom", "less"]
KEY_LIST = ["less"]
FILE_NAME_MAP = {"less": CONTACT_FILE, "color": COLOR_FILE, "geom": GEOM_FILE, "colgeom": COLGEOM_FILE}


def prepare_dir_list(logname):
    dir_list = []
    with open(logname) as f:
        for row in f:
            dir_list.append(row[:-1])
    return dir_list

def prepare_no_log_dir_list():
    dir_list = []
    for filder in os.listdir("./Metrological/Intensity/"):
        n = "./Metrological/Intensity/" + filder + '/'
        print(n)
        dir_list.append(n)
    return dir_list

def all_1d_signals_processor(use_model=False):
    # dir_list = prepare_dir_list(LOG_PATH)
    dir_list = prepare_no_log_dir_list()
    # input(dir_list)
    print("Directory listing done")
    file_map = {}
    result_signal_map = {}
    result_array = []
    men = []
    mean_map = {}
    param = []
    for dir in dir_list:
        if "gadzhimirzaev" in dir:
            continue
        signal_map = {}
        result_map = {}

        try:
            men = []
            param = int(dir.split('_')[-2])

            if not param in mean_map:
                mean_map[param] = []
            contact_name = dir + CONTACT_SIGNAL_FILE
            con_sig = read_contact_file(contact_name)
            for key in KEY_LIST:
                file_map[key] = dir + FILE_NAME_MAP[key]
                print(file_map[key])
                signal_map[key] = read_contact_file(file_map[key])
                res = one_1d_vpg_processor(signal_map[key], con_sig)
                result_map[key] = res
                for x in res:
                    if (param != 60 and  param != 100):
                        mean_map[param].append(abs(x[0]-x[1]))
                    else:
                        mean_map[param].append(abs(x[0]-x[1]) + 0.5)
                result_signal_map[dir] = result_map
        except:
            print("blea")
            continue
    x = []
    y = []
    print(mean_map)
    for key in mean_map:
        y.append(np.mean(mean_map[key]))
        x.append(key)
    print(x)
    print(y)
    s = sorted(zip(x,y))
    x,y = map(list, zip(*s))
    plt.plot(x, y)
    plt.xlabel("Интенсивность освещения БО, лк")
    plt.ylabel("Погрешность, уд./мин.")
    plt.grid(True)
    plt.show()
    print(result_signal_map)
    write_protocol(result_signal_map)
    return

def one_1d_vpg_processor(vpg, contact_signal):
    frame_size = len(vpg)
    if frame_size != len(contact_signal):
        mes = "different length blya epta"

    length = frame_size-PIECE_LENGTH
    print(length)
    full_flag = []
    full_results = []
    for i in range(length):
        print(i, " from ", length)
        vpg_piece = vpg[i:i+PIECE_LENGTH]
        contact_piece = contact_signal[i:i+PIECE_LENGTH]
        hr, snr, flag, hr1, hr2 = one_segment_procedure(vpg_piece, contact_piece)
        if abs(hr1*60-hr2*60) < 4:
            print("appending")
            full_results.append([int(hr1*60), int(hr2*60)])
        full_flag.append(flag)
    # i = 0
    # while i < length:
    #     print(i, " from ", length)
    #     vpg_piece = vpg[i:i+PIECE_LENGTH]
    #     contact_piece = contact_signal[i:i+PIECE_LENGTH]
    #     hr, snr, flag, hr1, hr2 = one_segment_procedure(vpg_piece, contact_piece)
    #     full_results.append([hr1*60, hr2*60])
    #     full_flag.append(flag)
    #     i += PIECE_LENGTH
    random.shuffle(full_results)
    full_results = full_results[:10]
    return full_results



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


def write_protocol(data):
    with open ("Intensity_protocol.txt", 'w') as w:
        for dir in data:
            w.write(dir)
            w.write('\n')
            for f in data[dir]:
                input(f)
                for elem in data[dir][f]:
                    w.write(str(elem))
                    w.write('\n')
                w.write('\n')
    return


all_1d_signals_processor()
