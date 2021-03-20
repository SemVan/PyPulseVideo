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

LOG_PATH = "logger_less_dist.txt"
COLOR_FILE = "color.txt"
GEOM_FILE = "geom.txt"
COLGEOM_FILE = "colgeom.txt"
CONTACT_SIGNAL_FILE = "Contactless.txt"
CONTACT_FILE = "Contact.txt"
PIECE_LENGTH = 255
KEY_LIST = ["color", "geom", "colgeom"]
# KEY_LIST = ["less"]
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

def RR_processor(use_model=False):
    dir_list = prepare_no_log_dir_list()
    print("Directory listing done")
    file_map = {}
    result_signal_map = {}
    result_array = []
    for dir in dir_list:
        try:
            signal_map = {}
            result_map = {}
            contact_name = dir + CONTACT_FILE
            wt_name = dir + "RR_" + CONTACT_FILE
            con_sig = read_contact_file(contact_name)
            RRs = RR_procedure(con_sig)
            write_RRs(RRs,wt_name)
            for key in KEY_LIST:
                rd_name = dir + FILE_NAME_MAP[key]
                wt_name = dir + "RR_" + FILE_NAME_MAP[key]
                sig = read_contactless_file(rd_name)
                RRs = RR_procedure(sig)
                write_RRs(RRs,wt_name)
        except:
            print("No files in: " + dir)
    return


def write_RRs(RRs,name):
    with open(name,'w') as txtfile:
        for elem in RRs:
            txtfile.write(str(elem)+"\n")
    return


def read_contactless_file(fileName):
    # if os.path.isfile(fileName):
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



RR_processor()