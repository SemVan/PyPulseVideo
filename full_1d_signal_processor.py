from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os


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
            dir_list.append(row)
    return dir_list

def all_1d_signals_processor(use_model=False):
    dir_list = prepare_dir_list(LOG_PATH)
    for dir in dir_list:
        print(dir)

    print("Directory listing done")
    file_map = {}
    signal_map = {}
    result_map = {}
    for dir in dir_list:
        print(dir)
        contact_name = dir + CONTACT_SIGNAL_FILE
        con_sig = read_contact_file(contact_name)
        for key in KYE_LIST:
            file_map[key] = dir + FILE_NAME_MAP[key]
            signal_map[key] = read_contact_file(file_map[key])
            result_map[key] = one_1d_vpg_processor(signal, con_sig)

    return

def one_1d_vpg_processor(vpg, contact_signal):
    frame_size = len(vpg)
    if frame_size != len(contact_signal):
        mes = "different length blya epta"

    length = frame_size-PIECE_LENGTH
    full_flag = []
    for i in range(length):
        print(i, " from ", length)
        vpg_piece = vpg[i:i+PIECE_LENGTH]
        contact_piece = contact_signal[i:i+PIECE_LENGTH]
        hr, snr, flag = one_segment_procedure(vpg_piece, contact_piece)
        full_flag.append(flag)
    return full_flag


def measurement_statistics(result_map):
    for key in result_map:
        print()
        print(key)
        
    return
