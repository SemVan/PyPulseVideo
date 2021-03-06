from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os

FILES_PATH = "./Metrological/Intensity/"
CONTACT_FILES_PATH = "./Metrological/Intensity/"
SIGNAL_FILE = "signal_int.csv"
CONTACT_SIGNAL_FILE = "Contactless.txt"
PIECE_LENGTH = 64


def prepare_dir_list(root_dir, force = False):
    dir_list = []
    for i,j,y in os.walk(FILES_PATH):
        if not i == FILES_PATH:
            files_list = os.listdir(i)
            if force:
                dir_list.append(i)
            else:
                if not ("phase.csv" in files_list or "flag.csv" in files_list):
                    dir_list.append(i)

    return dir_list


def prepare_dir_list_from_logger(logname):
    dir_list = []
    with open(logname) as logger:
        for row in logger:
            dir_list.append(row[:-1])

    return dir_list

def all_signals_processor():
    dir_list = []
    dir_list = prepare_dir_list(FILES_PATH, force = True)
    # dir_list = prepare_dir_list_from_logger("seglogger_intensity.txt")
    for dir in dir_list:
        print(dir)
    print()
    # for i,j,y in os.walk(FILES_PATH):
    #     dir_list.append(i)

    print("Directory listing done")
    for dir in dir_list:
        print(dir)
        last_name = dir.split('/')
        # contact_dir = CONTACT_FILES_PATH + last_name[-1][:-4] + "/" + CONTACT_SIGNAL_FILE
        contact_dir = CONTACT_FILES_PATH + last_name[-1] + "/" + CONTACT_SIGNAL_FILE
        # final_dir = FILES_PATH + last_name[-1][:-4]
        final_dir = FILES_PATH + last_name[-1]
        file_name = final_dir + '/' + SIGNAL_FILE
        # input(file_name)


        if not os.path.isfile(contact_dir):
            mes = "No contact file blya " + " " + contact_dir
            print(mes)
            write_log(mes)
            continue

        if os.path.isfile(file_name):
            con_sig = read_contact_file(contact_dir)
            signal = read_segmented_file(file_name)
            print("file read")
            sig_res = one_vpg_processor(signal, con_sig)
            print()
            print("Gonna save to ", sig_res)
            print()
            write_metrics(sig_res, final_dir)
        else:
            print("suck my dict")

    return

def one_vpg_processor(vpg, contact_signal):
    frame_size = vpg.shape[0]
    if frame_size != len(contact_signal):
        mes = "different length blya epta"
        write_log(mes)
    full_phase = []
    full_hr = []
    full_snr = []
    full_flag = []
    length = frame_size-PIECE_LENGTH
    for i in range(length):
        print(i, " from ", length)
        vpg_piece_3ch = vpg[i:i+PIECE_LENGTH]
        contact_piece = contact_signal[i:i+PIECE_LENGTH]
        vpg_piece_weighted = get_channels_sum(vpg_piece_3ch) #changes shape
        phase_mask = full_fragment_phase_mask(vpg_piece_weighted)
        hr_mask, snr_mask, flag_mask = full_fragment_amp_procedure(vpg_piece_weighted, contact_piece)
        full_phase.append(phase_mask)
        full_hr.append(hr_mask)
        full_snr.append(snr_mask)
        full_flag.append(flag_mask)
    return [np.asarray(full_phase), np.asarray(full_hr), np.asarray(full_snr), np.asarray(full_flag)]

def write_log(message):
    with open ("log_int_1.txt", "a+") as file:
        file.write(message)
    return

all_signals_processor()
