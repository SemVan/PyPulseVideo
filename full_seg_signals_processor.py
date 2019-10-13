from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os

FILES_PATH = "./Segmented/Signals/"
CONTACT_FILES_PATH = "./Signals/"
SIGNAL_FILE = "signal.csv"
CONTACT_SIGNAL_FILE = "Contact.txt"
PIECE_LENGTH = 255


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

def all_signals_processor():
    dir_list = []
    dir_list = prepare_dir_list(FILES_PATH, force = True)
    for dir in dir_list:
        print(dir)
    input()
    # input()
    # for i,j,y in os.walk(FILES_PATH):
    #     dir_list.append(i)

    print("Directory listing done")
    for dir in dir_list[1:]:
        print(dir)
        last_name = dir.split('/')
        contact_dir = CONTACT_FILES_PATH + last_name[-1] + "/" + CONTACT_SIGNAL_FILE
        file_name = FILES_PATH + last_name[-1] + '/' + "signal.csv"
        if not os.path.isfile(contact_dir):
            mes = "No contact file blya " + " " + contact_dir
            input()
            print(mes)
            write_log(mes)
            continue

        if os.path.isfile(file_name):
            con_sig = read_contact_file(contact_dir)
            signal = read_segmented_file(file_name)
            print("file read")
            input()
            sig_res = one_vpg_processor(signal, con_sig)
            print()
            print("Gonna save to ", sig_res)
            print()
            write_metrics(sig_res, dir)
        else:
            print("suck my dict")
            imput()
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
    with open ("log.txt", "a+") as file:
        file.write(message)
    return

all_signals_processor()
