from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os

FILES_PATH = "./Segmented/Signals/"
SIGNAL_FILE = "signal.csv"
PIECE_LENGTH = 255

def all_signals_processor():
    dir_list = []
    for i,j,y in os.walk(FILES_PATH):
        dir_list.append(i)

    for dir in dir_list[1:]:
        print(dir)
        file_name = dir + "/" + SIGNAL_FILE
        if os.path.isfile(file_name):
            signal = read_segmented_file(file_name)
            print("file read")
            sig_res = one_vpg_processor(signal)
            write_metrics(sig_res, dir)
    return

def one_vpg_processor(vpg):
    frame_size = vpg.shape[0]
    full_phase = []
    full_hr = []
    full_snr = []
    full_flag = []
    length = frame_size-PIECE_LENGTH
    for i in range(length):
        print(i, " from ", length)
        vpg_piece_3ch = vpg[i:i+PIECE_LENGTH]
        vpg_piece_weighted = get_channels_sum(vpg_piece_3ch) #changes shape
        phase_mask = full_fragment_phase_mask(vpg_piece_weighted)
        hr_mask, snr_mask, flag_mask = 0,0,0#full_fragment_amp_procedure(vpg_piece_weighted)
        full_phase.append(phase_mask)
        full_hr.append(hr_mask)
        full_snr.append(snr_mask)
        full_flag.append(flag_mask)
    return [np.asarray(full_phase), np.asarray(full_hr), np.asarray(full_snr), np.asarray(full_flag)]

all_signals_processor()
