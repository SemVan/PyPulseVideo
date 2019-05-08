from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
import os

FILES_PATH = "./Segmented/Signals/"
SIGNAL_FILE = "signal.csv"
PIECE_LENGTH = 255

def all_signals_processor():
    dir_list = list(os.walk(FILES_PATH))[0]
    for dir in dir_list:
        file_name = FILES_PATH + '/' + dir + '/' + SIGNAL_FILE
        signal = read_segmented_file(file_name)
        sig_res = one_vpg_processor(signal)

    return

def one_vpg_processor(vpg):
    frame_size = vpg.shape[0]
    full_phase = []
    full_hr = []
    full_snr = []
    full_flag = []
    for i in range(frame_size-PIECE_LENGTH):
        vpg_piece_3ch = vpg[i:i+PIECE_LENGTH]
        vpg_piece_weighted = get_channels_sum(vpg_piece_3ch) #changes shape
        phase_mask = full_frame_phase_mask(vpg_piece)
        hr_mask, snr_mask, flag_mask = full_frame_amp_procedure(vpg_piece)
        full_phase.append(phase_mask)
        full_hr.append(hr_mask)
        full_snr.append(snr_mask)
        full_flag.append(flag_mask)
    return [np.asarray(full_phase), np.asarray(full_hr), np.asarray(full_snr), np.asarray(full_flag)]
