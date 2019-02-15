from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *


FILES_PATH = "./Segmented/Signals/"
PIECE_LENGTH = 255

def all_signals_processor():
    for filename in os.listdir(VIDEO_PATH):
        signal = read_segmented_file(file_name)
        sig_res = one_vpg_processor(signal)

    return

def one_vpg_processor(vpg):
    vpg_piece_3ch = vpg
    vpg_piece_weighted = get_channels_sum(vpg_piece_3ch)
    phase_mask = full_frame_phase_mask(vpg_piece)
    hr_mask, snr_mask = full_frame_amp_procedure(vpg_piece)

    return [phase_mask, hr_mask, snr_mask]
