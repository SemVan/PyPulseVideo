from segmented_io import *
from phase_processor import *
from amp_processor import *
from general_math import *
from metrics_io import *
import os
from scipy.signal import medfilt
import json



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
            sig_res, glued = one_vpg_processor(signal, con_sig)
            print()
            print("Gonna save to ", sig_res)
            print()
            write_metrics(sig_res, final_dir)
            # write_glued_signal(glued, final_dir)
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
    final_1d_signal = []
    fsignal = []
    prev = 0
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

        # one_frame_counter = 0
        # off_piece = 0
        # p = np.zeros(frame_size)
        # goods = []
        # for k in range(flag_mask.shape[0]):
        #     for j in range(flag_mask.shape[1]):
        #         if flag_mask[k][j] == 1:
        #             goods.append(vpg_piece_weighted[k][j])
        #             one_frame_counter += 1
        #             off_piece += (vpg_piece_weighted[k][j][0] + vpg_piece_weighted[k][j][1] + vpg_piece_weighted[k][j][2])/float(3)
        #             p1 = get_signal_piece_with_offset(vpg_piece_weighted[k][j], i, frame_size)
        #             p = np.add(p, p1)
        #
        # if one_frame_counter > 0:
        #     off_piece = float(off_piece)/one_frame_counter
        #     p = p/float(one_frame_counter)
        # else:
        #     off_piece = prev
        # prev = off_piece
        # if np.sum(p) != 0:
        #     fsignal.append(p)
        #     # for g in goods:
        #     #     plt.plot(range(len(g)), norm(g))
        #     # plt.plot(range(len(p)),p, label="final")
        #     # plt.plot(range(len(contact_piece)), norm(contact_piece), label='contact')
        #     # plt.legend()
        #     # plt.show()
        # final_1d_signal.append(off_piece)
    # print(np.asarray(fsignal).shape)
    true_final = []
    #
    # for i in range(len(fsignal)):
    #     s = 0
    #     cnt = 0
    #     for j in range(len(fsignal[i])):
    #         if fsignal[i][j] != 0:
    #             s += fsignal[i][j]
    #             cnt += 1
    #     true_final.append(float(s)/float(cnt))
    # true_final = butter_bandpass_filter(true_final, 0.1, 5, 33, order=5)
    # true_final = norm(true_final)
    # true_final = reverse_norm(true_final)
    # contact_signal = butter_bandpass_filter(contact_signal, 0.1, 5, 33, order=5)
    # contact_signal = norm(contact_signal)
    # contact_signal = reverse_norm(contact_signal)
    # plt.plot(range(len(true_final)), true_final)
    # plt.plot(range(len(contact_signal)), contact_signal)
    # plt.show()
    return [np.asarray(full_phase), np.asarray(full_hr), np.asarray(full_snr), np.asarray(full_flag)], true_final


def get_signal_piece_with_offset(signal_piece, offset, length):
    signal = np.zeros(length)
    for i in range(len(signal_piece)):
        signal[i+offset] = signal_piece[i]
    return norm(signal)

def reverse_norm(sig):
    return 1-sig

def norm(sig):
    sig = np.asarray(sig)
    sig = sig / float(np.max(np.abs(sig)))
    return sig


def write_glued_signal(signal, dir):
    findir = dir + "/glued.json"
    with open(findir, 'w') as f:
        json.dump({"glued":signal.tolist()}, f)
    return


def write_log(message):
    with open ("log_int_1.txt", "a+") as file:
        file.write(message)
    return

all_signals_processor()
