from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.signal import correlate
from scipy.signal import butter, lfilter, lfilter_zi, welch, convolve2d
from scipy.stats import mannwhitneyu

DESCRETISATION_PERIOD = 0.001

def plot_signals(ch1, ch2, offset):
    if offset>=0:
        ch2 = ch2[int(offset):]
    else:
        ch1 = ch1[:int(offset)]
    plt.plot(range(len(ch1)), ch1, color='red')
    plt.plot(range(len(ch2)), ch2, color='green')
    plt.show()
    return


def get_phase_shift(sig1, sig2):
    cross_corr = correlate(sig2, sig1, 'full', 'direct')
    # plt.plot(range(len(cross_corr)), cross_corr)
    # plt.show()
    shift = np.argmax(cross_corr) - len(sig1)
    return shift*DESCRETISATION_PERIOD


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y


def norm_signal(sp):
    max = np.max(sp)
    norm = []
    for i in range(len(sp)):
        norm.append(sp[i] / max)
    return norm


def get_spectra(signal):
    period = 1/1000
    complex_four = np.fft.fft(signal)
    spectra = np.absolute(complex_four)
    freqs = []
    for i in range(len(signal)):
        freqs.append(1/(period*len(signal))*i)

    plt.plot(freqs, spectra)
    plt.ylim([0, 20])
    plt.show()
    return spectra, freqs


def shift_matrix_procedure(mat, st_shift, mat_name):
    for part in mat:
        part[part>0.07] = 0.07
    for part in mat:
        part[part<-0.07] = -0.07

    mean_mat = np.mean(mat, axis = 0) - np.mean(st_shift)
    mean_mat = get_shift_from_center(mean_mat)

    row_mean = np.mean(mean_mat, axis = 0)
    col_mean = np.mean(mean_mat, axis = 1)
    r_diff = np.max(row_mean)-np.min(row_mean)
    c_diff = np.max(col_mean)-np.min(col_mean)
    surface_mean = get_roi_means(mean_mat)
    # show_surface_distribution(mean_mat)
    # show_surface_distribution(surface_mean)
    plt.show()
    # plot_phase_distribution(row_mean, col_mean, r_diff, c_diff, mat_name)
    return col_mean, row_mean, surface_mean

def get_shift_from_center(mat):
    new_mat = mat - mat[4][3]
    return new_mat

def plot_phase_distribution(rows, cols, row_diff, col_diff, fig_name):
    f = plt.figure()
    plt.subplot(2,1,1)
    plt.title("Horizontal   " + fig_name + "   " + str(row_diff))
    plt.plot(range(len(rows)), rows)
    plt.grid()

    plt.subplot(2,1,2)
    plt.title("Vertical   " + fig_name + "   " + str(col_diff))
    plt.plot(range(len(cols)), cols)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return f


def distrib_procedure(slice):
    m = np.mean(slice, axis=0)

    std = np.std(slice, axis=0)

    k = 1
    l = m + k*std
    u = m - k*std

    return m, l, u, std


def show_surface_distribution(mat):
    plt.figure()
    plt.imshow(mat, cmap='plasma')
    plt.legend()
    plt.show()

def get_roi_means(mat):
    kernel = np.full(shape=(4,3), fill_value=1)
    res = convolve2d(mat, kernel, mode='valid')
    res = res/12
    new_res = np.ndarray(shape=(2,2))
    new_res[0][0] = res[0][0]
    new_res[0][1] = res[0][-1]
    new_res[1][0] = res[-1][0]
    new_res[1][1] = res[-1][-1]
    return new_res

def full_signals_procedure(ch1, ch2):

    ch1 = butter_bandpass_filter(ch1, 0.1, 5, 1000, 3)
    ch2 = butter_bandpass_filter(ch2, 0.1, 5, 1000, 3)
    ch1 = norm_signal(ch1)
    ch2 = norm_signal(ch2)
    # get_spectra(sig1)
    sh = get_phase_shift(ch1, ch2)
    print(sh)
    return ch1, ch2, sh


def get_mann_whitney_result(data):
    level = 23
    d_t = np.transpose(data)
    sh = d_t.shape
    u_test = np.zeros(shape = (sh[0], sh[0]))

    for i in range(sh[0]):
        for j in range(sh[0]):
            res, p = mannwhitneyu(d_t[i], d_t[j])
            u_test[i][j] = int(res<=level)
    print(u_test)

    return
