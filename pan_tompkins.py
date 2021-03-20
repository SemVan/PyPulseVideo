import numpy as np
from scipy.signal import convolve
from matplotlib import pyplot as plt
from segmented_io import *
from general_math import *
from peakdetect import peakdetect
from scipy.interpolate import interp1d

PERIOD = 1/60
FS = 60#1 / PERIOD


def pan_tompkins_algo(signal):
    # signal = butter_bandpass_filter(signal, 0.1, 10, FS, 5)
    norm_signal = normalize(signal)
    deriv = get_derivative(norm_signal)
    sq_deriv = get_square(deriv)
    integr = get_integration(sq_deriv, 5)
    integr = normalize(integr)
    norm_signal = normalize(norm_signal)
    return integr, norm_signal


def get_derivative(signal):
    kernel = np.asarray([-1, 0, 1])
    out_signal = convolve(signal, kernel)
    return out_signal


def get_square(signal):
    return np.square(signal)


def get_integration(signal, window_size):
    kernel = np.ones(window_size)
    out_signal = convolve(signal, kernel)
    return out_signal


def normalize(signal):
    s = signal / np.max(np.abs(signal))
    s = s - np.mean(s)
    return s


def reverse_signal(signal):
    signal = np.max(signal)-signal
    return signal


def interpolation(signal, ratio):
    f = interp1d(range(len(signal)), signal, kind='cubic')
    step = 1.0 / ratio
    newx = np.linspace(0, len(signal)-1, len(signal)*ratio)
    return f(newx), newx


def hard_peaks(signal):
    signal = np.asarray(signal) / np.max(signal)
    peaks2 = peakdetect(signal,lookahead=31)
    peaks_max = []
    for i in peaks2[0]:
        peaks_max.append(i[1])

    peaks_max_ind = []
    for j in peaks2[0]:
        peaks_max_ind.append(j[0])

    peaks_min = []
    for i in peaks2[1]:
        peaks_min.append(i[1])

    peaks_min_ind = []
    for j in peaks2[1]:
        peaks_min_ind.append(j[0])
    x1 = np.arange(0, len(signal))
    y1 = np.array(signal)
    return np.asarray(peaks_max_ind) + 2


def from_new_x_to_old_x(peak_idx):
    to_old_idx = peak_idx / 10
    return to_old_idx.astype(np.int) - 1


def full_pan_tompkins(signal):
    file_name = "./Metrological/Distances/akishin_1000_60_3/Contactless.txt"
    # file_name_2 = "./Metrological/Distances/akishin_1000_60_3/colgeom.txt"
    sig_cont = np.asarray(read_contact_file(file_name))
    # sig_contactless = read_contactless_file(file_name_2)
    sig_contact, newx = interpolation(sig_cont, 10)
    sig_cont = reverse_signal(sig_cont)
    # sig_contact = reverse_signal(sig_contact)
    res, signal = pan_tompkins_algo(sig_contact)

    indices = hard_peaks(res)
    old_idx = from_new_x_to_old_x(indices)
    max_dots = signal[indices]
    max_dots_old = sig_cont[old_idx]
    plt.figure()
    plt.plot(range(len(signal)), signal)
    plt.plot(range(len(res)), res)
    plt.scatter(indices, max_dots, color='red')
    plt.figure()
    plt.plot(range(len(sig_cont)), sig_cont)
    plt.scatter(old_idx, max_dots_old, color='black')
    plt.show()


# full_pan_tompkins([])
