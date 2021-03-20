from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from classification import *
from peakdetect import peakdetect
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, lfilter_zi, welch, convolve2d, convolve

PERIOD = 4/100
WINDOW_SIZE = 1000

Fs = 25
Ts = 1 / Fs
FACTOR = 1

def full_fragment_amp_procedure(vpg, con_sig):
    vpg_shape = vpg.shape
    vpg_hr = np.zeros(shape=vpg_shape[:-1])
    vpg_snr = np.zeros(shape=vpg_shape[:-1])
    vpg_flag = np.zeros(shape=vpg_shape[:-1])
    for i in range(vpg_shape[0]):
        for j in range (vpg_shape[1]): #changed shape element number
            hr, snr, flag = one_segment_procedure(vpg[i][j], con_sig)
            vpg_hr[i][j] = hr
            vpg_snr[i][j] = snr
            vpg_flag[i][j] = flag
    return vpg_hr, vpg_snr, vpg_flag

def one_segment_procedure(segment_signal, contact_signal,p_window,d_window):
#     spectrum, freqs = get_fourier_result(segment_signal, PERIOD)
#     px, py = simple_peaks(spectrum, freqs, np.arange(1,2))
#     fc, fc_amp = getSpectrumCentralFrequencyAndAmp(px, py)
#     SNR = get_SNR(spectrum, fc_amp)

#     signal_flag = onePairProcedure(contact_signal, segment_signal)
    freq1, maxes1 = full_contact_signal_procedure(contact_signal)
    freq2, maxes2 = full_contact_signal_procedure(segment_signal, p_window=p_window, d_window=d_window)
    signal_flag = -1
    if abs(freq1*60-freq2*60) <= 3:
        signal_flag = 1
#     return np.abs(freq2*60)//1, SNR, signal_flag, freq1, freq2
    return np.abs(freq1*60)//1, signal_flag # - for integr_measurements

def RR_procedure(signal):
    signal = np.asarray(signal) / np.max(signal)
    peaks2 = peakdetect(signal,lookahead=int(0.32*FACTOR/Ts))
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

    peaks_diff = []
    for i in range(len(peaks_max_ind)-1):
        peaks_diff.append(peaks_max_ind[i+1]-peaks_max_ind[i])
    peaks_diff = np.asarray(peaks_diff)*Ts/FACTOR
    
    return peaks_diff


def get_fourier_result (signal, period):
    complex_four = np.fft.fft(signal)
    spectra = np.absolute(complex_four)
    freqs = []
    for i in range(len(signal)):
        freqs.append(1/(period*len(signal))*i)
    return spectra, freqs

def get_SNR(spectr, central_freq_amp):
    SNR = central_freq_amp/np.sum(spectr)
    return SNR

def normalizeSpectrum(spectra):
    nSp = [spectra[i]/spectra[0] for i in range(len(spectra))]
    return nSp

def getSpectrumCentralFrequencyAndAmp(peakX, peakY):
    z = list(zip(peakY, peakX))
    peaks_cut = [z[i][0] for i in range(len(z)) if (z[i][1] > 0.5 and z[i][1] < 3.0)]
    peaks_cut_x = [z[i][1] for i in range(len(z)) if (z[i][1] > 0.5 and z[i][1] < 3.0)]
    if len(peaks_cut) == 0:
        return -1, -1
    maxY = np.max(peaks_cut)
    maxX = peaks_cut_x[np.argmax(peaks_cut)]
    #print("freq max ", maxX)
    return maxX, maxY

def simple_peaks(signal, xAx, dotSize):
    x = []
    y = []
    for i in range(1, len(signal)-1):
        if (signal[i]>signal[i+1] and signal[i]>signal[i-1]):
            x.append(xAx[i])
            y.append(signal[i])
    return x,y

def get_median(data):
    dsort = sorted(data)
    return dsort[int(len(data)/2)]

def hard_peaks(signal, d_window=0.32):
    signal = np.asarray(signal) / np.max(signal)
    peaks2 = peakdetect(signal,lookahead=int(d_window*FACTOR/Ts))
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

    peaks_diff = []
    for i in range(len(peaks_max_ind)-1):
        peaks_diff.append(peaks_max_ind[i+1]-peaks_max_ind[i])
    freq = -100
    if len(peaks_diff) > 0:
        freq = 1/(np.mean(peaks_diff)*Ts/FACTOR)

    # x1 = np.arange(0, len(signal))
    # y1 = np.array(signal)
    # plt.plot(x1, y1)
    # # plt.plot(peaks_max_ind, peaks_max, 'x')
    # plt.plot(peaks_min_ind, peaks_min, 'x')
    # plt.show()
    # input()
    return freq, np.asarray(peaks_max_ind)//FACTOR


def full_contact_signal_procedure(sig, p_window=0.52, d_window=0.32):
    pt,inte = contact_signal_procedure(sig, p_window)
    freq, maxes = hard_peaks(pt, d_window)
    return freq, maxes


def contact_signal_procedure(ch1, p_window=0.52):
    ch2 = interpolate(ch1,FACTOR)
    ch3 = butter_bandpass_filter(ch2, 0.2, 4, FACTOR*Fs, 3)
    ch4 = normalize(ch3)
    ch5 = pan_tompkins_algo(ch4, p_window)
    return ch5,ch4


def interpolate(signal,factor):
    x = np.arange(0,len(signal))
    INT = interp1d(x,signal)
    x = np.arange(0,len(signal)-1,1/factor)
    signal = INT(x)
    return np.asarray(signal)


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


def pan_tompkins_algo(signal_to_process, window):
    deriv = get_derivative(signal_to_process)
    sq_deriv = np.square(deriv)
    integr = get_integration(sq_deriv, int(window*FACTOR/Ts))
    integr = normalize(integr)
    return integr


def get_derivative(signal):
    kernel = np.asarray([-1, 0, 1])
    out_signal = convolve(signal, kernel)
    return out_signal[:-80]


def get_integration(signal, window_size):
    kernel = np.ones(window_size)
    out_signal = convolve(signal, kernel)
    return out_signal


def normalize(signal):
    signal = signal - np.mean(signal)
    s = signal / np.max(np.abs(signal))
    return s


def get_y_reverse_signal(sig):
    sig_max = np.max(sig)
    new_sig = sig_max-sig
    return new_sig


def reverse_signal(signal):
    signal = np.max(signal)-signal
    return signal


def from_new_x_to_old_x(peak_idx):
    to_old_idx = peak_idx / FACTOR
    return to_old_idx.astype(np.int) - 1

def prepare_indices(sg, ind):
    full_ind = []
    for i in range(len(sg)):
        if i in ind:
            full_ind.append(1)
        else:
            full_ind.append(0)
    return full_ind

def full_pan_tompkins(signal, cless_signal):
    res, signal_trans = pan_tompkins_algo(signal)
    cless_res, cless_signal = pan_tompkins_algo(cless_signal)

    indices = hard_peaks(res)
    maxes = res[indices]
    old_idx = from_new_x_to_old_x(indices)

    max_dots = signal_trans[indices]
    cless_dots = cless_signal[indices]
    full_indices = prepare_indices(cless_signal,indices)
    return signal_trans, res[:len(cless_signal)], full_indices
