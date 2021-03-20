from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from classification import *
from peakdetect import peakdetect
from pan_tompkins import *

PERIOD = 1/100
WINDOW_SIZE = 1000

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

def one_segment_procedure(segment_signal, contact_signal):
    spectrum, freqs = get_fourier_result(segment_signal, PERIOD)
    px, py = simple_peaks(spectrum, freqs, np.arange(1,2))
    fc, fc_amp = getSpectrumCentralFrequencyAndAmp(px, py)
    SNR = get_SNR(spectrum, fc_amp)
    print(np.asarray(contact_signal).shape)
    signal_flag = onePairProcedure(contact_signal, segment_signal)
    cont_pan_tom, fuck1 = pan_tompkins_algo(contact_signal)
    # plt.plot(range(len(cont_pan_tom)), cont_pan_tom)
    # plt.plot(range(len(fuck)), fuck)
    # plt.show()
    less_pan_tom, fuck =  pan_tompkins_algo(segment_signal)

    freq1 = hard_peaks(cont_pan_tom)
    freq2 = hard_peaks(less_pan_tom)
    signal_flag = -1
    print("CONTACT ", freq1)
    if freq1 == -100:
        freq1 = -10000
    print("LESS ", freq2)
    # input()
    if abs(freq1*60-freq2*60) <=3:
        # plt.plot(range(len(cont_pan_tom)), cont_pan_tom)
        # plt.plot(range(len(less_pan_tom)), less_pan_tom)
        # plt.show()
        signal_flag = 1
    return freq1*60, SNR, signal_flag#, freq1, freq2

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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y

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

def hard_peaks(signal):
    signal = np.asarray(signal) / np.max(signal)
    peaks2 = peakdetect(signal,lookahead=8)
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
    for i in range(len(peaks_min_ind)-1):
        peaks_diff.append(peaks_min_ind[i+1]-peaks_min_ind[i])
    print(peaks_diff)
    print(np.mean(peaks_diff))
    freq = -100
    if len(peaks_diff) > 0:
        freq = 1/(np.mean(peaks_diff)*0.04)
    #
    # x1 = np.arange(0, len(signal))
    # y1 = np.array(signal)
    # plt.plot(x1, y1)
    # # plt.plot(peaks_max_ind, peaks_max, 'x')
    # plt.plot(peaks_min_ind, peaks_min, 'x')
    # plt.show()
    # input()
    return freq
