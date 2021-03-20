import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sp_sig
from scipy.stats import moment
# import pywt
from scipy.signal import butter, lfilter, lfilter_zi, welch
from scipy.signal import find_peaks_cwt
from scipy.signal import correlate
from scipy.signal import cwt
from scipy.interpolate import CubicSpline
import math
import csv
import time

from sklearn import linear_model
import pandas as pd
import numpy as np
from joblib import dump, load

# from math_utils import *
model = load('regression_sag_new.joblib')


def readFile(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(",")
            data[0].append(rowList[0])
            if len(rowList)==2:
                data[1].append(float(rowList[1]))
            else:
                if (float(rowList[1])+float(rowList[3])>0):
                    data[1].append(float(rowList[2])/(float(rowList[1])+float(rowList[3])))
                else:
                    data[1].append(float(rowList[2]))
    return (get_y_reverse_signal(np.asarray(data[1])))


def read_new_file(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(" ")
            data[0].append(rowList[0])
            if len(rowList)==2:
                data[1].append(float(rowList[1]))
            else:
                if (float(rowList[0])+float(rowList[2])>0):
                    data[1].append(float(rowList[1])/(float(rowList[0])+float(rowList[2])))
                else:
                    data[1].append(float(rowList[1]))
    return (get_y_reverse_signal(np.asarray(data[1])))



def get_y_reverse_signal(sig):
    sig_max = np.max(sig)
    new_sig = sig_max-sig
    return new_sig

def get_fourier_result (signal, period):
    complex_four = np.fft.fft(signal)
    spectra = np.absolute(complex_four)
    freqs = []
    for i in range(len(signal)):
        freqs.append(1/(period*len(signal))*i)
    return spectra, freqs

def get_wavelet(s):
    fs = 25
    period = 0.04
    widths = np.arange(1, 40, 1)
    cwtmatr, freqs = pywt.cwt(s, widths, 'mexh', period)
    cwtmatr = cwtmatr[:20,:]
    freqs = freqs[:20]
    return cwtmatr

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


def normalizeSpectrum(spectra):
    nSp = spectra/spectra[0]
    return nSp


def normalizeSignal(signal):
    m = max(signal)
    new_sig = signal/m
    return new_sig


def findNearestElemInTheList(lst, elem):
    diffs = [abs(lst_el-elem) for lst_el in lst]
    return np.argmin(diffs)


def getSignalIntegral(s_y, s_x, start_x, stop_x):
    str_idx = findNearestElemInTheList(s_x, start_x)
    stp_idx = findNearestElemInTheList(s_x, stop_x)
    return np.sum(s_y[:int(len(s_y)/2)])


def getDixonCriteria(amps, central):
    if (len(amps)>2):
        amps = sorted(amps)
        cen_ndx = findNearestElemInTheList(amps, central)
        dix = -1
        if cen_ndx != 0:
            dix = (amps[cen_ndx] - amps[cen_ndx-1])/(amps[cen_ndx] - amps[0])
        return dix
    else:
        return -1


def get_offset(y1, y2):
    max = 0
    best_offset = 0
    for i in range(1,50):
        scal = np.dot(y1,y2)
        if scal>max:
            max = scal
            best_offset = i-1
        y2 = y2[i:]
        y1 = y1[:-i]
    return best_offset


def move_best(y1, y2):
    offset = get_offset(y1, y2)
    print('offset=', offset)
    new_y1 = []
    new_y2 = []
    if offset>=0:
        new_y1 = y1[offset:]
        new_y2 = y2[:offset]
    else:
        new_y2 = y2[offset:]
        new_y1 = y1[:offset]
    plt.plot(range(len(new_y1)), new_y1)
    plt.plot(range(len(new_y2)), new_y2)
    plt.show()
    return new_y1, new_y2


def interpolation(signal, x):
    step = 0.1
    new_x = np.arange(x[0], x[len(x)-1], step)

    new_y = CubicSpline(x, signal)
    return new_x, new_y(new_x)


def peaks(signal, xAx, dotSize, num):
    peaks = find_peaks_cwt(signal, dotSize)
    y = np.asarray([signal[i] for i in peaks])
    # y = [num for i in peaks]
    x = np.asarray([xAx[i] for i in peaks])
    return x, y


def delete_peak_doubling(peakX, peakY):
    truePeaksX = []
    truePeaksY = []
    prev = peakX[0]
    aver = prev
    medi = [prev]
    averY = [peakY[0]]
    count = 1
    for i in range(1, len(peakX)):
        if peakX[i]-prev<5:
            aver +=peakX[i]
            medi.append(peakX)
            averY.append(peakY[i])
            count += 1
        else:
            truePeaksX.append(aver/count)
            truePeaksY.append(max(averY))
            prev = peakX[i]
            aver = prev
            median = [prev]
            averY = [peakY[i]]
            count = 1

    return truePeaksX, truePeaksY


def compare_by_peaks(y1, x1, y2, x2):
    x1 = range(len(y1))
    x2 = range(len(y2))

    x1, y1 = interpolation(y1, x1)
    x2, y2 = interpolation(y2, x2)


    x1_p, y1_p = peaks(y1, x1, [2,3,4,5], 1)
    x1_p, y1_p = delete_peak_doubling(x1_p, y1_p)
    x2_p, y2_p = peaks(y2, x2, [2,3,4,5], 1)
    x2_p, y2_p = delete_peak_doubling(x2_p, y2_p)

    plt.scatter(x1_p, y1_p, color='red')
    plt.plot(x1, y1)
    plt.scatter(x2_p, y2_p, color='blue')
    plt.plot(x2, y2)
    plt.show()
    if len(x1_p)>2 and len(x2_p)>2:
        if len(x1_p) == len(x2_p):
            d_max = 0
            p1 = x1_p[1:-1]
            p2 = x2_p[1:-1]
            for i in range(len(p1)):
                d = abs(p1[i]-p2[i])
                if d>d_max:
                    d_max = d
            if d_max<= 10:
                return [True, len(x1_p), len(x2_p), x1_p, x2_p, y1_p, y2_p, y1, y2]
    return [False, len(x1_p), len(x2_p), x1_p, x2_p, y1_p, y2_p, y1, y2]


def get_signal_peaks_metrics (pX):
    hr = []
    for i in range(len(pX)-1):
        hr.append(pX[i+1]-pX[i])
    mean = np.mean(hr)
    sd = np.std(hr)
    maxi = np.max(hr)
    mini = np.min(hr)
    return mean, sd, maxi, mini


def simple_peaks(signal, xAx, dotSize):
    x = []
    y = []
    for i in range(1, len(signal)-1):
        if (signal[i]>signal[i+1] and signal[i]>signal[i-1]):
            x.append(xAx[i])
            y.append(signal[i])

    z = list(zip(y, x))
    y = [z[i][0] for i in range(len(z)) if (z[i][1] > 0.2 and z[i][1] < 4.0)]
    x = [z[i][1] for i in range(len(z)) if (z[i][1] > 0.2 and z[i][1] < 4.0)]
    return x,y


def getPeakCount(peaks_x, peaks_y):
    peaks_cut_x = [peaks_x[i] for i in range(len(peaks_x)) if (peaks_x[i] > 0.2 and peaks_x[i] < 3.0)]
    return len(peaks_cut_x)


def getMeanValAndSD(sig):
    if (len(sig)>0):
        return np.mean(sig), np.std(sig)
    else:
        return 0, -1


def getSpectrumCentralFrequencyAndAmp(peakX, peakY):
    if len(peakX) == 0:
        return 0, 0, 0, 0
    result = list(reversed(sorted(zip(peakY, peakX))))
    maxY = result[0][0]
    maxX = result[0][1]
    fsec = 0
    fsec_x = 0
    if (len(result)>1):
        fsec = result[1][0]
        fsec_x = result[1][1]
    return maxX, maxY, fsec_x, fsec


def getPearsonCorrelation(signal1, signal2):
    return np.corrcoef(signal1,signal2)[0][1]


def getCorrelationFuncMax(signal1, signal2):
    minlen = min(len(signal1), len(signal2))
    corrF = correlate(signal1[0:minlen], signal2[0:minlen], 'full', 'direct')
    return max(corrF)


def getDensityMoments(signal):
    moments = []
    for i in range(1,5):
        mom = moment(signal, moment = i)
        mom = np.nan_to_num(mom)
        moments.append(mom)
    return moments


def oneContactlessSignalPiece(signal, period):
    d = {}

    spectra, freqs = get_fourier_result(signal, period)
    spectra_n = normalizeSpectrum(spectra)
    pX, pY = simple_peaks(spectra_n, freqs, np.arange(1, 2))
    a, b, Fsec_x, Fsec = getSpectrumCentralFrequencyAndAmp(pX, pY)
    dixi = getDixonCriteria(pY, b)

    moms = getDensityMoments(spectra_n)
    d['first_moment'] = moms[0]
    d['second_moment'] = moms[1]
    d['third_moment'] = moms[2]
    d['forth_moment'] = moms[3]

    if (Fsec == 0):
        d['relation'] = 0
    else:
        d['relation'] = b/Fsec

    d['central_freq'], d['central_freq_amp'] = a, b
    d['dixi'] = dixi
    d['peak_count'] = getPeakCount(pX, pY)
    d['peak_mean'], d['peak_SD'] = getMeanValAndSD(pY)
    d['sig_mean'], d['sig_SD'] = getMeanValAndSD(signal)
    if math.isnan(d['sig_mean']):
        d['sig_mean'] = 0
    if math.isnan(d['sig_SD']):
        d['sig_SD'] = 0

    d['peak_matters_SD'] = abs(d['central_freq_amp']-d['peak_mean'])/d['peak_SD']
    if math.isnan(d['peak_matters_SD']):
        d['peak_matters_SD'] = -1
    integ = getSignalIntegral(spectra_n, freqs, 0, 3)

    if integ>0.000001:
        d['peak_matters_Int'] = d['central_freq_amp'] / integ
    else:
        d['peak_matters_Int'] = 10000000

    return d, spectra_n, pX, pY, freqs, Fsec_x, Fsec


def oneContactSignalPiece(signal, period):
    d = {}
    spectra, freqs = get_fourier_result(signal, period)
    spectra = normalizeSpectrum(spectra)
    pX, pY = simple_peaks(spectra, freqs, np.arange(1, 3))
    Fcentr, FcentrAmp, Fsec_x, Fsec = getSpectrumCentralFrequencyAndAmp(pX, pY)
    return Fcentr, FcentrAmp, spectra, freqs, pX, pY, freqs, Fsec_x, Fsec


def get_signal_fragment_params(contactless, window=128):
    per = 0.04
    fullD = []
    targets = 0
    total = 0
    mean_correlation = 0
    small_window = 128
    printed = False

    contactless_cut = normalizeSignal(contactless)
    contactless_cut = butter_bandpass_filter(contactless_cut, 0.5, 3, 25, 3)

    dictLess, less_sp, p_x_l, p_y_l, fr_l, Fsx_less, Fs_less = oneContactlessSignalPiece(contactless_cut, per)
    if (dictLess == 0):
        return []
    if (Fsx_less == 0):
        dictLess['x_relation'] = 0
    else:
        dictLess['x_relation'] = dictLess['central_freq'] / Fsx_less
    # if printed and dictLess['x_relation'] > 0.6:
    #     plt.plot(fr_l, less_sp)
    #     plt.scatter(dictLess['central_freq'], dictLess['central_freq_amp'], color='red')
    #     plt.scatter(Fsx_less, Fs_less, color='red')
    #     plt.xlim([0,4])
    #     plt.plot(fr, contact_sp)
    #     plt.scatter(contactHr, hr_fr, color='black')
    #     plt.scatter(Fsx_cont, Fs_cont, color='black')
    #     plt.show()
    return dictLess


def check_if_fragment_is_reliable(signal):
    features = ['central_freq', 'peak_matters_Int', 'x_relation', 'sig_SD']
    feature_dict = get_signal_fragment_params(signal)
    fit_result = 0
    if len(list(feature_dict.keys())) > 0:
        feature_vector = [[feature_dict[feature] for feature in features]]
        print("FITTING")
        print(feature_vector)
        fit_result = model.predict(feature_vector)
    print("FIT RESULT ", fit_result)
    return fit_result
