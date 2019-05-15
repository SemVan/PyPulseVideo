import numpy as np
from scipy import signal as sp_sig
from scipy.signal import butter, lfilter, lfilter_zi, welch
import math
import csv
import time


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
