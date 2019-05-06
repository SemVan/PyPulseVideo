from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np

PERIOD = 1/100
WINDOW_SIZE = 1000

def full_fragment_amp_procedure(vpg):
    vpg_shape = vpg.shape
    vpg_hr = np.zeros(shape=vpg_shape[:-1])
    vpg_snr = np.zeros(shape=vpg_shape[:-1])
    vpg_flag = np.zeros(shape=vpg_shape[:-1])
    for i in range(vpg_shape[0]):
        for j in range (vpg_shape[1]): #changed shape element number
            vpg_hr, vpg_snr, vpg_flag = one_segment_procedure(vpg[i][j])

    return vpg_hr, vpg_snr, vpg_flag

def one_segment_procedure(segment_signal):
    # @todo classify signal here!!!
    spectrum, freqs = get_fourier_result(segment_signal, PERIOD)
    px, py = simple_peaks(spectrum, freqs, np.arange(1,2))
    fc, fc_amp = getSpectrumCentralFrequencyAndAmp(px, py)
    SNR = get_SNR(spectrum, fc_amp)

    # @todo classify signal here!!!
    signal_flag = 1
    return fc, SNR, signal_flag

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
