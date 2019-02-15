from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np



def full_frame_amp_procedure(vpg):

    return

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

def get_SNR(signal):
    SNR = 0
    return SNR
