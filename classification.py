import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sp_sig
from classification_math import *



def oneContactlessSignalPiece(signal, period):
    spectra, freqs = get_fourier_result(signal, period)
    spectra_n = normalizeSpectrum(spectra)
    pX, pY = simple_peaks(spectra_n, freqs, np.arange(1, 2))
    a, b, Fsec_x, Fsec = getSpectrumCentralFrequencyAndAmp(pX, pY)
    return a, Fsec_x, Fsec


def oneContactSignalPiece(signal, period):
    spectra, freqs = get_fourier_result(signal, period)
    spectra = normalizeSpectrum(spectra)
    pX, pY = simple_peaks(spectra, freqs, np.arange(1, 3))
    Fcentr, FcentrAmp, Fsec_x, Fsec = getSpectrumCentralFrequencyAndAmp(pX, pY)
    return Fcentr, Fsec_x, Fsec


def onePairProcedure(contact, contactless):
    per = 0.04
    contact_cut = normalizeSignal(contact)
    contactless_cut = normalizeSignal(contactless)
    contact_cut = butter_bandpass_filter(contact_cut, 0.5, 3, 25, 3)
    contactless_cut = butter_bandpass_filter(contactless_cut, 0.5, 3, 25, 3)

    centralFreq, Fsx_less, Fs_less = oneContactlessSignalPiece(contactless_cut, per)
    contactHr, Fsx_cont, Fs_cont = oneContactSignalPiece(contact_cut, per)

    true_equal = abs(contactHr - centralFreq)<=0.1 and abs(Fsx_cont - Fsx_less)<=0.1
    quasi_equal = abs(contactHr-Fsx_less)<=0.1 and abs(Fsx_cont- centralFreq)<=0.1

    target = int( (true_equal or quasi_equal) and Fsx_cont>0 and  Fsx_less>0)

    if target >= 0.9:
        target = 1
    else:
        target = -1

    return target
