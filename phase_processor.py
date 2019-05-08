from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.signal import correlate
from scipy.signal import butter, lfilter, lfilter_zi, welch, convolve2d
from scipy.stats import mannwhitneyu

DESCRETISATION_PERIOD = 0.040

def full_fragment_phase_mask(vpg):
    """For one video piece actually"""
    i_cent = int(vpg.shape[0]/2)
    j_cent = int(vpg.shape[1]/2)
    sig_ref = vpg[i_cent][j_cent]
    phase_mask = np.zeros((vpg.shape[0:2]))
    for row in range(vpg.shape[0]):
        for column in range(vpg.shape[1]):
            phase_mask[row][column] = get_phase_shift(sig_ref, vpg[row][column])
    return phase_mask

def get_phase_shift(sig1, sig2):
    cross_corr = correlate(sig2, sig1, 'full', 'direct')
    shift = np.argmax(cross_corr) - len(sig1)
    return shift*DESCRETISATION_PERIOD


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
