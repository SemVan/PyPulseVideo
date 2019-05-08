import csv
import numpy as np



PHASE_NAME = "phase.csv"
HR_NAME = "hr.csv"
SNR_NAME = "snr.csv"
FLAG_NAME = "flag.csv"
NAMES = [PHASE_NAME, HR_NAME, SNR_NAME, FLAG_NAME]

def write_metrics(metrics, folder):
    num = len(metrics)
    for i in range(1):
        file_name = folder + "/" + NAMES[i]
        write_metric(metrics[i], file_name)
    return

def read_metrics(folder):
    metrics = []
    for i in range(len(NAMES)):
        file_name = folder + "/" + NAMES[i]
        metrics.append(read_metric(file_name))
    return metrics

def write_metric(metric, file_name):
    """Dims are fragment-row_column"""
    dims = metric.shape
    packed_metric = pack_metric(metric)
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(dims)
        for i in range(packed_metric.shape[0]):
            writer.writerow(packed_metric[i])
    return

def read_metric(file_name):
    packed_sig = []
    dims = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        dims = next(reader)
        for row in reader:
            packed_metric.append(row)
    dims = [int(x) for x in dims]

    return unpack_metric(packed_metric, dims)


def pack_metric(unpacked):
    old_shape = unpacked.shape
    new_shape = (old_shape[0], old_shape[1]*old_shape[2] )
    packed = np.reshape(unpacked, new_shape)
    return packed

def unpack_metric(packed, dim):
    packed = np.asarray(packed)
    unpacked = np.reshape(packed, tuple(dim))
    return unpacked
