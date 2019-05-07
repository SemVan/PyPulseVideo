import numpy as np
import csv


def write_segmented_file(file_path,video_signal):
    dims = video_signal.shape
    packed_sig = pack_signal(video_signal)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(dims)
        for i in range(packed_sig.shape[0]):
            for j in range(packed_sig.shape[1]):
                writer.writerow(packed_sig[i][j])
    return

def read_segmented_file(file_name):
    packed_sig = []
    dims = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        dims = next(reader)
        for row in reader:
            packed_sig.append(row)
    dims = [int(x) for x in dims]
    return unpack_signal(packed_sig, dims)

def pack_signal(unpacked):
    """Shape was frame-channel-row-column. Shape became frame-channel-row*column"""
    old_shape = unpacked.shape
    new_shape = (old_shape[0], old_shape[1], old_shape[2]*old_shape[3] )
    packed = np.reshape(unpacked, new_shape)
    return packed

def unpack_signal(packed, dim):
    packed = np.asarray(packed)
    unpacked = np.reshape(packed, tuple(dim))
    return unpacked
