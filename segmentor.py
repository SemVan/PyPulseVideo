import cv2
from image_processor import*
import numpy as np
from matplotlib import pyplot as plt
import csv


float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

ver =  [6, 5, 57, 50, 33, 30, 29, 28]
hor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15]


def write_segmented_file(file_path,video_signal):
    dims = video_signal.shape
    packed_sig = pack_signal(video_signal)
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(dims)
        for i in range(packed_sig.shape(0)):
            for j in range(packed_sig.shape(1))
            writer.writerow(packed_sig[i][j])
    return

def read_segmented_file(file_name):
    packed_sig = []
    dims = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        dims = next(reader)
        for row in reader:
            packed_sig.append(row)

    return unpack_signal(packed_sig, dims)

def pack_signal(unpacked):
    old_shape = unpacked.shape
    new_shape = (old_shape[0], old_shape[1], old_shape[2]*old_shape[3] )
    packed = np.reshape(sig_t, new_shape)
    return packed

def unpack_signal(packed, dim):
    packed = np.asarray(packed)
    unpacked = np.reshape(packed, tuple(dim))
    return unpacked

def get_segmented_video(file_name):
    cap = cv2.VideoCapture(file_name)

    if not(cap.isOpened()):
        print("fuck opened")
        return 0, 0

    full_video_signals = []
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        one_vpg = get_segmented_frame(img)
        if one_vpg == None:
            return None, None
        full_video_signals.append(one_vpg)
    return np.asarray(full_video_signals)

def get_segmented_frame(img):
    face_frame, rectangle = detect_face(frame)
    if rectangle == None:
        return None
    im_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(frame, rectangle)

    channels = cv2.split(img)
    height, width = channels[0].shape
    one_frame_vpg = np.zeros(shape=(3, len(ver)-1, len(hor)-1))

    for i in range(len(hor)-1):
        hl_x = points[hor[i]][0,0]
        lr_x = points[hor[i+1]][0,0]

        for j in range(len(ver)-1):
            hl_y = points[ver[j+1]][0,1]
            lr_y = points[ver[j]][0,1]
            nimg = img.copy()
            # cv2.line(nimg, (hl_x, 0), (hl_x, height), color = (0, 0, 0))
            # cv2.line(nimg, (lr_x, 0), (lr_x, height), color = (0, 0, 0))
            # cv2.line(nimg, (0, hl_y), (width, hl_y), color = (0, 0, 0))
            # cv2.line(nimg, (0, lr_y), (width, lr_y), color = (0, 0, 0))
            # cv2.imshow("lines", nimg)
            # cv2.waitKey(0)

            submats = np.asarray([x[hl_y:lr_y, hl_x:lr_x] for x in channels])
            means = np.mean(submats, axis = (1,2))
            print(means)
            for k in range(len(channels)):
                one_frame_vpg[k][len(ver)-j-2][i] = np.mean(submats[k])
    return one_frame_vpg


frame = cv2.imread("girl.jpg")

vpg = get_segmented_signal(frame, dot_array)
print(vpg[1])
plt.imshow(vpg[0], cmap = 'gray')
plt.show()


# face = get_face_contour(dot_array)
# new_im = annotate_landmarks(im_grey, dot_array)
# cv2.imshow("g", new_im)
# cv2.imshow("vpg", vpg[0])
# cv2.waitKey()
