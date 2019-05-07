import cv2
from image_processor import*
import numpy as np
from matplotlib import pyplot as plt
from segmented_io import *
import datetime as dt

import time


float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

ver =  [6, 5, 57, 50, 33, 30, 29, 28]
hor = [3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13]


def get_segmented_video(file_name):
    cap = cv2.VideoCapture(file_name)

    if not(cap.isOpened()):
        print("fuck opened")
        return []

    full_video_signals = []
    while cap.isOpened():
        st = dt.datetime.now()
        ret, img = cap.read()
        if ret == False:
            break
        one_vpg = get_segmented_frame(img)
        if one_vpg == []:
            return []
        full_video_signals.append(one_vpg)
        frametime = dt.datetime.now() - st
        print(frametime.microseconds/1000)
    return np.asarray(full_video_signals)

def get_segmented_frame(img):
    face_frame, rectangle = detect_face(img)
    #if rectangle == None:
    if len(rectangle) == 0:
        return []
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(img, rectangle)

    channels = cv2.split(img)
    height, width = channels[0].shape
    one_frame_vpg = np.zeros(shape=(3, len(ver)-1, len(hor)-1))

    for i in range(len(hor)-1):
        hl_x = points[hor[i]][0,0]
        lr_x = points[hor[i+1]][0,0]
        # if hl_x>lr_x:
        #     lr_x, hl_x = hl_x, lr_x
        # if (hl_x == lr_x):
        #     print("pizda x")


        for j in range(len(ver)-1):
            hl_y = points[ver[j+1]][0,1]
            lr_y = points[ver[j]][0,1]
            # if hl_y>lr_y:
            #     lr_y, hl_y = hl_y, lr_y
            # if (hl_y == lr_y):
            #     print("pizda y")

            # nimg = img.copy()
            # cv2.line(nimg, (hl_x, 0), (hl_x, height), color = (0, 0, 0))
            # cv2.line(nimg, (lr_x, 0), (lr_x, height), color = (0, 0, 0))
            # cv2.line(nimg, (0, hl_y), (width, hl_y), color = (0, 0, 0))
            # cv2.line(nimg, (0, lr_y), (width, lr_y), color = (0, 0, 0))
            # cv2.imshow("lines", nimg)
            # cv2.waitKey(0)

            submats = np.asarray([x[hl_y:lr_y, hl_x:lr_x] for x in channels])

            # means = np.mean(submats, axis = (1,2))
            # print("submts together", means)
            for k in range(len(channels)):
                m = np.mean(submats[k])
                if np.isnan(m):
                    m = 0
                one_frame_vpg[k][len(ver)-j-2][i] = np.mean(submats[k])

    return one_frame_vpg


# frame = cv2.imread("girl.jpg")
# sig = []
# vpg = get_segmented_frame(frame)
# sig.append(vpg)
# sig = np.asarray(sig)
# file_path = "./Segmented/fuck.csv"
# write_segmented_file(file_path, sig)
# sig_r = read_segmented_file(file_path)
# print(np.isclose(sig, sig_r, atol=0.001))
