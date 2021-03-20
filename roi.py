import cv2
from image_processor import*
import numpy as np
from matplotlib import pyplot as plt
from segmented_io import *
import datetime as dt

ver =  [50, 33, 30, 29, 28]
hor = [3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13]


def get_segmented_frame(img):
    face_frame, rectangle = detect_face(img)
    if len(rectangle) == 0:
        return []
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(img, rectangle)

    channels = cv2.split(img)
    height, width = channels[0].shape
    one_frame_vpg = np.zeros(shape=(3, len(ver)-1, len(hor)-1))
    mask = np.zeros(img.shape).astype(img.dtype)
    for i in range(len(hor)-1):
        hl_x = points[hor[i]][0,0]
        lr_x = points[hor[i+1]][0,0]

        for j in range(len(ver)-1):
            hl_y = points[ver[j+1]][0,1]
            lr_y = points[ver[j]][0,1]
            print(ver[j])
            if ver[j+1] == 24:
                lr_y = points[ver[j+1]][0,1] + (points[ver[j+1]][0,1] - points[36][0,1])
                hl_y = points[ver[j+1]][0,1]
            if hl_y>lr_y:
                lr_y, hl_y = hl_y, lr_y
            if (hl_y == lr_y):
                print("pizda y")

            nimg = img.copy()
            if (hor[i] != 8 and hor[i] != 7 or ver[j] == 50) and not (ver[j] == 50 and hor[i] in [3, 12]):
                mask[hl_y:lr_y, hl_x:lr_x, 0] = 255
                mask[hl_y:lr_y, hl_x:lr_x, 1] = 255
                mask[hl_y:lr_y, hl_x:lr_x, 2] = 255

    # result = cv2.bitwise_and(img, mask)
    # cv2.imshow("lines", result)
    # cv2.waitKey(0)
    return result
