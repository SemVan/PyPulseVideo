import cv2
import platform
import numpy
from image_processor import *
import time


def full_video_file_procedure(file_name):
    cap = cv2.VideoCapture(file_name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not(cap.isOpened()):
        print("fuck opened")
        return 0, 0

    geometrical = []
    colorful = []
    colgeom = []
    num = 1
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        g, c, cg = full_frame_procedure(img)
        print('frame {} out of {}'.format(num, total), end='\r')
        num += 1
        if g == None:
            return geometrical, colorful, colgeom
        geometrical.append(g)
        colorful.append(c)
        colgeom.append(cg)
    return geometrical, colorful, colgeom
