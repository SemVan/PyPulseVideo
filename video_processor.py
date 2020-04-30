import cv2
import platform
import numpy
from image_processor import *
import time


def full_video_file_procedure(file_name):
    cap = cv2.VideoCapture(file_name)

    if not(cap.isOpened()):
        print("fuck opened")
        return 0, 0

    start = dt.datetime.now()
    geometrical = []
    colorful = []
    colgeom = []
    while cap.isOpened():
        ret, img = cap.read()
        if ret == False:
            break
        g, c, cg = full_frame_procedure(img)
        if g == None:
            return None, None, None
        geometrical.append(g)
        colorful.append(c)
        colgeom.append(cg)
    
    print('processing time')
    print(dt.datetime.now()-start)
    return geometrical, colorful, colgeom
