import cv2
from metrics_io import*
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


def make_edited_video(video_name, video_edited_name, metric_path):
    
    # Correctly open .csv files: for this I need an opener in metrics
    metrics = read_metrics(metric_path)
    phase = metrics(0)
    hr = metrics(1)
    snr = metrics(2)
    flag = metrics(3)
    
    frames, X, Y = flag.shape
    
    # Read source video
    cap = cv2.VideoCapture(video_name)

    if not(cap.isOpened()):
        print("video did not open (/Measured/..)")
        return []
    
    # Get frame size
    ret, img = cap.read()
    width,height,channels = img.shape
    
    # get handler to VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_edited_name, fourcc, 30, (width,height))
    
    frame_number = 0
    
    while (cap.isOpened() and frame_number<frames):
        st = dt.datetime.now()
        ret, img = cap.read()
        if ret == False:
            break
        frame_number += 1
        x, y = get_segmentation_grid(img)
        if x == []:
            x = []
            y = []
        # Теперь модифицируем кадр img, закрашивая на нём отдельные области
        if len(x)>0:
            for i in range(len(x)-1):
                for j in range(len(y)-1):
                    img[x[i]:x[i+1],y[j]:y[j+1],:] *= flag(frame_number, i, j)
        
        out.write(img)
        frametime = dt.datetime.now() - st
        print(frametime.microseconds/1000)
    out.release()
    cap.release()
    return


def get_segmentation_grid(img):
    face_frame, rectangle = detect_face(img)
    
    if len(rectangle) == 0:
        return []
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(img, rectangle)

    x = points[hor][0,0]
    y = points[ver][0,1]
    
    return x, y
