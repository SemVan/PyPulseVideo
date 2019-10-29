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
    phase = metrics[0]
    hr = metrics[1]
    snr = metrics[2]
    flag = metrics[3]
    
    frames, Y, X = flag.shape
    print(X)
    print(Y)
    print(frames)
    
    # Read source video
    cap = cv2.VideoCapture(video_name)

    if not(cap.isOpened()):
        print("video did not open (/Measured/..)")
        return []
    
    # Get frame size
    ret, img = cap.read()
    height,width,channels = img.shape
    print(width)
    print(height)
    print(channels)
    
    # get handler to VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
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
        mask = np.ones(img.shape)
        if len(x[0])>0:
            for i in range(X-1):
                for j in range(Y-1):
                    fl = (flag[frame_number-1, j, i]+1)/2
                    rect = np.array([[x[i][0,0],y[j][0,0]],[x[i][0,0],y[j+1][0,0]],[x[i+1][0,0],y[j+1][0,0]],[x[i+1][0,0],y[j][0,0]]])
                    print(rect)
                    cv2.fillPoly(mask, np.int32([rect]), (fl,fl,fl))
        #cv2.imshow("",mask)
        #cv2.waitKey(0)
        img = img*np.uint8(mask)
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

    x = points[hor,0]
    y = points[ver,1]
    #print(x)
    #print(y)
    
    return x, y
