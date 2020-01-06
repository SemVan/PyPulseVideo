import cv2
from metrics_io import*
from image_processor import *
import numpy as np
from matplotlib import pyplot as plt
from segmented_io import *
import datetime as dt

import time


float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

colormap = plt.get_cmap('rainbow')

ver =  [6, 5, 57, 50, 33, 30, 29, 28]
hor = [3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13]


def make_edited_video(video_name, video_edited_name, metric_path):

    # Correctly open .csv files: for this I need an opener in metrics
    metrics = read_metrics(metric_path)
    phase = metrics[0]
    hr = metrics[1]
    snr = metrics[2]
    flag = metrics[3]
    
    maxSNR = np.max(snr)
    minSNR = np.min(snr)
    SNR = maxSNR - minSNR
    
    print('Предельные значения SNR')
    print(maxSNR)
    print(minSNR)
    
    flag = filter_flag(flag)

    frames, Y, X = flag.shape
    print('Оси сегментирующей сетки')
    print(X)
    print(Y)
    print('Число кадров')
    print(frames)

    # Read source video
    cap = cv2.VideoCapture(video_name)

    if not(cap.isOpened()):
        print("video did not open (/Measurements/..)")
        return []

    # Get frame size
    ret, img = cap.read()
    height,width,channels = img.shape
    print('Размеры кадра')
    print(width)
    print(height)
    print('Число каналов')
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
        x, y, face_dots = get_segmentation_grid(img)
        img = highlight_face(img, face_dots)
        if x == []:
            x = []
            y = []
        # Теперь модифицируем кадр img, закрашивая на нём отдельные области
        mask = np.ones(img.shape)
        
        # draw segmentation grid
        if len(x[0])>0:
            for i in range(X):
                for j in range(Y):
                    cv2.rectangle(img,(x[i][0,0],y[j][0,0]),(x[i+1][0,0],y[j+1][0,0]),(0,0,0))
                    
        # highlight good areas
        if len(x[0])>0:
            for i in range(X):
                for j in range(Y):
                    fl = (flag[frame_number-1, Y-1-j, i]+1)/2
                    sn = snr[frame_number-1, Y-1-j, i]
                    sn = (maxSNR-sn)/SNR
                    if fl>0:
                        col = colormap(sn)
                        color = np.array(col)*255
                        r = int(color[2])
                        g = int(color[1])
                        b = int(color[0])
                        x1 = int(x[i][0,0])
                        y1 = int(y[j][0,0])
                        start = (x1,y1)
                        x2 = int(x[i+1][0,0])
                        y2 = int(y[j+1][0,0])
                        final = (x2,y2)
                        color = (b,g,r)
                        cv2.rectangle(img,start,final,color)
        # img = img*np.uint8(mask)
        # opacity = 0.5
        # cv2.addWeighted(np.uint8(mask), opacity, img, 1 - opacity, 0, img)
        # cv2.imshow("",img)
        # cv2.waitKey(5)
        
        
        out.write(img)
        frametime = dt.datetime.now() - st
        # print(frametime.microseconds/1000)
    out.release()
    cap.release()
    return


def get_segmentation_grid(img):
    face_frame, rectangle = detect_face(img)

    if len(rectangle) == 0:
        return []
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(img, rectangle)
    face = get_face_contour(points)

    x = points[hor,0]
    y = points[ver,1]
    #print(x)
    #print(y)

    return x, y, face

def highlight_face(img, face):
    false_mask = np.full(img.shape, 0, img.dtype)
    true_color = [255, 255, 255]
    cv2.fillPoly(false_mask, [face], true_color)
    opacity = 0.2
    cv2.addWeighted(false_mask, opacity, img, 1 - opacity, 0, img)
    return img


def filter_flag(flag):
    frames, Y, X = flag.shape
    counter = np.uint8(0)
    for x in range(X):
        for y in range(Y):
            prev = -1
            for i in range(frames-1):
                fl = flag[i,y,x]
                if (fl-prev)>0 or counter>0:
                    counter += np.uint8(1)
                    flag[i,y,x] = 1
                prev = flag[i,y,x]
    return flag