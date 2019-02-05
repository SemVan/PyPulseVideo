import cv2
from image_processor import*
import numpy as np

ver =  [5, 57, 50, 33, 30, 29, 28, 19]
hor = [4, 5, 6, 7, 9, 10 ,11, 12]

def draw_lines(img, points):
    left = 0

    height, width = img.shape


    for h in hor:
         x = points[h]
         x = x[0, 0]
         cv2.line(img, (x, 0), (x, height), color = (0, 0, 0))

    for v in ver:
         y = points[v]
         y = y[0, 1]
         cv2.line(img, (0, y), (width, y), color = (0, 0, 0))

    cv2.imshow('lines', img)
    cv2.waitKey()
    return

def get_segmented_signal(img, points):
    channels = cv2.split(img)
    for i in range(len(hor)-1):
        hl_x = points[hor[i]][0,0]
        lr_x = points[hor[i+1]][0,0]
        for j in range(len(ver)-1):
            hl_y = points[ver[j+1]][0,1]
            lr_y = points[ver[j]][0,1]
            print(hl_x, hl_y, lr_x, lr_y)
            # submats = np.asarray([x[points[hor[i]]:points[hor[i+1]],points[ver[j]]:points[ver[j+1]] for x in channels])
            # print(submats.shape)
            # means = np.mean(submats)
    return


frame = cv2.imread("girl.jpg")

face_frame, rectangle = detect_face(frame)

im_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

true_contours = []
false_contours = []
dot_array = get_landmarks(frame, rectangle)

get_segmented_signal(frame, dot_array)
draw_lines(im_grey, dot_array)

# face = get_face_contour(dot_array)
new_im = annotate_landmarks(im_grey, dot_array)
cv2.imshow("g", new_im)
cv2.waitKey()
