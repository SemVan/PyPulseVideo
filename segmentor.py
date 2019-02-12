import cv2
from image_processor import*
import numpy as np
from matplotlib import pyplot as plt


float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

ver =  [6, 5, 57, 50, 33, 30, 29, 28]
hor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15]

def get_segmented_video(file_path):

    return

def get_segmented_signal(img, points):
    face_frame, rectangle = detect_face(frame)
    im_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = get_landmarks(frame, rectangle)

    channels = cv2.split(img)
    height, width = channels[0].shape
    one_frame_vpg = np.zeros(shape=(3, len(ver)-1, len(hor)-1))
    nimg = img.copy()
    for i in range(len(hor)-1):
        hl_x = points[hor[i]][0,0]
        lr_x = points[hor[i+1]][0,0]

        for j in range(len(ver)-1):
            hl_y = points[ver[j+1]][0,1]
            lr_y = points[ver[j]][0,1]

            cv2.line(nimg, (hl_x, 0), (hl_x, height), color = (0, 0, 0))
            cv2.line(nimg, (lr_x, 0), (lr_x, height), color = (0, 0, 0))
            cv2.line(nimg, (0, hl_y), (width, hl_y), color = (0, 0, 0))
            cv2.line(nimg, (0, lr_y), (width, lr_y), color = (0, 0, 0))
            cv2.imshow("lines", nimg)
            cv2.waitKey(0)
            nimg = img.copy()

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
draw_lines(im_grey, dot_array)

# face = get_face_contour(dot_array)
# new_im = annotate_landmarks(im_grey, dot_array)
# cv2.imshow("g", new_im)
# cv2.imshow("vpg", vpg[0])
# cv2.waitKey()
