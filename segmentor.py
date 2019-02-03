import cv2
from image_processor import*

horizontal =  [5, 19, 28, 29, 30, 33, 50, 57]
vertical = [4, 5, 6, 7, 9, 10 ,11, 12]

def draw_lines(img, points):
    left = 0

    height, width = img.shape


    for hor in horizontal:
         y = points[hor]
         y = y[0, 1]
         cv2.line(img, (0, y), (width, y), color = (0, 0, 0))

    for ver in vertical:
         x = points[ver]
         x = x[0, 0]
         cv2.line(img, (x, 0), (x, height), color = (0, 0, 0))

    cv2.imshow('lines', img)
    cv2.waitKey()
    return


frame = cv2.imread("girl.jpg")

face_frame, rectangle = detect_face(frame)

im_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

true_contours = []
false_contours = []
dot_array = get_landmarks(frame, rectangle)

draw_lines(im_grey, dot_array)

# face = get_face_contour(dot_array)
# new_im = annotate_landmarks(im_grey, dot_array)
# cv2.imshow("g", new_im)
# cv2.waitKey()
