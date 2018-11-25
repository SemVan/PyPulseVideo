import cv2
import platform
import numpy
from image_processor import *


def full_video_file_procedure(file_name):
    cap = cv2.VideoCapture(file_name)

    if not(cap.isOpened()):
        print("fuck opened")
        return 0, 0

    geometrical = []
    colorful = []
    while cap.isOpened():
        ret, img = cap.read()
        geometrical.append(geometrical_frame_procedure(img))
        colorful.append(colorful_frame_procedure(img))
    return geometrical, colorful
