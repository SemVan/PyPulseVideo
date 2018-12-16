import cv2
import dlib
import numpy
import datetime as dt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)



def full_frame_procedure(frame):
    start = dt.datetime.now()
    face_frame, rectangle = detect_face(frame)

    if rectangle == None:
        print("returning None")
        return None, None
    face_stop = dt.datetime.now()
    face_el = face_stop-start
    print("face detection " + str(face_el.microseconds/1000))
    geom = geometrical_frame_procedure(frame, rectangle)
    geom_stop = dt.datetime.now()
    geom_el = geom_stop-face_stop
    print("geometrical processing " + str(geom_el.microseconds/1000))

    color = colorful_frame_procedure(face_frame, frame)
    color_stop = dt.datetime.now()
    color_el = color_stop-geom_stop
    full_el = color_stop-start
    print("color processing " + str(color_el.microseconds/1000))
    print("full processing " + str(full_el.microseconds/1000))
    print()
    return geom, color

def geometrical_frame_procedure(frame, rect):
    im_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    true_contours = []
    false_contours = []
    dot_array = get_landmarks(frame, rect)

    face = get_face_contour(dot_array)
    eye_l = dot_array[36:41]
    eye_r = dot_array[42:47]
    mouth = dot_array[48:60]

    true_contours.append(face)
    false_contours.append(eye_l)
    false_contours.append(eye_r)
    false_contours.append(mouth)

    final_img = fill_black_out_contours(frame, true_contours, false_contours)
    return get_sum_channels(final_img)


def colorful_frame_procedure(face, frame):
    skin = detect_skin(face, frame)
    # cv2.imshow("color", skin)
    # cv2.waitKey(10)
    return get_sum_channels(skin)


def get_sum_channels(frame):
    channels = cv2.split(frame)
    r = cv2.sumElems(channels[0])
    g = cv2.sumElems(channels[1])
    b = cv2.sumElems(channels[2])

    r = r[0]/cv2.countNonZero(channels[0])
    g = g[0]/cv2.countNonZero(channels[1])
    b = b[0]/cv2.countNonZero(channels[2])
    return r, g, b



def get_landmarks(im, rect):
    x,y,w,h = rect
    # print(x,y,w,h)
    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def get_face_contour(landmarks):
    cut_dot_array = landmarks[0:26]

    face_con = cut_dot_array[0:16]
    brow_con = cut_dot_array[17:26]

    brow_con = numpy.flip(brow_con, axis = 0)
    cut_dot_array = numpy.concatenate((face_con, brow_con), axis = 0)

    return cut_dot_array


def fill_black_out_contours(img, true_conts, false_conts):
    true_mask = numpy.zeros(img.shape).astype(img.dtype)
    false_mask = numpy.full(img.shape, 255, img.dtype)

    true_color = [255, 255, 255]
    false_color = [0, 0, 0]

    cv2.fillPoly(true_mask, true_conts, true_color)
    cv2.fillPoly(false_mask, false_conts, false_color)

    result = cv2.bitwise_and(img, true_mask)
    result = cv2.bitwise_and(result, false_mask)
    return result



def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def detect_skin(face, background):
    min_YCrCb = numpy.array([0,133,77],numpy.uint8)
    max_YCrCb = numpy.array([255,173,127],numpy.uint8)

    imageYCrCb = cv2.cvtColor(face,cv2.COLOR_BGR2YCR_CB)
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skin = cv2.bitwise_and(face, face, mask=skinRegion)
    image, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 2000:
            masked_img = face.copy()
            cv2.fillPoly(face, contours, [0, 0, 0])
            face = masked_img - face
    return face


def detect_face(image):
    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), maxSize=(250, 250))
    if len(faces) > 0:
        strt_x = faces[0][0]
        strt_y = faces[0][1]
        width = faces[0][2]
        height = faces[0][3]
        only_face = image[strt_y:strt_y+height, strt_x:strt_x+width]
        return only_face, faces[0]
    return image, None
