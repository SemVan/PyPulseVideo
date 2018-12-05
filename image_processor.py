import opencv2
import dlib



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)



def geometrical_frame_procedure(frame):
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    true_contours = []
    false_contours = []
    dot_array = get_landmarks(img)

    face = get_face_contour(dot_array)
    eye_l, eye_r = get_eyes_contours(dot_array)
    mouth = get_lips_contour(dot_array)

    true_contours.append(face)
    false_contours.append(eye_l)
    false_contours.append(eye_r)
    false_contours.append(mouth)

    final_img = fill_black_out_contours(img, true_contours, false_contours)


    return get_sum_channels(final_img)


def colorful_frame_procedure(frame):
    face = detect_face(image)
    skin = detect_skin(face, image)
    return get_sum_channels(skin)


def get_sum_channels(img):
    channels = cv2.split(image)
    r = cv2.sumElems(channels[0])
    g = cv2.sumElems(channels[1])
    b = cv2.sumElems(channels[2])
    return r, g, b



def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    x,y,w,h = rects[0]
    print(x,y,w,h)
    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def get_face_contour(landmarks):
    cut_dot_array = landmarks[0:26]

    face_con = cut_dot_array[0:16]
    brow_con = cut_dot_array[17:26]

    brow_con = numpy.flip(brow_con, axis = 0)
    cut_dot_array = numpy.concatenate((face_con, brow_con), axis = 0)

    return cut_dot_array


def get_eyes_contours(landmarks):
    return landmarks[36:41], landmarks[42:47]


def get_lips_contour(landmarks):
    return landmarks[48:60]


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
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([255,173,127],np.uint8)

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
        return only_face
    return image
