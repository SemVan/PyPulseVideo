import imutils
from imutils import face_utils
import cv2
import dlib
import numpy as np
import datetime as dt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def full_frame_procedure(frame):
    start = dt.datetime.now()
    face_frame, rectangle = detect_face(frame)

    if rectangle is None:
        print("returning None")
        return None, None, None
    face_stop = dt.datetime.now()
    face_el = face_stop-start
    # print("face detection " + str(face_el.microseconds/1000))
    geom = geometrical_frame_procedure(frame, rectangle)
    geom_stop = dt.datetime.now()
    geom_el = geom_stop-face_stop
    # print("geometrical processing " + str(geom_el.microseconds/1000))

    color=[]
    color = colorful_frame_procedure(face_frame, frame)
    color_stop = dt.datetime.now()
    color_el = color_stop-geom_stop
    # print("color processing " + str(color_el.microseconds/1000))

    colgeom = geometrical_and_color(frame,rectangle)
    geom_color_stop = dt.datetime.now()
    geom_color_el = geom_color_stop-color_stop
    # print("geometrical & color processing " + str(geom_color_el.microseconds/1000))
    full_el = geom_color_stop-start
    # print("full processing " + str(full_el.microseconds/1000))
    # print()
    return geom, color, colgeom

def geometrical_and_color(frame, rectangle):
    image = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # create the ground color
    ground = np.zeros(image.shape)
    ground[:,0::4,1] = 180

    #get points
    x,y,w,h = rectangle
    rect = dlib.rectangle(x,y,x+w,y+h)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    shape = np.int32(shape)

    #get face contours and create mask
    lEye = shape[36:42,:]
    rEye = shape[42:48,:]
    mouth = shape[48:60,:]
    edge = np.concatenate((shape[0:17,:],shape[26:16:-1,:]))
    upperSide = np.concatenate(([[x,y]],shape[17:27,:],[[x+w,y]]))
    lowerSide = np.concatenate(([[x,y]],[shape[17,:]],shape[0:17,:],[shape[26,:]],[[x+w,y]]))

    mask=np.zeros(frame.shape)
    cv2.fillPoly(mask,(edge,mouth,lEye,rEye),(1,1,1))

    # get face color distribution
    indices = np.where(mask>0)
    BGR = image[indices]

    # linear approximation
    vec,point = get_baseline(BGR)

    # get mean squared distance to the axis
    mean = get_meanDistances2(vec,point,BGR)
    mean = mean ** 0.5
    mean = 10

    # calculate distances to the axis
    distances = get_distance(vec,point, image)
    #cv2.imshow("Distances",1-distances/np.max(distances))
    #cv2.waitKey(0)

    # highlight area of interest
    maskDist = np.zeros(distances.shape)
    cv2.fillPoly(maskDist,[upperSide],(1,1,1))
    distances = distances*maskDist
    maskIm = np.zeros(image.shape)
    cv2.fillPoly(maskIm,(lowerSide, mouth, lEye, rEye),(1,1,1))

    # highlight remote pixels
    indicesHighlight=np.where(
    distances>mean)
    maskIm[indicesHighlight]=0
    maskGr=np.ones(image.shape)-maskIm
    imNew=np.uint8(image*maskIm)#+ground*maskGr)

    # calculate mean intensity
    # indicesFace=np.where(maskIm>0)
    # intensities=imNew[indicesFace]
    # intB=intensities[0::3]
    # intG=intensities[1::3]
    # intR=intensities[2::3]
    # meanB=np.mean(intB)
    # meanG=np.mean(intG)
    # meanR=np.mean(intR)
    # meanIntensity=meanG/(meanR+meanB)

    # show image
    # cv2.imshow("geometrical and color output", imNew)
    # cv2.waitKey(10)
    return get_sum_channels(imNew)

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
    # cv2.imshow("hgkjg", final_img)
    # cv2.waitKey(0)
    return get_sum_channels(final_img)


def find_skin_regions(img, face, rd, gr, bl):
    strt_x = face[0]
    strt_y = face[1]
    width = face[2]
    height = face[3]
    only_face = img[strt_y:strt_y+height, strt_x:strt_x+width]
    lower = np.array([rd-0.2*rd, gr-0.2*gr, bl-0.2*bl], dtype = only_face.dtype)
    upper = np.array([rd+0.2*rd, gr+0.2*gr, bl+0.2*bl], dtype = only_face.dtype)
    skin_mask = cv2.inRange(only_face, lower, upper)
    final = cv2.bitwise_and(only_face, only_face, mask = skin_mask)
    return final

def colorful_frame_procedure(face, frame):
    img = frame.copy()
    fc = face.copy()
    skin = detect_skin(fc, img)
    # cv2.imshow("color", skin)
    # cv2.waitKey(10)
    return get_sum_channels(skin)


def get_sum_channels(frame):
    channels = cv2.split(frame)
    r = cv2.sumElems(channels[0])
    g = cv2.sumElems(channels[1])
    b = cv2.sumElems(channels[2])

    rn = cv2.countNonZero(channels[0])
    gn = cv2.countNonZero(channels[1])
    bn = cv2.countNonZero(channels[2])

    if rn==0:
        rn = 1
    if gn == 0:
        gn = 1
    if bn == 0:
        bn = 1

    r = r[0]/rn
    g = g[0]/gn
    b = b[0]/gn
    return r, g, b



def get_landmarks(im, rect):
    x,y,w,h = rect
    # print(x,y,w,h)
    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def get_face_contour(landmarks):
    cut_dot_array = landmarks[0:26]

    face_con = cut_dot_array[0:16]
    brow_con = cut_dot_array[17:26]

    brow_con = np.flip(brow_con, axis = 0)
    cut_dot_array = np.concatenate((face_con, brow_con), axis = 0)

    return cut_dot_array


def fill_black_out_contours(img, true_conts, false_conts):
    true_mask = np.zeros(img.shape).astype(img.dtype)
    false_mask = np.full(img.shape, 255, img.dtype)

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
    # image, contours = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 2000:
            masked_img = face.copy()
            cv2.fillPoly(face, contours, [0, 0, 0])
            face = masked_img - face
    return face


def detect_face(image):
    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), maxSize=(550, 550))
    if len(faces) > 0:
        strt_x = faces[0][0]
        strt_y = faces[0][1]
        width = faces[0][2]
        height = faces[0][3]
        only_face = image[strt_y:strt_y+height, strt_x:strt_x+width]
        return only_face, faces[0]
    return image, []


def get_distance(vec,point,image):
    image=np.int32(image)
    distance2=np.cross([[vec]],image-point,axis=2)
    distance=np.linalg.norm(distance2,axis=2)
    return distance


def get_meanDistances2(vec,point,BGR):
    B=BGR[::3]
    G=BGR[1::3]
    R=BGR[2::3]
    BGR=np.int32([B,G,R])
    BGR=np.transpose(BGR)
    distances2=np.cross([vec],BGR-point,axis=1)
    distances=np.linalg.norm(distances2,axis=1)
    return np.mean(distances**2)


def get_baseline(BGR):
    B=BGR[::3]
    G=BGR[1::3]
    R=BGR[2::3]
    B = np.array(B)
    G = np.array(G)
    R = np.array(R)
    b=np.mean(B)
    r=np.mean(R)
    g=np.mean(G)
    indBM=np.where(B>b)[0]
    indBm=np.where(B<b)[0]
    indGM=np.where(G>g)[0]
    indGm=np.where(G<g)[0]
    indRM=np.where(R>r)[0]
    indRm=np.where(R<r)[0]
    BM=B[indBM]
    Bm=B[indBm]
    GM=G[indGM]
    Gm=G[indGm]
    RM=R[indRM]
    Rm=R[indRm]
    x0=np.mean(Bm)
    x1=np.mean(BM)
    y0=np.mean(Gm)
    y1=np.mean(GM)
    z0=np.mean(Rm)
    z1=np.mean(RM)
    vec=np.array([x1-x0,y1-y0,z1-z0])
    normale=np.linalg.norm(vec)
    vec=vec/normale
    point=[x0,y0,z0]
    return vec,point
