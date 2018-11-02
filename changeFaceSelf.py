#!/usr/bin/python

import cv2 as cv
import dlib
import numpy as np
from hist_match import histMatch

import sys

PREDICTOR_PATH = 'model/shape_predictor_68_face_landmarks.dat'
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

TRANSFORM_POINT = [17,26,57]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im ,winname = 'debug'):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    draw = im.copy()
    for _, d in enumerate(rects):
        cv.rectangle(draw,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),3)

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv.putText(im, str(idx), pos,
                    fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv.convexHull(points) # 得到凸包
    cv.fillConvexPoly(im, points, color=color) # 绘制填充


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0)) #-> rgb rgbr rgb

    im = (cv.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


# 读取图片文件并获取特征点
def read_im_and_landmarks(fname):
    im = cv.imread(fname, cv.IMREAD_COLOR)

    im = cv.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))

    # 68个特征点
    s = get_landmarks(im,fname) # mat

    return im, s


def warp_im(mask, M, dshape):
    output_im = np.zeros(dshape, dtype=mask.dtype)
    cv.warpAffine(mask,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv.BORDER_TRANSPARENT,
                   flags=cv.WARP_INVERSE_MAP)
    return output_im


def getAffineTransform(_srcPoint,_dstPoint):
    srcPoint = _srcPoint.astype(np.float32)
    dstPoint = _dstPoint.astype(np.float32)
    return cv.getAffineTransform(srcPoint,dstPoint)


# im1 贴到im2上
im1, landmarks1 = read_im_and_landmarks(sys.argv[2])
im2, landmarks2 = read_im_and_landmarks(sys.argv[1])

cv.imshow('face1', im1)
cv.imshow('face2', im2)


# 变换矩阵
M = getAffineTransform(landmarks1[TRANSFORM_POINT],landmarks2[TRANSFORM_POINT])

mask = get_face_mask(im2, landmarks2)

warped_mask = warp_im(mask, M, im1.shape)


combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],
                          axis=0)  # 两张图片mask并集

warped_im2 = warp_im(im2, M, im1.shape)

histMatch_im = histMatch(warped_im2.astype(np.uint8),im1,mask=combined_mask)

output_im_hist = im1 * (1.0 - combined_mask) + histMatch_im * combined_mask

output_im_hist = output_im_hist.astype(np.uint8)

cv.imshow('changeface', output_im_hist)
cv.waitKey()