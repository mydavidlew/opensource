import face_recognition as fr
import cv2 as cv
import numpy as np
import logging as log
import datetime as dt
import sys
from time import sleep

def auto_detect_canny(image, sigma):
	# compute the median
	mi = np.median(image)

	# computer lower & upper thresholds
	lower = int(max(0, (1.0 - sigma) * mi))
	upper = int(min(255, (1.0 + sigma) * mi))
	image_edged = cv.Canny(image, lower, upper)

	return image_edged

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)

video_capture = cv.VideoCapture(0)
log.basicConfig(filename='webcam.log', level=log.INFO)
anterior = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    #ret, frame = video_capture.read()
    frame = cv.imread("EXO.jpg")

    display = False
    if display:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30, 30)
                                            )
        # x, y, w, h = face
        # faces = [[275 398 463 463]]
        print("faces [{}] = {}".format(len(faces), faces))
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # green

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxes = fr.face_locations(rgb)
        # y1, x2, y2, x1 = box
        # boxes = [(451, 707, 913, 245)]
        print("boxes [{}] = {}".format(len(boxes), boxes))
        for (y1, x2, y2, x1) in boxes:
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # red

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        # Display the resulting frame
        cv.imshow('Video', frame)
        print("shape: {}".format(frame.shape))

    blurimg = True
    if  blurimg:
        img = frame
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mblur = cv.medianBlur(gray, 5)
        gblur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        medges = cv.Canny(mblur, 50, 200)
        gedges = cv.Canny(gblur, 50, 200)
        ## (2) Threshold
        th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        mth, mthreshed = cv.threshold(mblur, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        gth, gthreshed = cv.threshold(gblur, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        #print("threshed: {}".format(threshed))
        ## (3) Find the min-area contour
        _cnts = cv.findContours(threshed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        _mcnts = cv.findContours(mthreshed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        _gcnts = cv.findContours(gthreshed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
        print("median-gray: {} median-mblur: {} median-gblur: {}".format(np.median(gray), np.median(mblur), np.median(gblur)))
        print("median: {} _cnts: {} _mcnts: {} _gcnts: {}".format(np.median(img), len(_cnts), len(_mcnts), len(_gcnts)))
        cnts = sorted(_cnts, key=cv.contourArea)
        for cnt in cnts:
            n = cv.contourArea(cnt)
            ## (4) Create mask and do bitwise-op
            mask = np.zeros(gray.shape[:2], np.uint8)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            dst = cv.bitwise_and(gray, gray, mask=mask)

        #cv.imshow('gray', gray)
        #print("gray: {}".format(gray.shape))
        cv.imshow('threshed', threshed)
        print("threshed: {}".format(threshed.shape))
        cv.imshow('dst', dst)
        print("dst: {}".format(dst.shape))
        #cv.imshow('medges', medges)
        #print("medges: {}".format(medges.shape))
        #cv.imshow('mblur', mblur)
        #print("mblur: {}".format(mblur.shape))
        #cv.imshow('mthreshed', mthreshed)
        #print("mthreshed: {}".format(mthreshed.shape))
        #cv.imshow('gedges', gedges)
        #print("gedges: {}".format(gedges.shape))
        #cv.imshow('gblur', gblur)
        #print("gblur: {}".format(gblur.shape))
        #cv.imshow('gthreshed', gthreshed)
        #print("gthreshed: {}".format(gthreshed.shape))

    if cv.waitKey(0) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()





from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import sys
import dlib
from skimage import io

parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='../FaceRecog/Images/Random/EXO.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

if True:
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    # Load the image into an array
    image = src
    win = dlib.image_window()
    print("shape=".format(src.shape))

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(image, 1)

    # Open a window on the desktop showing the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at: Left:{} Top:{} Right:{} Bottom:{}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win.add_overlay(face_rect, dlib.rgb_pixel(0, 255, 0))

    # Wait until the user hits <enter> to close the window
    dlib.hit_enter_to_continue()

if False:
    bgr_planes = cv.split(src)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ), ( 255, 0, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ), ( 0, 255, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ), ( 0, 0, 255), thickness=2)

    cv.imshow('Source image', src)
    cv.imshow('calcHist Demo', histImage)
    cv.waitKey(0)

if False:
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('xfiles4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv.destroyAllWindows()





#!/usr/bin/env python

'''
face detection using haar cascades
USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    # default is scaleFactor=1.3, minNeighbors=4, minSize=(30, 30)
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "classifier/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "classifier/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('data/EXO.jpg')))
    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #gray = cv.equalizeHist(gray)
        gray = cv.medianBlur(gray, 5)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)
        if cv.waitKey(1) == 27:
            cam.release()
            break

    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
