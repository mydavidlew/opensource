#!/usr/bin/env python
'''
face detection using haar cascades
USAGE:
    face-detect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), lineType=cv.LINE_AA)

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def create_capture(source = 0):
    cap = cv.VideoCapture(source)
    return cap

def detect(img, cascade):
    #rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    rects = cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects # (x1, y1, x2, y2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0

    args = dict(args)
    cascade_fn = args.get('--cascade', "Classifier/haarcascade_frontalface_alt2.xml")
    nested_fn  = args.get('--nested-cascade', "Classifier/haarcascade_eye.xml")
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    totalframe = 0
    cam = create_capture(video_src)
    print('Start video streaming')
    while True:
        t = clock()
        totalframe += 1

        _ret, img = cam.read()
        #img = cv.imread("Images/Random/EXO.jpg")

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
        for x, y, w, h in rects:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi = gray[y:y+h, x:x+w]
            subrects = nested.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            for x1, y1, w1, h1 in subrects:
                cv.rectangle(img, (x+x1, y+y1), (x+x1+w1, y+y1+h1), (255, 0, 0), 2)
                vis_roi = roi[y+y1:y+y1+h1, x+x1:x+x1+w1]

        dt = clock() - t
        cv.rectangle(img, (1, 1), (170, 25), (0, 255, 0), cv.FILLED)
        cv.putText(img, "time: %.2fms" % (dt*1000), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), thickness=2)
        cv.imshow('FaceDetect', img)

        if cv.waitKey(1) == 27:
            print('Total frames:', totalframe)
            print('Stop video streaming')
            cam.release()
            break

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
