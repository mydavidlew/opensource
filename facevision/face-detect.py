#!/usr/bin/env python
'''
face detection using haar cascades
USAGE:
    face-detect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), lineType=cv.LINE_AA)

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

def create_capture(source = 0):
    cap = cv.VideoCapture(source)
    return cap

def detect(img, cascade):
    #rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
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
    cascade_fn = args.get('--cascade', "Classifier/haarcascade_frontalface_alt2.xml")
    nested_fn  = args.get('--nested-cascade', "Classifier/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src)
    while True:
        t = clock()

        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)

        #vis = img.copy()
        #draw_rects(vis, rects, (0, 255, 0))
        draw_rects(img, rects, (0, 255, 0))

        subrects = ""
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                #subrects = detect(roi.copy(), nested)
                subrects = detect(roi, nested)

                #vis_roi = vis[y1:y2, x1:x2]
                #draw_rects(vis_roi, subrects, (255, 0, 0))
                vis_roi = img[y1:y2, x1:x2]
                draw_rects(vis_roi, subrects, (255, 0, 0))

        dt = clock() - t
        #draw_str(vis, (10, 20), 'time: %.2f ms' % (dt*1000))
        #cv.imshow('FaceDetect', vis)
        draw_str(img, (10, 20), "{0} & {1}".format(len(rects), len(subrects)) + " time: %.2f ms" % (dt * 1000))
        cv.imshow('FaceDetect', img)

        if cv.waitKey(5) == 27:
            break

    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()