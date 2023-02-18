import face_recognition as fr
import cv2 as cv
import numpy as np
import logging as log
import datetime as dt
from time import sleep

def auto_detect_canny(image, sigma):
	# compute the median
	mi = np.median(image)

	# computer lower & upper thresholds
	lower = int(max(0, (1.0 - sigma) * mi))
	upper = int(min(255, (1.0 + sigma) * mi))
	image_edged = cv.Canny(image, lower, upper)

	return image_edged

cascPath = "Classifier/haarcascade_frontalface_default.xml"
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
    ret, frame = video_capture.read()
    frame = cv.imread("Images/Random/EXO.jpg")

    display = True
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

        cv.imshow('gray', gray)
        print("gray: {}".format(gray.shape))
        cv.imshow('threshed', threshed)
        print("threshed: {}".format(threshed.shape))
        cv.imshow('dst', dst)
        print("dst: {}".format(dst.shape))
        cv.imshow('medges', medges)
        print("medges: {}".format(medges.shape))
        cv.imshow('mblur', mblur)
        print("mblur: {}".format(mblur.shape))
        cv.imshow('mthreshed', mthreshed)
        print("mthreshed: {}".format(mthreshed.shape))
        cv.imshow('gedges', gedges)
        print("gedges: {}".format(gedges.shape))
        cv.imshow('gblur', gblur)
        print("gblur: {}".format(gblur.shape))
        cv.imshow('gthreshed', gthreshed)
        print("gthreshed: {}".format(gthreshed.shape))

    if cv.waitKey(0) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()
