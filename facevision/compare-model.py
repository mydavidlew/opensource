import face_recognition as fr
import cv2 as cv
import logging as log
import datetime as dt
import sys
from time import sleep

cascPath1 = "Classifier/haarcascade_frontalface_default.xml"    # 1.1  / 5 / 30, 30
cascPath2 = "Classifier/haarcascade_frontalface_alt.xml"        # 1.1  / 3 / 30, 30
cascPath3 = "Classifier/haarcascade_frontalface_alt2.xml"       # 1.1  / 6 / 30, 30
cascPath4 = "Classifier/haarcascade_frontalface_alt_tree.xml"   # 1.05 / 1 / 30, 30
faceCascade = cv.CascadeClassifier(cascPath2)

video_capture = cv.VideoCapture(0)
log.basicConfig(filename='webcam.log', level=log.INFO)
anterior = 0
totalfrm = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #frame = cv.imread("Images/Random/EXO.jpg")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( gray,
                                          scaleFactor=1.05,
                                          minNeighbors=3,
                                          minSize=(30, 30)
                                        )
    # x, y, w, h = face
    # faces = [[275 398 463 463]]
    print("faces [{}] = {}".format(len(faces), faces))
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # green
        cv.putText(frame, "fc", (x+3, y+h-6), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    boxes = fr.face_locations(rgb)
    # y1, x2, y2, x1 = box
    # boxes = [(451, 707, 913, 245)]
    print("boxes [{}] = {}".format(len(boxes), boxes))
    for (y1, x2, y2, x1) in boxes:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # red
        cv.putText(frame, "fr", (x1+3, y2-6), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Video', frame)
    totalfrm += 1

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("total frames:", str(totalfrm))
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()