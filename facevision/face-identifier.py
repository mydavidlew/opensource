#!/usr/bin/env python
'''
identify an encoded face from live video feed based on the network that has already been trained to create 128-d embeddings on a dataset of ~3 million images.
USAGE:
    face-identifier.py [--haarcascade <Classifier/haarcascade_frontalface_alt2.xml>] [--encodings <Vectors/myface_encoded>]
'''

import face_recognition as fr
import cv2 as cv
import numpy as np
import pickle
import argparse
import os

from time import sleep

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), lineType=cv.LINE_AA)

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

def get_haarcascade():
    # find path of xml file containing haarcascade file
    # cascPathface = os.path.dirname(cv2.__file__) + "Classifier/haarcascade_frontalface_alt2.xml"
    cascPathface = os.getcwd() + args["haarcascade"] # "/Classifier/haarcascade_frontalface_alt2.xml"
    print("Classifier:", cascPathface)

    # load the harcaascade in the cascade classifier
    faceCascade = cv.CascadeClassifier(cascPathface)
    return faceCascade

def get_encodedface():
    # find path of vector file containing encoded face file
    # enbedPathface = os.path.dirname(cv2.__file__) + "Vectors/myface_encoded"
    enbedPathface = os.getcwd() + args["encodings"]  # "/Vectors/myface_encoded"
    print("EmbeddedFace:", enbedPathface)

    # load the known faces and embeddings saved in last file
    embedFace = pickle.loads(open(enbedPathface, "rb").read())
    return embedFace

def main(args):
    # load the harcaascade in the cascade classifier
    faceCascade = get_haarcascade()

    # load the known faces and embeddings saved in last file
    data = get_encodedface()

    print("Streaming started")
    video_capture = cv.VideoCapture(0)
    # loop over frames from the video file stream
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#        faces = faceCascade.detectMultiScale(gray,
#                                             scaleFactor=1.1,
#                                             minNeighbors=5,
#                                             minSize=(30, 30),
#                                             flags=cv.CASCADE_SCALE_IMAGE)

        # convert the input frame from BGR to RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Use Face_recognition to locate faces
        boxes = fr.face_locations(rgb, model="hog")
        # the facial embeddings for face in input
        encodings = fr.face_encodings(rgb, boxes)

        t = clock()
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for (encoding, box) in zip(encodings, boxes):
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely & False for rest
            matches = fr.compare_faces(data["encodings"], encoding)
            # list of values (floating point) of the facial distance, lowest value present the best or closet facial match
            fvalues = fr.face_distance(data["encodings"], encoding)
            # get the lowest facial distance value from the array object
            matchedIdx = np.argmin(fvalues)

            # set name = unknown if no encoding matches
            name = "NaN"
            if matches[matchedIdx]:
                name = data["names"][matchedIdx]

            # draw the predicted face name on the image
            y1, x2, y2, x1 = box
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(frame, name, (x1+3, y2-6), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            # display the encoding statistics
            print("Distance: {}  Pos: {}  Name: {}".format(fvalues, box, name))

        dt = clock() - t
        draw_str(frame, (10, 20), 'time: %.4f ms' % (dt * 1000))
        cv.imshow("Frame", frame)
        if cv.waitKey(3) == 27:
            break

    # free up video devices
    video_capture.release()
    print("Streaming terminated")

if __name__ == '__main__':
    print(__doc__)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--haarcascade", required=False, type=str, default="/Classifier/haarcascade_frontalface_alt2.xml",
                    help="path to input directory of haarcascade model")
    ap.add_argument("-e", "--encodings", required=False, type=str, default="/Vectors/myface_encoded",
                    help="path to embedded db of facial encodings")
    args = vars(ap.parse_args())
    # run the face identification application
    main(args)
    cv.destroyAllWindows()