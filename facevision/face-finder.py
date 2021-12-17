#!/usr/bin/env python
'''
identify an encoded face from live video feed based on the network that has already been trained to create 128-d embeddings on a dataset of ~3 million images.
USAGE:
    face-finder.py [--haarcascade <Classifier/haarcascade_frontalface_alt2.xml>] [--encodings <Vectors/myface_encoded>]
'''

import face_recognition as fr
import cv2 as cv
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

def main(args):
    # find path of xml file containing haarcascade file
    cascPathface = os.getcwd() + args["haarcascade"]
    enbedPathface = os.getcwd() + args["encodings"]
    print("Classifier:", cascPathface)
    print("EmbeddedFace:", enbedPathface)

    # load the harcaascade in the cascade classifier
    faceCascade = cv.CascadeClassifier(cascPathface)

    # load the known faces and embeddings saved in last file
    data = pickle.loads(open(enbedPathface, "rb").read())

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
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(30, 30),
                                             flags=cv.CASCADE_SCALE_IMAGE)

        # convert the input frame from BGR to RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Use Face_recognition to locate faces
        boxes = fr.face_locations(rgb, model="hog") #fr.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        # the facial embeddings for face in input
        encodings = fr.face_encodings(rgb, boxes)
        names = []

        t = clock()
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for (encoding, box) in zip(encodings, boxes):
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            matches = fr.compare_faces(data["encodings"], encoding)
            # set name = Unknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

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