#!/usr/bin/env python
'''
identify an encoded face from image files based on the network that has already been trained to create 128-d embeddings on a dataset of ~3 million images.
USAGE:
    image-finder.py [--dataset <Images/Random>] [--haarcascade <Classifier/haarcascade_frontalface_alt2.xml>] [--encodings <Vectors/myface_encoded>]
'''

import face_recognition as fr
import cv2 as cv
import argparse
import pickle
import os

from imutils import paths

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

def get_imagelists():
    imagePaths = list(paths.list_images(os.getcwd() + args["dataset"]))
    return imagePaths

def main(args):
    # load the harcaascade in the cascade classifier
    faceCascade = get_haarcascade()
    # load the known faces and embeddings saved in last file
    data = get_encodedface()

    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    imagePaths = get_imagelists()

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # convert image to Greyscale for haarcascade
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv.CASCADE_SCALE_IMAGE)

        # convert the input image from BGR to RGB
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        boxes = fr.face_locations(rgb, model="hog") #fr.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        encodings = fr.face_encodings(rgb, boxes)
        print("Found [ {0} / {1} ] faces in".format(len(encodings), len(boxes)), imagePath)

        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely & False for rest
            matches = fr.compare_faces(data["encodings"], encoding)

            # set name = unknown if no encoding matches
            name = "NaN"
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
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(image, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        scale_percent = 70  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv.resize(image,dim)
        cv.imshow("Frame", image)
        cv.waitKey(0)

if __name__ == '__main__':
    print(__doc__)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, type=str, default="/Images/Random",
                    help="path to input directory of faces + images")
    ap.add_argument("-i", "--haarcascade", required=False, type=str, default="/Classifier/haarcascade_frontalface_alt2.xml",
                    help="path to input directory of haarcascade model")
    ap.add_argument("-e", "--encodings", required=False, type=str, default="/Vectors/myface_encoded",
                    help="path to embedded db of facial encodings")
    args = vars(ap.parse_args())
    # run the face identification application
    main(args)
    cv.destroyAllWindows()