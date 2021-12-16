#!/usr/bin/env python
'''
face encoding based on the network has already been trained to create 128-d embeddings on a dataset of ~3 million images.
USAGE:
    face-encoder.py [--dataset <Images/Identity>] [--encodings <Vectors/myface_encoded>] [--facemodel <hog/cnn>] [--showface <true/false>]
'''

import face_recognition as fr
import cv2 as cv
import pickle
import argparse
import os

from imutils import paths
from PIL import Image

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def main(args):
    # get paths of each file in folder named Images
    # Images here contains my data(folders of various persons)
    imagePaths = list(paths.list_images(args["dataset"]))
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # start time
        t = clock()

        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv.imread(imagePath)
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #rgb = fr.load_image_file(imagePath, mode="RGB")

        # Use Face_recognition to locate faces
        # only two face detection methods, either 'hog' or 'cnn'
        if args["facemodel"] == "hog":
            boxes = fr.face_locations(rgb, number_of_times_to_upsample=2, model=args["facemodel"])
        elif args["facemodel"] == "cnn":
            boxes = fr.face_locations(rgb, number_of_times_to_upsample=0, model=args["facemodel"])
        else:
            boxes = fr.face_locations(rgb, model="hog")

        # compute the facial embedding for the face
        encodings = fr.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

        # display the detected face
        if (args["showface"] == True):
            for box in boxes:
                # Get the location of each face in this image
                top, right, bottom, left = box
                # You can access the actual face itself like this:
                face_image = rgb[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image = pil_image.resize([200, 200])
                pil_image.show()

        # finish time
        dt = clock() - t
        w, h = getsize(rgb)
        print("ID:", name, "[{0}]:".format(i), imagePath, "w={0}px h={1}px".format(w, h), "t=%.2fs" %dt, "face={}".format(len(boxes)))

    # save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}

    # use pickle to save data into a file for later use
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Names[{0}]=".format(len(knownNames)), knownNames.__str__())

if __name__ == '__main__':
    print(__doc__)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=False, type=str, default="Images/Identity",
                    help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=False, type=str, default="Vectors/myface_encoded",
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--facemodel", required=False, type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    ap.add_argument("-s", "--showface", required=False, type=bool, default=False,
                    help="display the face detection images")
    args = vars(ap.parse_args())
    # run the face identification application
    main(args)
    cv.destroyAllWindows()
