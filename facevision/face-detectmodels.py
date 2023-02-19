from __future__ import print_function
from __future__ import division
import argparse
import dlib
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

parser = argparse.ArgumentParser(description='Face Detection Models.')
parser.add_argument('--input', help='Path to input image.', default='Images/Random/EXO.jpg')
args = parser.parse_args()

src = cv2.imread(cv2.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

def haar_cascade():
    print('Face detection models = Haar Cascade')
    classifier = cv2.CascadeClassifier('Classifier/haarcascade_frontalface_default.xml')
    img = src
    faces = classifier.detectMultiScale(img) # result
    # to draw faces on image
    for result in faces:
        x, y, w, h = result
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow('Haar Cascade', img)
    cv2.waitKey(0)

def dlib_hog():
    print('Face detection models = Dlib HOG')
    detector = dlib.get_frontal_face_detector()
    img = src
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # result
    # to draw faces on image
    for result in faces:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow('Dlib HOG', img)
    cv2.waitKey(0)

def mtcnn():
    print('Face detection models = MTCNN')
    detector = MTCNN()
    img = src
    faces = detector.detect_faces(img)  # result
    # to draw faces on image
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow('MTCNN', img)
    cv2.waitKey(0)

def dnn():
    print('Face detection models = DNN') # https://github.com/keyurr2/face-detection
    modelFile = "Classifier/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "Classifier/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    img = src
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    # to draw faces on image
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow('DNN', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    print(__doc__)
    # Face Detection Models: Which to Use and Why?
    # https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
    haar_cascade()
    dlib_hog()
    mtcnn()
    dnn()
    cv2.destroyAllWindows()