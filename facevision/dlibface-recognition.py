import cv2 as cv
import numpy as np
import logging as log
import datetime as dt
import dlib
import sys
from time import sleep

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
# Load the predictor
face_predictor = dlib.shape_predictor("Classifier/shape_predictor_68_face_landmarks.dat")

drawpoint = False
video_capture = cv.VideoCapture(0)
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, image = video_capture.read()
    image = cv.imread("Images/Single/IMG_20211215_135015.jpg")
    #image = cv.imread("Images/Random/EXO.jpg")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Load the image into an array
    win = dlib.image_window()

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(gray, 1)

    # Open a window on the desktop showing the image
    #win.set_image(image)
    print("shape=".format(image.shape))

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        x1 = face_rect.left()  # left point
        y1 = face_rect.top()  # top point
        x2 = face_rect.right()  # right point
        y2 = face_rect.bottom()  # bottom point

        # Draw a box around each face we found
        #win.add_overlay(face_rect, dlib.rgb_pixel(0, 255, 0))
        cv.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)

        # Look for the landmarks
        # DLib algorithms to detect these features we actually get a map of points that surround each feature.
        # This map composed of 67 points (called landmark points) can identify the following features:
        # - Jaw Points = 0–16
        # - Right Brow Points = 17–21
        # - Left Brow Points = 22–26
        # - Nose Points = 27–35
        # - Right Eye Points = 36–41
        # - Left Eye Points = 42–47
        # - Mouth Points = 48–60
        # - Lips Points = 61–67
        landmarks = face_predictor(image=gray, box=face_rect)

        # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle
            if drawpoint:
                cv.circle(img=image, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
            else:
                cv.putText(image, str(n), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), lineType=cv.LINE_AA)

        # of the top, left, right and bottom edges
        print("Face #{} at: Left:{} Top:{} Right:{} Bottom:{}".format(i, x1, y1, x2, y2))

    # Wait until the user hits <enter> to close the window
    #dlib.hit_enter_to_continue()

    # show the image
    cv.imshow(winname="Face", mat=image)

    if cv.waitKey(delay=3) == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()