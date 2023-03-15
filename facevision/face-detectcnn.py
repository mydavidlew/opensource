"""
    Created on Wed Nov 28 15:36:11 2018

    @author: keyur-r
    CNN (Convolutional Neural Network) based face detector available in dlib
    https://github.com/keyurr2/face-detection

    To find faces in image -> python face_detection_hog_svn.py -i <input-image> -w mmod_human_face_detector.dat
    To find faces realtime -> python face_detection_hog_svn.py -w mmod_human_face_detector.dat
"""
import dlib
import cv2
import argparse
import os
# face detection cnn libraries
from PIL import Image
import face_recognition


def write_to_disk(image, face_cordinates):
    '''
    This function will save the cropped image from original photo on disk
    '''
    for (x1, y1, w, h) in face_cordinates:
        cropped_face = image[y1:y1 + h, x1:x1 + w]
        cv2.imwrite(str(y1) + ".jpg", cropped_face)


def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    '''
    To draw some fancy box around founded faces in stream
    '''
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# Don't use realtime face detection using dlib's cnn trained model.
# It's really very slow but good performance over frontal face


def face_detection_realtime():
    cap = cv2.VideoCapture(0)

    while True:
        # Getting out image by webcam
        _, image = cap.read()

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = cnn_face_detector(gray, 0)

        face_cordinates = []
        # For each detected face
        for (i, rect) in enumerate(rects):
            # Finding points for rectangle to draw on face
            x1, y1, x2, y2, w, h = rect.rect.left(), rect.rect.top(), rect.rect.right() + 1, \
                rect.rect.bottom() + 1, rect.rect.width(), rect.rect.height()

            # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
            draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

            # Drawing simple rectangle around found faces
            # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            face_cordinates.append((x1, y1, w, h))

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x1 - 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 51), 2)

        # Show the image
        cv2.imshow("Output", image)

        # To capture found faces from camera
        if cv2.waitKey(30) & 0xFF == ord('s'):
            write_to_disk(image, face_cordinates)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def face_detection(image):
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces from image
    rects = cnn_face_detector(gray, 0)

    # For each detected face, draw boxes.
    for (i, rect) in enumerate(rects):
        # Finding points for rectangle to draw on face
        x1, y1, x2, y2, w, h = rect.rect.left(), rect.rect.top(), rect.rect.right() + 1, \
            rect.rect.bottom() + 1, rect.rect.width(), rect.rect.height()

        # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
        draw_fancy_box(image, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)

        # Drawing simple rectangle around found faces
        # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x1 - 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 51), 2)

    # Show the image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face_detection_cnn(image):
    # Load the jpg file into a numpy array
    imagefile = face_recognition.load_image_file(image)
    np_image = Image.fromarray(imagefile, "RGB")
    np_image.show()

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition.face_locations(imagefile, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # You can access the actual face itself like this:
        face_image = imagefile[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()


if __name__ == "__main__":
    # Please change your base path
    HOME = "/media/superadmin/Office/Workspace/Projects/Python/MyApp"

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--weights', help='Path to Weights',
                    default='Classifier/mmod_human_face_detector.dat')
    ap.add_argument('-i', '--image', required=False, help='Path to image file',
                    default="Images/Random/EXO.jpg")
    args = ap.parse_args()

    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)
    image = None
    if args.image:
        # load input image
        img = os.path.join(HOME, args.image)
        image = cv2.imread(img)

    if image is None:
        print("Real time face detection is starting ... ")
        face_detection_realtime()
    else:
        print("Face detection for image")
        face_detection(image)
        face_detection_cnn(args.image)
