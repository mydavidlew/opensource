from __future__ import print_function
from __future__ import division
import argparse
import cv2 as cv
import dlib

parser = argparse.ArgumentParser(description='Detect Face using Dlib in an Image.')
parser.add_argument('--input', help='Path to input image.', default='Images/Random/people.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Load the image into an array
image = src
win = dlib.image_window()

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
print("shape=[".format(src.shape))
for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at: Left:{} Top:{} Right:{} Bottom:{}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect, dlib.rgb_pixel(0, 255, 0))
print("]")

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()
cv.destroyAllWindows()
exit(0)
