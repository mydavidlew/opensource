from __future__ import print_function
from __future__ import division
import argparse
import matplotlib.pyplot as plt
import cv2 as cv
import cvlib as cvl
from cvlib.object_detection import draw_bbox

parser = argparse.ArgumentParser(description='Count Objects in an Image.')
parser.add_argument('--input', help='Path to input image.', default='Images/Random/cars.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

image = src
box, label, count = cvl.detect_common_objects(image)
output = draw_bbox(image, box, label, count)

print("Number of cars in this image are " + str(label.count('car')))
print("Number of trucks in this image are " + str(label.count('truck')))
print("Number of person in this image are " + str(label.count('person')))

plt.imshow(output)
plt.show()
