#!/usr/bin/env python3
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

def main():
    face_distance()
    identify_faces()

def face_distance():
    # Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
    # You can do that by using the face_distance function.

    # The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
    # be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
    # positive matches at the risk of more false negatives.

    # Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
    # smaller distance are more similar to each other than ones with a larger distance.

    # Load some images to compare against
    known_obama_image = face_recognition.load_image_file("Images/Single/obama.jpg")
    known_biden_image = face_recognition.load_image_file("Images/Single/biden.jpg")

    # Get the face encodings for the known images
    obama_face_encoding = face_recognition.face_encodings(known_obama_image)[0]
    biden_face_encoding = face_recognition.face_encodings(known_biden_image)[0]

    known_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]

    # Load a test image and get encondings for it
    image_to_test = face_recognition.load_image_file("Images/Single/obama2.jpg")
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

    for i, face_distance in enumerate(face_distances):
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
        print()

def identify_faces():
    # This is an example of running face recognition on a single image
    # and drawing a box around each person that was identified.

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("Images/Identity/david/IMG_20211215_135015.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("Images/Identity/darren/IMG_20210917_200703.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "DavidLew",
        "DarrenLew"
    ]
    known_face_count = len(known_face_encodings)

    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file("Images/Random/IMG_20190223_142749.jpg")

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    count = 0
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"
        count = count + 1

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Find the known faces with threshold set to 0.45 cutoff would reduce false
        # positive matches at the risk of more false negatives.
        for i in range(known_face_count):
            if face_distances[i] < 0.45:
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            print("[{}.{}] face_distances {:.3}".format(count, i, face_distances[i]), "-->", name)

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 128), outline=(0, 0, 64))
        draw.text((left + 6, bottom - text_height - 5), str(count) + " " + name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

    # You can also save a copy of the new image to disk if you want by uncommenting this line
    # pil_image.save("image_with_boxes.jpg")

if __name__ == "__main__":
    main()