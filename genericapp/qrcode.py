# Import necessary libraries
import os
import cv2
import base64
import qrcode
import numpy as np
from pyzbar import pyzbar
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding


def generate_qr_code(data, file_name="qr_code.png"):
    """
    Generates a QR code from the given data and saves it as an image file.

    :param data: The data to encode in the QR code (e.g., text, URL).
    :param file_name: The name of the output file (default: "qr_code.png").
    """
    # Create a QRCode object
    # Parameters:
    #   version: Controls the size of the QR code (1 is the smallest, 40 is the largest). Set to None to let the library automatically #determine the size.
    #   error_correction: Specifies the error correction level:
    #      ERROR_CORRECT_L: About 7% of the data can be recovered.
    #      ERROR_CORRECT_M: About 15% of the data can be recovered.
    #      ERROR_CORRECT_Q: About 25% of the data can be recovered.
    #      ERROR_CORRECT_H: About 30% of the data can be recovered.
    #   box_size: Size of each "box" in pixels.
    #   border: Thickness of the border around the QR code (minimum value is 4).
    qr = qrcode.QRCode(
        version=3,  # Controls the size of the QR Code (1 is the smallest, 40 is the largest)
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each "box" in pixels
        border=4,  # Border thickness (minimum value is 4)
    )

    # Add the data to the QRCode object
    qr.add_data(data)
    qr.make(fit=True)  # Automatically adjust the size to fit the data

    # Create an image from the QRCode object
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the image to a file
    img.save(file_name)
    #img.show(file_name)
    print(f"Data to Encode: {data}")
    print(f"QR Code Image: {file_name}")

def read_qr_code(image_path="qr_code.png"):
    """
    Reads a QR code from the given image file and returns the decoded data.

    :param image_path: Path to the image file containing the QR code.
    :return: Decoded data from the QR code.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at path {image_path}. Please check the file path.")
        return None

    # Convert the image to grayscale (optional but improves processing)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect and decode QR codes using pyzbar
    qr_codes = pyzbar.decode(gray_image)

    # Check if any QR codes were found
    if not qr_codes:
        print("No QR code detected in the image.")
        return None

    # Extract and print data from each QR code
    for qr_code in qr_codes:
        # Extract the data and type of the QR code
        data = qr_code.data.decode('utf-8')  # Decode bytes to string
        qr_type = qr_code.type

        # Print the results
        print(f"QR Code Type: {qr_type}")
        print(f"QR Code Data: {data}")

        # Optionally, draw a rectangle around the QR code in the image
        points = qr_code.polygon
        if len(points) == 4:  # Ensure it's a valid quadrilateral
            pts = [(point.x, point.y) for point in points]
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the image with the QR code highlighted
    cv2.imshow("QR Code", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_qr_code_from_camera():
    """
    Reads QR codes from a live camera feed and displays the decoded data.
    """
    # Initialize the camera (use 0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit the program.")

    while True:
        # Capture a single frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from the camera.")
            break

        # Convert the frame to grayscale (optional but improves processing)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and decode QR codes using pyzbar
        qr_codes = pyzbar.decode(gray_frame)

        # Process each detected QR code
        for qr_code in qr_codes:
            # Extract the data and type of the QR code
            data = qr_code.data.decode('utf-8')  # Decode bytes to string
            qr_type = qr_code.type

            # Print the results
            print(f"QR Code Type: {qr_type}")
            print(f"Decoded Data: {data}")
            print(f"Base64 Data: {decode_base64(data)}")

            # Optionally, draw a rectangle around the QR code in the frame
            points = qr_code.polygon
            if len(points) == 4:  # Ensure it's a valid quadrilateral
                pts = [(point.x, point.y) for point in points]
                pts = [(int(x), int(y)) for x, y in pts]  # Convert to integers
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Display the decoded data on the frame
            text_position = (pts[0][0], pts[0][1] - 10)  # Position above the QR code
            cv2.putText(frame, data, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame with the highlighted QR code
        cv2.imshow("QR Code Scanner", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def encode_base64(input_data):
    """
    Encodes data to Base64 format.

    :param input_text: The input data (string or bytes) to encode.
    :return: The Base64-encoded string.
    """
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')  # Convert string to bytes
    encoded_data = base64.b64encode(input_data)
    output_data = encoded_data.decode('utf-8')  # Convert bytes to string
    return output_data

def decode_base64(base64_data, output_file=None):
    """
    Decodes a Base64-encoded string and optionally saves the result to a file.

    :param base64_data: The Base64-encoded string to decode.
    :param output_file: Optional file path to save the decoded data (e.g., for binary files).
    :return: The decoded data as bytes or a string.
    """
    try:
        # Decode the Base64 data
        decoded_data = base64.b64decode(base64_data)

        # If an output file is specified, save the decoded data to the file
        if output_file:
            with open(output_file, 'wb') as file:
                file.write(decoded_data)
            print(f"Decoded data saved to {output_file}")

        # Try to decode the bytes to a UTF-8 string (for text data)
        try:
            decoded_string = decoded_data.decode('utf-8')
            return decoded_string
        except UnicodeDecodeError:
            print("Decoded Binary Data (cannot be converted to text):")
            return decoded_data

    except Exception as e:
        print(f"Error decoding Base64 data: {e}")
        return None

def generate_key():
    """
    Generates a random 32-byte (256-bit) AES key.
    """
    return os.urandom(32)

def encrypt_aes(key, plaintext):
    """
    Encrypts plaintext using AES-256 in CBC mode.

    :param key: The AES key (must be 32 bytes for AES-256).
    :param plaintext: The plaintext data to encrypt (string or bytes).
    :return: A tuple of (iv, ciphertext), where iv is the initialization vector.
    """
    # Ensure plaintext is in bytes
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')

    # Generate a random 16-byte initialization vector (IV)
    iv = os.urandom(16)

    # Create a Cipher object using the key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Create an encryptor object
    encryptor = cipher.encryptor()

    # Pad the plaintext to match the block size (128 bits for AES)
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()

    # Encrypt the padded plaintext
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    return iv, ciphertext

def decrypt_aes(key, iv, ciphertext):
    """
    Decrypts ciphertext using AES-256 in CBC mode.

    :param key: The AES key (must be 32 bytes for AES-256).
    :param iv: The initialization vector used during encryption.
    :param ciphertext: The encrypted data to decrypt.
    :return: The decrypted plaintext as a string.
    """
    # Create a Cipher object using the key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Create a decryptor object
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the plaintext
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    # Convert bytes to string
    return plaintext.decode('utf-8')





# Main function to execute the QR code reader
if __name__ == "__main__":
    data = "Malaysian Immigration Department (JIM) records 13,846 users have used the application and 117,000 users have downloaded it overall."
    # Call the function to generate the QR code
    generate_qr_code(data)
    # Call the function to read the QR code
    read_qr_code()
    # Read the QR code capture from camera
    read_qr_code_from_camera()

    plaintext = "Malaysian Immigration Department"
    print(f"Plain Text: {plaintext}")
    base64_data = encode_base64(input_data=plaintext)
    print(f"Base64 encode: {base64_data}")
    decode_base64 = decode_base64(base64_data)
    print(f"Base64 decode: {decode_base64}")

    # Generate a random AES key
    key = generate_key()
    print(f"AES Key (hex): {key.hex()}")
    # Define plaintext data
    plaintext = "This is a secret message!"
    print(f"Original Plaintext: {plaintext}")
    # Encrypt the plaintext
    iv, ciphertext = encrypt_aes(key, plaintext)
    print(f"Ciphertext inHex: {ciphertext.hex()}")
    # Decrypt the ciphertext
    decrypted_text = decrypt_aes(key, iv, ciphertext)
    print(f"Decrypted Text: {decrypted_text}")
