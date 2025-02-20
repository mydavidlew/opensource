import streamlit as st
import logging, random, time, os, io

import tempfile
import timm  # Install timm: pip install timm
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from transformers import DetrFeatureExtractor, DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# --- Visualization (requires matplotlib) ---
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('tkagg')

st.set_page_config(page_title="Application #05", page_icon="🪻", layout="wide")
st.sidebar.title("🪻 Computer Vision")
st.sidebar.markdown(
    """This demo illustrates a combination of different Computer Vision Transformers architecture.
    Try with different combination of AI python frameworks with Streamlit platform. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

# COCO classes that DETR is trained on (index 0 is a placeholder)
CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

#1. Using Pretrained DETR Model from torchvision
def DETR_Torchvision():
    # Load a pre-trained DETR model
    model = torch.hub.load(repo_or_dir="facebookresearch/detr", model="detr_resnet50", pretrained=True, trust_repo=True)
    model.eval()

    # Load and preprocess an image
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()]) # [T.Resize((800, 800)), T.ToTensor()]
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Run inference to get outputs
    with torch.no_grad():
        outputs = model(img_tensor)

    # The outputs contain:
    # - "pred_logits": classification logits for each of the 100 queries
    # - "pred_boxes": bounding box predictions for each query
    logits = outputs["pred_logits"][0]  # shape: [num_queries, num_classes+1]
    boxes = outputs["pred_boxes"][0]  # shape: [num_queries, 4]

    # Convert logits to probabilities using softmax along the last dimension
    probs = logits.softmax(-1)

    # Get the highest probability and the corresponding label for each query
    scores, labels = probs.max(-1)

    # Define a threshold to filter out low-confidence predictions
    confidence_threshold = 0.7
    keep = scores > confidence_threshold

    # Filter out predictions that are below the threshold
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]
    filtered_boxes = boxes[keep]

    # Print the detected object labels and their scores
    st.subheader("1. Using Pretrained DETR Model from torchvision")
    st.write("Detected objects:")
    i = 0
    for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
        i += 1
        label_idx = label.item()  # get integer label
        class_name = CLASSES[label_idx] if label_idx < len(CLASSES) else "NULL"
        st.write(f"Label[:blue[{i}]]: :blue[{class_name}], Score: :green[{score.item():.3f}], Box: :green[{[round(i, 4) for i in box.tolist()]}]")

    # Print detected objects
    st.write("outputs: ", outputs)
    st.write("labels: ", labels)
    st.write("scores: ", scores)
    st.write("boxes: ", boxes)

#2. Using Hugging Face Transformers Library
def DETR_Transformers():
    # Load DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Load and preprocess image
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # The outputs contain:
    # - "logits": classification logits for each of the 100 queries
    # - "pred_boxes": bounding box predictions for each query
    logits = outputs.logits  # shape: [num_queries, num_classes+1]
    boxes = outputs.pred_boxes  # shape: [num_queries, 4]

    # Get predicted labels and bounding boxes (scaled to image size)
    labels = logits[0].argmax(-1) #logits.argmax(-1).tolist()[0] --> Get predicted class indices
    scores = torch.nn.functional.softmax(logits, dim=-1)[0, :, labels].max(dim=-1)[0] #.tolist() --> Get confidence scores
    boxes = boxes[0] #.tolist() --> Get bounding box coordinates (normalized)

    # Define a threshold to filter out low-confidence predictions
    confidence_threshold = 0.9
    keep = scores > confidence_threshold

    # Filter out predictions that are below the threshold
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]
    filtered_boxes = boxes[keep]

    # Print the detected object labels and their scores
    st.subheader("2. Using Hugging Face Transformers Library")
    st.write("Detected objects:")
    i = 0
    for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
        i += 1
        label_idx = label.item()  # get integer label
        class_name = CLASSES[label_idx] if label_idx < len(CLASSES) else "NULL"
        st.write(f"Label[:blue[{i}]]: :blue[{class_name}], Score: :green[{score.item():.3f}], Box: :green[{[round(i, 4) for i in box.tolist()]}]")

    # Print detected objects
    st.write("outputs: ", outputs)
    st.write("labels: ", filtered_labels)
    st.write("scores: ", filtered_scores)
    st.write("boxes: ", filtered_boxes)

#3. Using DETR with OpenCV for Real-time Webcam Detection
def DETR_Webcam():
    # Load pretrained model
    model = torch.hub.load(repo_or_dir="facebookresearch/detr", model="detr_resnet50", pretrained=True, trust_repo=True)
    model.eval()

    # Preprocessing transformation
    transform = T.Compose([T.ToTensor()])

    # Capture from webcam
    st.subheader("3. Using DETR with OpenCV for Real-time Webcam Detection")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL image and preprocess
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        # Display frame (Modify to draw boxes using OpenCV)
        cv2.imshow("DETR Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#4. Using DETR with Custom Dataset via PyTorch Training
def DETR_PyTorch():
    # Define a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, image_folder):
            self.image_folder = image_folder
            self.images = os.listdir(image_folder)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_folder, self.images[idx])
            image = Image.open(img_path).convert("RGB")
            image = F.to_tensor(image)
            return image, {}

    # Load dataset and model
    dataset = CustomDataset("temp") # path_to_your_dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = torch.hub.load(repo_or_dir="facebookresearch/detr", model="detr_resnet50", pretrained=True, trust_repo=True)
    model.eval()

    # Inference on dataset
    st.subheader("4. Using DETR with Custom Dataset via PyTorch Training")
    for images, _ in dataloader:
        with torch.no_grad():
            outputs = model(images)
        st.write("outputs: ", outputs)

#5. Deploying DETR with Flask for Web API
def DETR_WebAPI():
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    # Load model
    model = torch.hub.load(repo_or_dir="facebookresearch/detr", model="detr_resnet50", pretrained=True, trust_repo=True)
    model.eval()
    transform = T.Compose([T.ToTensor()])

    @app.route("/detect", methods=["POST"])
    def detect():
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read()))
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        return jsonify(outputs)

    if __name__ == "__main__":
        app.run()



# Load the pre-trained DETR model from torchvision
def load_detr_model():
    model = torch.hub.load(repo_or_dir='facebookresearch/detr', model='detr_resnet50', pretrained=True, trust_repo=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Standard PyTorch image transforms for DETR input
def get_transforms():
    return T.Compose([
    #   T.Resize(800),  # Resize the image to a fixed size
        T.ToTensor(),   # Convert the image to a PyTorch tensor
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

# Perform object detection using DETR
def detect_objects(model, image, transform):
    # Apply the transformation to the image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract predictions
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Class probabilities
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0], image.size)  # Rescale bounding boxes

    return probas, bboxes_scaled

# Rescale bounding boxes to the original image size
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = out_bbox * torch.tensor(data=[img_w, img_h, img_w, img_h], dtype=torch.float32)
    b = b.numpy()
    return b

# Plot the detected objects on the image
def plot_results(image, probas, bboxes, threshold=0.9):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Iterate over predictions and draw bounding boxes
    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes):
        cl = p.argmax().item()  # Get the predicted class
        if p[cl] > threshold:  # Only display objects above the confidence threshold
            label = f"{CLASSES[cl]}: {p[cl]:.2f}"
            width = xmax - xmin
            height = ymax - ymin
            ax.add_patch(plt.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth=2))
            ax.text(xmin, ymin, label, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

# Main function
def DETR_main():
    # Load the DETR model
    detr_model = load_detr_model()

    # Load an image from file
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")

    # Define the image transforms
    transform = get_transforms()

    # Perform object detection
    probas, bboxes = detect_objects(detr_model, image, transform)

    # Visualize the results
    plot_results(image, probas, bboxes)



def DETR_Visualize1():
    # DETR Object Detection (using the transformers library)
    # 1. Load Pre-trained Model and Feature Extractor
    model_name = "facebook/detr-resnet-50"  # Or "facebook/detr-swin-base" for a larger model
    feature_extractor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

    # 2. Load and Preprocess Image
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")  # PyTorch tensors

    # 3. Perform Object Detection
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-process Predictions
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    confidence_threshold = 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]
    scores = results["scores"]; labels = results["labels"]; boxes = results["boxes"]

    # 5. Visualize Results
    st.subheader("DETR (End-to-End Object Detection) model with ResNet-50 Method-1")
    plt.figure(figsize=(16, 10))  # Width, height in inches.
    plt.imshow(image)
    ax = plt.gca()
    # Object identified from image based on the selected threshold value
    for score, label, box in zip(scores, labels, boxes):
        score = round(score.item(), 3)
        label = model.config.id2label[label.item()]
        box = [round(i, 2) for i in box.tolist()]
        st.write(f"Detected :blue[{label}] with confidence :blue[{score}] at location :blue[{box}]")
        # plot the bounding box of detected objects
        x_min, y_min, x_max, y_max = box
        ax.add_patch(plt.Rectangle(xy=(x_min, y_min), width=(x_max - x_min), height=(y_max - y_min), fill=False, edgecolor='cyan', linewidth=1))
        plt.text(x_min + 2, y_max - 2, s=f"{label.upper()}: {score:.2f}", fontsize=8, fontweight='normal', color='black', bbox=dict(facecolor='yellow', alpha=0.5))
    # Overlay and display on the actual image
    plt.axis('on')
    plt.show()

def DETR_Visualize2():
    # DETR Object Detection (using the transformers library)
    # Load pre-trained DETR model and feature extractor
    model_name = "facebook/detr-resnet-50"
    feature_extractor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

    # Preprocess the image with Example image (replace with your own image)
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"  # Replace with your image path
    image = Image.open(image_path)  # Make sure you have PIL (Pillow) installed: pip install Pillow
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the predictions
    # (This part is simplified; you'll likely need to adjust it based on your needs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Extract predictions
    # Convert bounding boxes to pixel coordinates (example)
    width, height = image.size
    logits_scaled = logits.softmax(-1)[0, :, :-1]  # Class probabilities
    bboxes_scaled = bboxes[0] * torch.tensor(data=[width, height, width, height], dtype=torch.float32)
    bboxes_scaled = bboxes_scaled.numpy()

    # Print or visualize the results
    st.subheader("DETR (End-to-End Object Detection) model with ResNet-50 Method-2")
    st.write("outputs: ", outputs)
    st.write(f"logits {logits.shape}: ", logits_scaled)  # Class probabilities
    st.write(f"bboxes {bboxes.shape}: ", bboxes_scaled)  # Bounding boxes (normalized)

    # Display the image with bounding boxes
    plt.figure(figsize=(16, 10)) # Width, height in inches.
    plt.imshow(image)
    ax = plt.gca()
    threshold = 0.8
    for logit, box in zip(logits_scaled, bboxes_scaled):
        klass = logit.argmax().item()  # Get the predicted class
        if logit[klass] > threshold:  # Only display objects above the confidence threshold
            label = f"{CLASSES[klass]}: {logit[klass]:.2f}"
            x, y, w, h = box.tolist()
            x0 = x - (w/2); y0 = y - (h/2)
            ax.add_patch(patches.Rectangle(xy=(x0, y0), width=w, height=h, linewidth=1, edgecolor='cyan', facecolor='none'))
            ax.text(x0 + 2, y0 - 3, label.upper(), fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
            st.write(f"Box :blue[({label.upper()})] : :green[{[round(i, 2) for i in box.tolist()]}]")
    plt.axis('on')
    plt.show()

def DETR_Visualize3():
    # 1. Load Pre-trained Model and Feature Extractor
    model_name = "facebook/detr-resnet-50"  # Or "facebook/detr-swin-base" for a larger model
    feature_extractor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)

    # 2. Load and Preprocess Image
    image_path = "temp/5851546454_4fdd60e8d5_o.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")  # PyTorch tensors

    # 3. Perform Object Detection
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-process Predictions
    # Convert logits to probabilities and get bounding boxes
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Get predicted labels and bounding boxes (scaled to image size)
    labels = logits.argmax(-1)[0].tolist()  # Get predicted class indices
    scores = torch.nn.functional.softmax(logits, dim=-1)[0, :, labels].max(dim=-1)[0].tolist() # Get confidence scores
    boxes = boxes[0].tolist()  # Get bounding box coordinates (normalized)

    # Convert normalized boxes to pixel coordinates
    image_width, image_height = image.size
    predicted_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * image_width)
        y_min = int(y_min * image_height)
        x_max = int(x_max * image_width)
        y_max = int(y_max * image_height)
        predicted_boxes.append([x_min, y_min, x_max, y_max])

    # 5. Visualize Results
    st.subheader("DETR (End-to-End Object Detection) model with ResNet-50 Method-3")
    st.write("labels: ", labels)
    st.write("scores: ", scores)
    st.write("boxes: ", boxes)
    st.write("predicted_boxes: ", predicted_boxes)

    # Get class names (you'll need to map indices to names)
    id2label = model.config.id2label  # Dictionary mapping class indices to names
    threshold = 0.99  # Adjust confidence threshold as needed

    plt.figure(figsize=(16, 10))  # Width, height in inches.
    plt.imshow(image)
    ax = plt.gca()

    for label, score, box in zip(labels, scores, predicted_boxes):
        if score > threshold:  # Adjust confidence threshold as needed
            x_min, y_min, x_max, y_max = box
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor='green', linewidth=1))
            class_name = id2label[label] if label in id2label else f"Class {label}" # Handle cases where label is not in id2label
            plt.text(x_min, y_min - 10, f"{class_name}: {score:.2f}", fontsize=10, color='red')

    plt.axis('on')
    plt.show()

    # Example of how to save the image (optional)
    #plt.savefig("detected_objects.jpg")

def ViT_01():
    # ViT Object Detection (using a library like timm)

    # Load pre-trained ViT model (example: vit_base_patch16_224)
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()

    # Feature extraction (you can adapt this for object detection)
    image_path = "temp/new-york-city-1.jpg"  # Replace with your image path
    image = Image.open(image_path)
    preprocess = timm.data.create_transform(
        model.default_cfg, is_training=False  # Use the model's default transforms
    )
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model.forward_features(input_tensor)  # Extract features

    # Object Detection Head (you'll need to add this)
    # ViT itself doesn't directly do object detection.  You'll need to add
    # a detection head on top of the extracted features.  This could be
    # a simple linear layer or something more complex.  There are many
    # ways to design this head.  Here's a very basic example:

    num_classes = 1000  # Example number of classes (adjust as needed)
    detection_head = torch.nn.Linear(model.embed_dim, num_classes)  # Simple linear layer

    with torch.no_grad():
        detections = detection_head(features)

    print(detections.shape)  # Output of the detection head

    # Note: This is a VERY basic example, and it won't perform well
    # without proper training and a more sophisticated detection head.
    # For real object detection with ViT, look into models like
    # DETR or DINO which use Transformers for the detection task.



def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 Torchvision", "👻 Transformers", "👻 Webcam", "👻 DETRvisual", "👻 DETRothers"])
    with tab01:
        st.subheader("Using Pretrained DETR Model from torchvision")
        tab01_btn = st.button(label="Click to **Start**", key="tab01_btn")
        if tab01_btn is True:
            DETR_Torchvision()
    with tab02:
        st.subheader("Using Hugging Face Transformers Library")
        tab02_btn = st.button(label="Click to **Start**", key="tab02_btn")
        if tab02_btn is True:
            DETR_Transformers()
    with tab03:
        st.subheader("Using DETR with OpenCV for Real-time Webcam Detection")
        tab03_btn = st.button(label="Click to **Start**", key="tab03_btn")
        if tab03_btn is True:
            DETR_Webcam()
    with tab04:
        st.subheader("Using DETR End-to-End Object Detection with Transformers")
        tab04_btn = st.button(label="Click to **Start**", key="tab04_btn")
        if tab04_btn is True:
            DETR_Visualize1() # or DETR_Visualize2()
    with tab05:
        st.subheader("DEtection TRansformer (DETR) Methods")
        tab05_btn = st.button(label="Click to **Start**", key="tab05_btn")
        if tab05_btn is True:
            DETR_Visualize3()
            #DETR_main()

if __name__ == '__main__':
    st.title("Computer Vision Transformers")
    main()