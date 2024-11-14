import streamlit as st

# Set page configuration at the very beginning
st.set_page_config(page_title="Women Safety Analytics", page_icon="🛡️", layout="wide")

# Continue with the rest of your imports
import torch
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from torchvision import transforms

# Load models with caching
@st.cache_resource
def load_models():
    yolo_model = YOLO('best.pt')
    resnet_model = load_model('GenderClassification.h5')
    return yolo_model, resnet_model

yolo_model, resnet_model = load_models()

# Define preprocessing for ResNet model
def preprocess_for_resnet(img):
    img = img.resize((100, 200))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Detection and classification function
def detect_and_classify(image):
    results = yolo_model.predict(source=np.array(image), save=False)
    image_np = np.array(image)
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image.crop((x1, y1, x2, y2))
            
            # Gender classification
            processed_img = preprocess_for_resnet(cropped_img)
            prediction = resnet_model.predict(processed_img)
            gender = "Male" if np.argmax(prediction) == 0 else "Female"
            
            # Draw bounding box and label
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)
    
    return image_np

# Streamlit UI
st.title("Women Safety Analytics – Protecting Women from Safety Threats")
st.markdown("""
This application uses **YOLOv8** for person detection and **ResNet50** for gender classification. 
Upload an image to analyze.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        result_image = detect_and_classify(image)

    # Display the result
    st.image(result_image, caption="Processed Image with Gender Labels", use_column_width=True)
    st.success("Detection complete!")
