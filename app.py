import streamlit as st  
import torch  
import cv2  
import numpy as np  
from PIL import Image  

# Load the YOLOv5 model (change 'yolov5s' to the appropriate model if needed)  
@st.cache_resource  
def load_model():  
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  
    return model  

model = load_model()  

def detect_objects(image):  
    # Convert the image to a format compatible with YOLOv5  
    result = model(image)  
    return result  

st.title("YOLOv5 Object Detection")  

# File uploader for images  
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  

if uploaded_file is not None:  
    # Read the image using PIL  
    image = Image.open(uploaded_file)  

    # Display the image  
    st.image(image, caption='Uploaded Image', use_column_width=True)  

    # Convert to OpenCV format (RGB to BGR)  
    img = np.array(image)[:, :, ::-1]  

    # Perform object detection  
    results = detect_objects(img)  

    # Render results (the image with bounding boxes)  
    st.image(results.render()[0], caption='Detection Results', use_column_width=True)